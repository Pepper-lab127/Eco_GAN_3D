"""Transparent PNGs of the site for layering in Illustrator / InDesign:

  - topography: contour lines from the 1 m lidar surface model (DSM: ground
    plus trees and buildings), every 1 m, every 5th heavier
  - grid: lines every 100 m on the lidar's own coordinate grid (NRCan
    Lambert, EPSG:3979), so they fall on round coordinates

Both cover the same extent at the same pixel size (default 0.25 m per pixel),
so they stack exactly. Their dpi is set so that placed at 100% they print
at 1:--scale (0.25 m per px at 1:5,000 = 508 dpi). Lines are antialiased by drawing at --ss times the
size and averaging down; the PNG is black ink with transparency.

    python -m tools.topo_png [--px 0.25] [--out outputs/png]
"""

import argparse
import os

import contourpy
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None


def render(size, lines, widths_px, ss, tile=2048, ink=(17, 17, 17)):
    """lines: list of (N, 2) arrays in final pixel coords; widths_px per line.
    Returns an RGBA image."""
    W, H = size
    alpha = np.zeros((H, W), np.uint8)
    boxes = np.array([[l[:, 0].min(), l[:, 1].min(), l[:, 0].max(), l[:, 1].max()] for l in lines])
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            tw, th = min(tile, W - tx), min(tile, H - ty)
            pad = max(widths_px) + 2
            hit = np.nonzero((boxes[:, 2] >= tx - pad) & (boxes[:, 0] <= tx + tw + pad) &
                             (boxes[:, 3] >= ty - pad) & (boxes[:, 1] <= ty + th + pad))[0]
            if not len(hit):
                continue
            im = Image.new("L", (tw * ss, th * ss), 0)
            d = ImageDraw.Draw(im)
            for i in hit:
                p = (lines[i] - [tx, ty]) * ss
                w = max(1, int(round(widths_px[i] * ss)))
                d.line([tuple(q) for q in p], fill=255, width=w, joint="curve")
                if w > 2:   # round ends
                    for q in (p[0], p[-1]):
                        d.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=255)
            small = im.resize((tw, th), Image.BOX)
            alpha[ty:ty + th, tx:tx + tw] = np.asarray(small)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = ink
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--px", type=float, default=0.25, help="metres per pixel")
    ap.add_argument("--interval", type=float, default=1.0, help="m between contours")
    ap.add_argument("--major", type=int, default=5, help="every n-th contour heavier")
    ap.add_argument("--smooth", type=float, default=1.0, help="m, Gaussian smoothing of the DSM (0 = none)")
    ap.add_argument("--minor-w", type=float, default=0.30, help="m on the ground (0.06 mm at 1:5,000)")
    ap.add_argument("--major-w", type=float, default=0.60, help="m on the ground")
    ap.add_argument("--grid", type=float, default=100.0, help="m between grid lines")
    ap.add_argument("--grid-w", type=float, default=0.50, help="m on the ground")
    ap.add_argument("--ss", type=int, default=4, help="supersampling")
    ap.add_argument("--scale", type=float, default=5000, help="print scale the PNG's dpi is set for (placed at 100%%)")
    ap.add_argument("--out", default="outputs/png")
    a = ap.parse_args()

    src = rasterio.open(a.dsm)
    z = src.read(1).astype(float)
    if src.nodata is not None:
        z[z == src.nodata] = np.nan
    fill = np.where(np.isfinite(z), z, np.nanmean(z))
    if a.smooth:
        fill = ndimage.gaussian_filter(fill, a.smooth / src.res[0])
    rows, cols = z.shape
    k = src.res[0] / a.px                         # px per raster cell
    W, H = int(round(cols * k)), int(round(rows * k))
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.px:g}m-per-px".replace(".", "p")
    dpi = 25.4 / (a.px * 1000 / a.scale)          # so 100% in Illustrator / InDesign = 1:scale

    # ---------------------------------------------------------- topography
    gen = contourpy.contour_generator(z=np.where(np.isfinite(z), fill, np.nan))
    lines, widths = [], []
    lo = np.ceil(np.nanmin(fill) / a.interval) * a.interval
    for level in np.arange(lo, np.nanmax(fill), a.interval):
        heavy = int(round(level / a.interval)) % a.major == 0
        for ln in gen.lines(level):
            if len(ln) < 2:
                continue
            lines.append((ln + 0.5) * k)          # cell centres
            widths.append((a.major_w if heavy else a.minor_w) / a.px)
    topo = render((W, H), lines, widths, a.ss)
    f_topo = os.path.join(a.out, f"lockport_dsm_contours_{a.interval:g}m_{tag}.png")
    topo.save(f_topo, dpi=(dpi, dpi))

    # ---------------------------------------------------------- grid
    left, bottom, right, top = src.bounds
    glines = []
    for e in np.arange(np.ceil(left / a.grid) * a.grid, right + 1e-6, a.grid):
        x = (e - left) / a.px
        glines.append(np.array([[x, 0], [x, H]]))
    for n in np.arange(np.ceil(bottom / a.grid) * a.grid, top + 1e-6, a.grid):
        y = (top - n) / a.px
        glines.append(np.array([[0, y], [W, y]]))
    grid = render((W, H), glines, [a.grid_w / a.px] * len(glines), a.ss)
    f_grid = os.path.join(a.out, f"lockport_grid_{a.grid:g}m_{tag}.png")
    grid.save(f_grid, dpi=(dpi, dpi))

    print(f"{W} x {H} px, {a.px} m per px; extent {right - left:.0f} x {top - bottom:.0f} m; "
          f"{len(lines)} contour lines; {len(glines)} grid lines")
    print(f_topo)
    print(f_grid)


if __name__ == "__main__":
    main()
