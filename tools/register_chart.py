"""Fit a chart screenshot onto the lidar grid by matching water edges.

Finds the similarity transform (scale, rotation, shift) taking chart pixels
to DEM (row, col) that best lays the chart's blue water edge on the lidar's
water edge at summer pool, starting from a rough guess.

    python tools/register_chart.py IMAGE --guess S DROW DCOL X0 Y0
"""

import argparse
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

sys.path.insert(0, ".")
from drawing import terrain  # noqa: E402


def chart_water_edge(path, x_max):
    im = np.array(Image.open(path).convert("RGB")).astype(int)
    r, b = im[..., 0], im[..., 2]
    blue = (b > 180) & (b - r > 60)
    blue[:, x_max:] = False                      # leave out the floodway
    blue = ndimage.binary_closing(blue, iterations=3)   # close over printed numbers
    lab, n = ndimage.label(blue)
    sizes = ndimage.sum(blue, lab, range(1, n + 1))
    keep = np.isin(lab, np.flatnonzero(sizes > 20000) + 1)
    edge = keep & ~ndimage.binary_erosion(keep)
    ys, xs = np.nonzero(edge)
    return xs.astype(float), ys.astype(float), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--guess", type=float, nargs=5, required=True, metavar=("S", "ROW", "COL", "X0", "Y0"),
                    help="m per pixel, and DEM row/col of chart pixel (X0, Y0)")
    ap.add_argument("--x-max", type=int, default=1340, help="chart pixels right of this are ignored")
    ap.add_argument("--levels", type=float, nargs=2, default=[222.9, 218.2], help="pool above / below dam")
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--out", default="data/lockport/bathymetry/chart_transform.json")
    args = ap.parse_args()

    dem, _, _ = terrain.load_dem(args.dem)
    rows = np.arange(dem.shape[0])[:, None]
    _, wu, wd = terrain.river_reaches(dem, args.dam_row, 223.2)
    up = terrain.flood(np.where(rows > args.dam_row, dem, np.inf), args.levels[0], wu)
    dn = terrain.flood(np.where(rows < args.dam_row, dem, np.inf), args.levels[1], wd)
    water = up | dn
    dem_edge = water & ~ndimage.binary_erosion(water)
    dist = ndimage.distance_transform_edt(~dem_edge)

    xs, ys, _ = chart_water_edge(args.image, args.x_max)
    s0, row0, col0, x0, y0 = args.guess

    def to_dem(p, x, y):
        s, th, dr, dc = p
        c, sn = np.cos(th), np.sin(th)
        col = col0 + dc + s * (c * (x - x0) - sn * (y - y0))
        row = row0 + dr + s * (sn * (x - x0) + c * (y - y0))
        return row, col

    def cost(p):
        row, col = to_dem(p, xs, ys)
        ok = (row >= 0) & (row < dist.shape[0] - 1) & (col >= 0) & (col < dist.shape[1] - 1)
        d = np.full(len(xs), 60.0)
        d[ok] = ndimage.map_coordinates(dist, [row[ok], col[ok]], order=1)
        return float(np.mean(np.minimum(d, 60.0)))

    best = None
    for th0 in np.radians([-3, 0, 3]):
        start = np.array([s0, th0, 0.0, 0.0])
        steps = np.diag([0.05, np.radians(2), 20.0, 20.0])
        res = optimize.minimize(cost, start, method="Nelder-Mead",
                                options=dict(xatol=1e-3, fatol=1e-3, maxiter=4000,
                                             initial_simplex=np.vstack([start, start + steps])))
        if best is None or res.fun < best.fun:
            best = res
    s, th, dr, dc = best.x
    out = dict(scale_m_per_px=s, rotation_deg=float(np.degrees(th)), row0=row0 + dr, col0=col0 + dc, x0=x0, y0=y0,
               mean_edge_misfit_m=best.fun)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
