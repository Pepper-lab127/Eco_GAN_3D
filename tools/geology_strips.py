"""Geology strips for the four sides of each 1:500 site model: print, cut out,
and glue around the model so its edges show what lies under the ground.

Each strip is at 1:500, true scale, the length of its side (800 or 440 mm):

  - its top edge follows the model's stepped card profile along that edge:
    with 3 mm card (1.5 m) cut at 218, 219.5 ... m, the model's surface is
    the highest layer present, so the base board's top stands for 216.5 m
  - below it, the layers from the two well logs (Lake Agassiz clay, till,
    shale, carbonate bedrock), interpolated west-east, hung from the lidar
    ground, down to --z-bottom (205 m: a base board plus plinth about 23 mm
    deep under the card)
  - where a side crosses the river: the riverbed (chart soundings) and the
    water above it, hatched
  - each strip reads as seen standing outside that side: the south side runs
    west to east, the north side east to west, the west side north to south,
    the east side south to north

    python -m tools.geology_strips [--out outputs/site_models/geology_strips.svg]
"""

import argparse
import math
import os

import numpy as np
import rasterio
from scipy import ndimage

from drawing import geology
from drawing.section import Section
from drawing.svg import Svg, points_to_path

INK = "#111111"
MODELS = [
    ("A", "THE BRIDGE ON ST. ANDREWS DAM", (2030, 2250, 1245, 1645)),
    ("B", "THE FLOODWAY OUTLET", (1300, 1520, 1490, 1890)),
    ("C", "OLD HENDERSON HIGHWAY", (1705, 1925, 1560, 1960)),
]
SIDES = {"S": "SOUTH SIDE · SEEN FROM THE SOUTH · W → E", "N": "NORTH SIDE · SEEN FROM THE NORTH · E → W",
         "W": "WEST SIDE · SEEN FROM THE WEST · N → S", "E": "EAST SIDE · SEEN FROM THE EAST · S → N"}


class Strips:
    def __init__(self, dtm, bed, base=218.0, layer_m=1.5, scale=500, z_bottom=205.0, water_max=220.3, smooth=2.0):
        self.dtm, self.bed = dtm, bed
        self.base, self.layer_m, self.scale, self.z_bottom = base, layer_m, scale, z_bottom
        self.water_max, self.smooth = water_max, smooth
        self.mm = 1000.0 / scale

    def edge(self, win, side):
        """Rows, cols, real ground, the model's stepped surface and the bed along one side, in viewing order."""
        r0, r1, c0, c1 = win
        p = 10
        zs = ndimage.gaussian_filter(np.nan_to_num(self.dtm[r0 - p:r1 + p, c0 - p:c1 + p],
                                                   nan=np.nanmedian(self.dtm[r0:r1, c0:c1])), self.smooth)[p:-p, p:-p]
        if side == "S":
            rows, cols, z = np.full(c1 - c0, r1 - 1), np.arange(c0, c1), zs[-1, :]
        elif side == "N":
            rows, cols, z = np.full(c1 - c0, r0), np.arange(c0, c1)[::-1], zs[0, ::-1]
        elif side == "W":
            rows, cols, z = np.arange(r0, r1), np.full(r1 - r0, c0), zs[:, 0]
        else:
            rows, cols, z = np.arange(r0, r1)[::-1], np.full(r1 - r0, c1 - 1), zs[::-1, -1]
        n = np.where(z >= self.base, np.floor((z - self.base) / self.layer_m) + 1, 0)
        top = self.base - self.layer_m + n * self.layer_m           # the model's surface along the edge
        bed = self.bed[rows, cols] if self.bed is not None else np.full_like(z, np.nan)
        return rows, cols, z, top, bed

    def size(self, win, side):
        rows, cols, z, top, bed = self.edge(win, side)
        return len(rows) * self.mm, (float(top.max()) - self.z_bottom) * self.mm

    def draw(self, svg, x0, y0, key, title, win, side, layers, seed=7):
        """Draw one strip with its top-left at (x0, y0). Returns (width, height) in mm."""
        rows, cols, z, top, bed = self.edge(win, side)
        ch = np.arange(len(rows), dtype=float)
        z_top = float(top.max())
        sec = Section(x0, y0, self.scale, self.scale, z_top)
        river = z < self.water_max
        cap = np.where(river, np.where(np.isfinite(bed), np.minimum(bed, top), top - 2.0), top)
        row_ref = int(np.median(rows))
        geology.draw_layers(svg, sec, ch, cols.astype(float), top, lambda c: float(self.dtm[row_ref, int(c)]),
                            self.z_bottom, INK, seed=seed, weight=0.08, top_cap=cap, label=True,
                            layer_names={k: layers[f"geo_{k}"] for k in geology.LAYERS})
        svg.layer(layers["water"])
        if river.any():
            d = []
            for zz in np.arange(math.floor(np.nanmin(cap[river])), z_top + 0.01, 0.4):
                seg = np.where(river & (cap <= zz) & (top >= zz), zz, np.nan)
                d.append(sec.path(ch, seg))
            svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.06, opacity=0.7)
        # The model's surface: stepped top, cut line around the strip.
        svg.layer(layers["cut"])
        xs, ys = [], []
        for i in range(len(ch)):
            xa, xb = sec.X(ch[i]), sec.X(ch[i] + 1)
            y = float(sec.Y(top[i]))
            xs += [xa, xb]
            ys += [y, y]
        outline = (f"M{xs[0]:.2f},{sec.Y(self.z_bottom):.2f}" + "".join(f"L{a:.2f},{b:.2f}" for a, b in zip(xs, ys))
                   + f"L{xs[-1]:.2f},{sec.Y(self.z_bottom):.2f}Z")
        svg.path(outline, fill="none", stroke=INK, stroke_width=0.25)
        # Base board top (216.5 m) and the real ground, for reference.
        svg.layer(layers["ref"])
        yb = float(sec.Y(self.base - self.layer_m))
        svg.line(sec.X(0), yb, sec.X(ch[-1] + 1), yb, stroke=INK, stroke_width=0.12, stroke_dasharray="2.5 1")
        svg.path(sec.path(ch + 0.5, z), fill="none", stroke=INK, stroke_width=0.08, opacity=0.6,
                 stroke_dasharray="0.6 0.6")
        svg.layer(layers["text"])
        W, H = len(ch) * self.mm, (z_top - self.z_bottom) * self.mm
        svg.text(x0, y0 + H + 4.2, f"{key}{side} · {key} {title} · {SIDES[side]} · 1:{self.scale}", size=2.0, fill=INK,
                 spacing=0.25)
        svg.text(x0 + W + 1.2, yb + 0.6, f"BASE BOARD TOP {self.base - self.layer_m:g} M", size=1.4, fill=INK)
        for zz in range(int(math.ceil(self.z_bottom / 5) * 5), int(z_top) + 1, 5):
            svg.text(x0 - 1.2, float(sec.Y(zz)) + 0.5, f"{zz}", size=1.4, fill=INK, anchor="end")
        return W, H


def build(a):
    dtm = rasterio.open(a.dtm).read(1).astype(float)
    dtm[dtm < -1000] = np.nan
    bed = rasterio.open(a.bed).read(1).astype(float) if os.path.exists(a.bed) else None
    if bed is not None:
        bed[bed < -1000] = np.nan
    S = Strips(dtm, bed, a.base, a.layer_m, a.scale, a.z_bottom)
    Wp, Hp = a.sheet
    svg = Svg(Wp, Hp)
    L = dict(geo_clay="GEOLOGY · LAKE AGASSIZ CLAY", geo_till="GEOLOGY · TILL", geo_shale="GEOLOGY · SHALE",
             geo_carbonate="GEOLOGY · CARBONATE BEDROCK", water="WATER · HATCH", ref="BASE BOARD TOP, LIDAR GROUND · DASHED",
             cut="CUT LINES · 0.25 mm", text="TEXT")
    for v in L.values():
        svg.layer(v)
    m, gap = 30.0, 14.0
    y = m + 6
    for key, title, win in MODELS:
        # Row 1: south; row 2: north; row 3: west and east side by side.
        for row in (("S",), ("N",), ("W", "E")):
            x = m + 8
            hmax = 0.0
            for side in row:
                w, h = S.draw(svg, x, y, key, title, win, side, L, seed=hash(key + side) % 1000)
                x += w + 40
                hmax = max(hmax, h)
            y += hmax + gap
        y += 6
    svg.layer(L["text"])
    svg.text(m + 8, m, "GEOLOGY STRIPS FOR THE MODEL SIDES · 1:500, TRUE SCALE · PRINT AT 100%, CUT ON THE SOLID LINE",
             size=3.0, fill=INK, spacing=0.35)
    svg.text(Wp - m, Hp - m + 10, "BELOW THE CARD: BASE BOARD + PLINTH TO 205 M = "
             f"{(a.base - a.layer_m - a.z_bottom) * 1000 / a.scale:.0f} MM", size=2.4, fill=INK, anchor="end", spacing=0.3)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    svg.save(a.out)
    print(f"wrote {a.out}; content to y = {y:.0f} mm of {Hp:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--base", type=float, default=218.0)
    ap.add_argument("--layer-m", type=float, default=1.5)
    ap.add_argument("--scale", type=int, default=500)
    ap.add_argument("--z-bottom", type=float, default=205.0)
    ap.add_argument("--sheet", type=float, nargs=2, default=[1117.6, 762.0], help="mm (44 x 30 in)")
    ap.add_argument("--out", default="outputs/site_models/geology_strips.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
