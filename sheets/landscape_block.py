"""Sheet 08 - The making of Lockport: a cut terrain block and how it was laid down.

A block of the ground around St. Andrews Lock and Dam, cut out and drawn in
oblique projection: its top surface as profile lines from the 1 m lidar,
its south and east faces cut through the layers the two drillers' logs
found. Three small blocks before it show how the layers came to be:

  1. about 450 million years ago (Late Ordovician): a shallow tropical sea
     lays down the lime mud of the Red River Formation - the limestone and
     dolostone under everything here.
  2. the last glaciation, ending about 12,000 years ago: the Laurentide ice
     sheet flows south over the site, grinding the bedrock and spreading a
     carbonate-rich till.
  3. about 12,000 to 8,000 years ago: glacial Lake Agassiz covers the site,
     and its still water settles clay over the till, a layer a year.
  then: the lake drains about 8,000 years ago and the Red River cuts down
  through the clay and till; at Lockport it reached the limestone and
  tumbled over it as the St. Andrews Rapids, where the dam stands now.

The surface before the river (blocks 2 and 3) is the lidar with the valley
filled in (a morphological closing over the river), an approximation.
Layer depths are the two logs, interpolated; thicknesses are exaggerated
vertically as stated.

    python -m sheets.landscape_block
"""

import argparse
import math
import os

import numpy as np
from scipy import ndimage

from drawing import geology, terrain
from drawing.section import Section
from drawing.svg import PAPER, Svg, points_to_path, scale_bar

INK = "#111111"


class Block:
    """Oblique projection of a DEM block: east to the right, north up-right
    at `angle` foreshortened by `depth_k`, heights exaggerated `ve` times."""

    def __init__(self, window, scale, ve, z_bottom, x0, y_base, angle=30.0, depth_k=0.5):
        self.r0, self.r1, self.c0, self.c1 = window
        self.s = 1000.0 / scale
        self.ve = ve
        self.zb = z_bottom
        self.x0, self.yb = x0, y_base
        self.kx = depth_k * math.cos(math.radians(angle))
        self.ky = depth_k * math.sin(math.radians(angle))

    def P(self, row, col, z):
        row, col, z = np.broadcast_arrays(np.asarray(row, float), np.asarray(col, float), np.asarray(z, float))
        back = (self.r1 - np.asarray(row, float)) * self.s
        X = self.x0 + (np.asarray(col, float) - self.c0) * self.s + back * self.kx
        Y = self.yb - (np.asarray(z, float) - self.zb) * self.s * self.ve - back * self.ky
        return X, Y

    def front(self, z_top):
        """Section for the south face (row r1), in sheet mm."""
        return Section(self.x0, self.yb - (z_top - self.zb) * self.s * self.ve, 1000.0 / self.s,
                       1000.0 / (self.s * self.ve), z_top)

    def east_matrix(self):
        """SVG matrix mapping a section drawn with x0 = 0 along the east face
        (chainage = metres north of the front) onto the block."""
        X0, _ = self.P(self.r1, self.c1, self.zb)
        return f"matrix({self.kx:.5f},{-self.ky:.5f},0,1,{X0:.3f},0)"


def surface(svg, layer_fill, layer_line, blk, z, rows_step, water=None, line_w=0.06, ink=INK):
    """Top surface as west-east profiles, back to front, each hiding what
    lies behind it (white fill down to the block's base)."""
    r0, r1, c0, c1 = blk.r0, blk.r1, blk.c0, blk.c1
    cols = np.arange(c0, c1, max(1, rows_step // 2))
    for row in range(r0, r1 + 1, rows_step):
        rr = min(row, r1 - 1)
        zz = z[rr - r0, cols - c0]
        X, Y = blk.P(rr, cols, zz)
        Xb, Yb = blk.P(rr, cols[[-1, 0]], blk.zb)
        poly = "M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in zip(X, Y)) + \
            f"L{Xb[0]:.2f},{Yb[0]:.2f}L{Xb[1]:.2f},{Yb[1]:.2f}Z"
        svg.layer(layer_line)
        svg.path(poly, fill="#ffffff", stroke="none")
        if water is not None:
            wet = water[rr - r0, cols - c0]
            svg.path(points_to_path(X, np.where(wet, np.nan, Y)), fill="none", stroke=ink, stroke_width=line_w,
                     opacity=0.9)
            svg.path(points_to_path(X, np.where(wet, Y, np.nan)), fill="none", stroke=ink, stroke_width=line_w * 0.6,
                     stroke_dasharray="0.3 0.6", opacity=0.7)
        else:
            svg.path(points_to_path(X, Y), fill="none", stroke=ink, stroke_width=line_w, opacity=0.9)


def faces(svg, L, blk, plain, cap, z_top, ink=INK, label=True, seed=7):
    """South and east cut faces with the logged layers, hung from the plain
    (the surface before the river) and cut down to `cap` (the surface of the
    time drawn: today's ground, or an earlier layer's top)."""
    r0, r1, c0, c1 = blk.r0, blk.r1, blk.c0, blk.c1
    cols = np.arange(c0, c1)
    g_front = cap[r1 - 1 - r0, :]
    p_front = plain[r1 - 1 - r0, :]

    def ground_at(c):
        return float(plain[r1 - 1 - r0, int(np.clip(c - c0, 0, c1 - c0 - 1))])

    sec = blk.front(z_top)
    names = {k: L[f"geo_{k}"] for k in geology.LAYERS}
    geology.draw_layers(svg, sec, cols - c0, cols, p_front, ground_at, blk.zb, ink, seed=seed, weight=0.07,
                        top_cap=g_front, layer_names=names, label=label)
    svg.layer(L["edges"])
    svg.path(sec.path(cols - c0, g_front), fill="none", stroke=ink, stroke_width=0.25)
    # East face: chainage northward from the front, drawn flat then sheared on.
    rows = np.arange(r1 - 1, r0 - 1, -1)
    g_east = cap[rows - r0, c1 - 1 - c0]
    p_east = plain[rows - r0, c1 - 1 - c0]
    ch = (r1 - 1 - rows).astype(float)
    east = Section(0.0, sec.y0, 1000.0 / blk.s, 1000.0 / (blk.s * blk.ve), z_top)
    # One layer for the whole east face, so its shear group opens and closes there.
    svg.layer(L["east"])
    svg.open_group(transform=blk.east_matrix())
    geology.draw_layers(svg, east, ch, np.full(len(ch), c1 - 1), p_east,
                        lambda c: float(plain[r1 - 1 - r0, c1 - 1 - c0]), blk.zb, ink, seed=seed + 1,
                        weight=0.07, top_cap=g_east, layer_names=None, label=False)
    svg.path(east.path(ch, g_east), fill="none", stroke=ink, stroke_width=0.25)
    svg.close_group()
    # Block edges.
    svg.layer(L["edges"])
    for a, b in (((r1, c0), (r1, c1)), ((r1, c1), (r0, c1))):
        X, Y = blk.P([a[0], b[0]], [a[1], b[1]], [blk.zb, blk.zb])
        svg.line(X[0], Y[0], X[1], Y[1], stroke=ink, stroke_width=0.25)
    for r, c in ((r1, c0), (r1, c1), (r0, c1)):
        zt = cap[min(r, r1 - 1) - r0, min(c, c1 - 1) - c0]
        X, Y = blk.P([r, r], [c, c], [blk.zb, zt])
        svg.line(X[0], Y[0], X[1], Y[1], stroke=ink, stroke_width=0.25)


def build(args):
    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    g = dem[r0:r1, c0:c1].astype(float)
    g = np.where(np.isfinite(g), g, np.nanmin(g))
    dam = args.dam_row - r0
    _, w_up, w_dn = terrain.river_reaches(g, dam, args.water_below, min_cells=500, keep_near=30)
    water = w_up | w_dn
    # The plain before the river: the valley filled by a closing wide enough
    # to bridge it, then smoothed.
    plain = ndimage.grey_closing(g[::4, ::4], size=(args.fill_m // 4, args.fill_m // 4))
    plain = ndimage.gaussian_filter(plain, 3)
    plain = np.kron(plain, np.ones((4, 4)))[:g.shape[0], :g.shape[1]]
    plain = np.maximum(plain, g)
    cols = np.arange(c0, c1)
    tops = geology.interfaces(cols, lambda c: float(dem[r1 - 1, int(c)]))
    # Each layer's top over the block (logs vary west-east only), hung from the plain.
    depth_till = (plain[-1] - tops["till"])[None, :] * np.ones((g.shape[0], 1))
    depth_carb = (plain[-1] - tops["carbonate"])[None, :] * np.ones((g.shape[0], 1))
    till_top = plain - depth_till
    carb_top = plain - depth_carb

    Wp, Hp = PAPER[args.paper]
    m = 14.0
    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(surf="SURFACE · PROFILES (LIDAR) WITH HIDDEN-LINE FILL · 0.05–0.06 mm",
             geo_clay="FACES · LAKE AGASSIZ CLAY, VARVES · 0.07 mm", geo_till="FACES · TILL, CLASTS · 0.07 mm",
             geo_shale="FACES · SHALE, PARTINGS · 0.07 mm", geo_carbonate="FACES · CARBONATE, BEDS, JOINTS · 0.07 mm",
             east="FACES · EAST FACE, ALL LAYERS · 0.07–0.25 mm",
             above="ABOVE · SEA, ICE, LAKE · 0.05–0.2 mm", edges="BLOCK EDGES AND FACE TOPS · 0.25 mm",
             leaders="CALLOUT LINES · 0.06 mm", scale="SCALE BARS", text="TEXT")
    for name in L.values():
        svg.layer(name)

    # Layout: the block today in the middle, its labels in a column at left;
    # the three stages stacked at right, their labels in a column beside them.
    # Every label sits level with what it names, joined by one straight line.
    def wrap(text, width):
        out, line = [], ""
        for w in text.split(" "):
            if line and len(line) + 1 + len(w) > width:
                out.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        return out + [line]

    def callout(text, col_x, ty, tx, side, width, size=1.7):
        """Label lines left-aligned at col_x, the first level with ty; a
        straight horizontal line from the label to the target x."""
        lines = wrap(text, width)
        svg.layer(L["text"])
        for j, ln in enumerate(lines):
            svg.text(col_x, ty + 0.6 + j * size * 1.45, ln, size=size, fill=INK, spacing=0.12)
        longest = max(len(ln) for ln in lines) * (size * 0.62 + 0.12)
        svg.layer(L["leaders"])
        if side == "left":                         # label at left, target to its right
            svg.line(col_x + longest + 2.0, ty, tx - 1.0, ty, stroke=INK, stroke_width=0.06)
        elif side == "right":                      # label at right, target to its left
            svg.line(tx + 1.0, ty, col_x - 2.0, ty, stroke=INK, stroke_width=0.06)
        if side != "below":
            svg.circle(tx, ty, 0.35, fill=INK, stroke="none")

    s_big = 1000.0 / args.scale
    big = Block(args.window, args.scale, args.ve, args.z_bottom, m + args.left_col, args.big_base)

    # ------------------------------------------------------------ the three stages
    small_scale = args.small_scale
    sw = (c1 - c0) * 1000.0 / small_scale
    row_w = sw * (1 + 0.45)
    gap = (Wp - 2 * m - args.left_col - 3 * row_w) / 2
    stages = [
        ("1", "ABOUT 450 MILLION YEARS AGO · A SHALLOW SEA LAYS DOWN LIME MUD", carb_top, "sea"),
        ("2", "THE LAST GLACIATION, TO ABOUT 12,000 YEARS AGO · ICE FLOWS SOUTH, SPREADING TILL", till_top, "ice"),
        ("3", "ABOUT 12,000–8,000 YEARS AGO · LAKE AGASSIZ SETTLES CLAY OVER THE TILL", plain, "lake"),
    ]
    zb_small = args.z_bottom
    label_y = args.stage_base + args.stage_label_gap
    for k, (num, text, top_surface, above) in enumerate(stages):
        bx = m + args.left_col + k * (row_w + gap)
        blk = Block(args.window, small_scale, args.ve, zb_small, bx, args.stage_base)
        surface(svg, None, L["surf"], blk, top_surface, args.small_rows, line_w=0.05)
        faces(svg, L, blk, plain, top_surface, args.z_top, label=False, seed=10 + k)
        svg.layer(L["above"])
        Xa, Ya = blk.P(blk.r1, blk.c0, float(np.nanmax(top_surface)))
        Xb, Yb = blk.P(blk.r0, blk.c1, float(np.nanmax(top_surface)))
        if above == "sea":
            for j in range(5):
                yy = min(Ya, Yb) - 4 - j * 3.2
                svg.line(Xa, yy, Xb, yy, stroke=INK, stroke_width=0.05, stroke_dasharray="0.4 1.0", opacity=0.7)
        elif above == "ice":
            for j in range(9):
                yy = min(Ya, Yb) - 2 - j * 2.4
                xs = np.linspace(Xa, Xb, 30)
                ys = yy + 0.6 * np.sin(xs / 6 + j)
                svg.path(points_to_path(xs, ys), fill="none", stroke=INK, stroke_width=0.12 if j % 3 == 0 else 0.05)
            for j in range(3):
                ax, ay = Xa + (Xb - Xa) * (0.25 + 0.25 * j), min(Ya, Yb) - 10
                svg.path(f"M{ax + 5:.2f},{ay - 3:.2f}L{ax:.2f},{ay:.2f}M{ax:.2f},{ay:.2f}l1.6,-0.1M{ax:.2f},{ay:.2f}l0.6,-1.4",
                         fill="none", stroke=INK, stroke_width=0.15)
        else:
            for j in range(7):
                yy = min(Ya, Yb) - 1.5 - j * 2.6
                svg.line(Xa - 2, yy, Xb + 2, yy, stroke=INK, stroke_width=0.06, opacity=0.8 - 0.08 * j)
        # Label under the block, from its left edge; a straight vertical line
        # from the middle of the block's front face down to the label.
        cx_ = bx + sw / 2
        callout(f"{num} · {text}", bx, label_y, cx_, "below", args.stage_wrap)
        svg.layer(L["leaders"])
        svg.line(cx_, args.stage_base + 1.0, cx_, label_y - 3.0, stroke=INK, stroke_width=0.06)

    # ------------------------------------------------------------ the block today
    s = s_big
    surface(svg, None, L["surf"], big, g, args.rows_step, water=water, line_w=0.06)
    faces(svg, L, big, plain, g, args.z_top, label=False)
    left_x = m + 2
    # The block as a whole: at its back-left top corner.
    tx_, ty_ = big.P(r0, c0, float(g[0, 0]))
    callout("4 · TODAY · THE RED RIVER HAS CUT THROUGH THE CLAY AND TILL; AT THE DAM IT CROSSED THE LIMESTONE "
            "AS THE ST. ANDREWS RAPIDS", left_x, float(ty_), float(tx_), "left", args.left_wrap)
    # Each layer: where it first shows on the front face from the west, half way through it.
    tops = geology.interfaces(cols, lambda c: float(plain[-1, int(c) - c0]))
    cap = g[-1]
    for i, kind in enumerate(geology.LAYERS):
        upper = np.minimum(cap, tops[kind])
        lower = tops[geology.LAYERS[i + 1]] if i + 1 < len(geology.LAYERS) else np.full_like(upper, args.z_bottom)
        lower = np.maximum(lower, args.z_bottom)
        thick = upper - np.minimum(lower, upper)
        present = np.nonzero(thick > 0.5)[0]
        if not len(present):
            continue
        k = int(present[0])
        tx_, ty_ = big.P(r1, cols[k], (upper[k] + max(lower[k], args.z_bottom)) / 2)
        callout(geology.NAMES[kind], left_x, float(ty_), float(tx_) + (0.0 if k else 0.0), "left", args.left_wrap)

    svg.layer(L["scale"])
    fx, fy = big.P(r1, c0, args.z_bottom)
    scale_bar(svg, fx, fy + 6, 200 * s, 200, "m", color=INK)
    vx = fx + 200 * s + 25
    svg.line(vx, fy + 6.8, vx, fy + 6.8 - 10 * s * args.ve, stroke=INK, stroke_width=0.15)
    svg.text(vx + 2, fy + 6.8, f"10 m (heights ×{args.ve:g})", size=1.8, fill=INK)

    annot(m + 8, m + 12, "THE MAKING OF LOCKPORT", size=5.2, fill=INK, spacing=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--window", type=int, nargs=4, default=[1500, 2700, 1000, 2200])
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--scale", type=int, default=2500)
    ap.add_argument("--small-scale", type=int, default=10000)
    ap.add_argument("--ve", type=float, default=8.0, help="vertical exaggeration")
    ap.add_argument("--z-top", type=float, default=240.0)
    ap.add_argument("--z-bottom", type=float, default=195.0)
    ap.add_argument("--left-col", type=float, default=96.0, help="mm of label column left of the large block")
    ap.add_argument("--left-wrap", type=int, default=44, help="characters per line, left labels")
    ap.add_argument("--stage-wrap", type=int, default=46, help="characters per line, stage labels")
    ap.add_argument("--stage-base", type=float, default=548.0, help="mm: front-bottom edge of the stage blocks")
    ap.add_argument("--stage-label-gap", type=float, default=12.0, help="mm from the stage blocks to their labels")
    ap.add_argument("--big-base", type=float, default=410.0, help="mm: front-bottom edge of the large block")
    ap.add_argument("--rows-step", type=int, default=8, help="m between surface profiles (large block)")
    ap.add_argument("--small-rows", type=int, default=30)
    ap.add_argument("--fill-m", type=int, default=400, help="m width bridged when filling the valley")
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="outputs/sheets/08_making_of_lockport.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
