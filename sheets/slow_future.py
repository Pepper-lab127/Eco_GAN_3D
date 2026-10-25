"""Sheet 07 - The slow future: the land tilting under the river.

Below St. Andrews Lock and Dam the Red River lies within about half a metre
of Lake Winnipeg's level, so its water rises with the lake. The land is still
rebounding from the weight of the Laurentide ice sheet, faster to the north
and east, so the basin tilts: Lake Winnipeg's outlet, about 250 km down the
tilt from Lockport, rises faster than Lockport, and the lake's water - and
with it the lower Red - creeps up the land here.

  - tilt: 1e-9 radians a year rising toward a bearing of 43 deg (lake-gauge
    evidence, Tackman and others 1999), projected onto the line from
    Lockport to the lake's outlet at Warren Landing: about 0.25 mm a year.
    Submerged trees on the south shore record a faster rise, about 60 cm in
    300 years (2 mm a year; Manitoba Geological Survey), drawn as the upper
    rate. The years of each future water edge are given for both.
  - future water edges: the lidar below the dam flooded through connected
    ground at +0.25, +0.5, +1 and +2 m over the lidar's water level; dots
    on the land each rise drowns, denser where it drowns sooner.
  - the bank itself: sections across the river comparing the 2014 (MB-Cooks
    Creek) and 2020 (MB-Red River) lidar surveys. They agree to a few
    centimetres on stable ground and the banks moved less than a metre in
    six years: no measurable retreat to project.

    python -m sheets.slow_future
"""

import argparse
import math
import os

import numpy as np
from scipy import ndimage

from drawing import terrain
from drawing.section import Section
from drawing.svg import PAPER, Svg, north_arrow, points_to_path, scale_bar
from sheets.calendar_cut import true_north_angle

INK = "#111111"


def build(args):
    # Rate of rise here relative to the lake's outlet, from the tilt.
    lat0, lon0 = args.site
    lat1, lon1 = args.outlet
    dn = (lat1 - lat0) * 111.2
    de = (lon1 - lon0) * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))
    along = dn * math.cos(math.radians(args.tilt_bearing)) + de * math.sin(math.radians(args.tilt_bearing))
    rate_low = args.tilt * along * 1000 * 1000          # mm/yr
    rate_high = args.shore_rate

    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    st = args.cell
    g = dem[r0:r1:st, c0:c1:st]
    H, W = g.shape
    dam = (args.dam_row - r0) // st
    _, w_up, w_dn = terrain.river_reaches(g, dam, args.water_below, min_cells=150, keep_near=10)
    barrier = np.zeros(g.shape, bool)
    barrier[max(dam - 1, 0):dam + 2, (args.dam_cols[0] - c0) // st:(args.dam_cols[1] - c0) // st] = True
    floodable = np.where(barrier | w_up, np.inf, g)
    z_water = float(np.nanmedian(g[w_dn]))
    edges = {}
    for rise in args.rises:
        edges[rise] = terrain.flood(floodable, z_water + rise, w_dn)
    top = max(args.rises)
    # Rise needed to drown each cell (finely), for the dots.
    thr = np.full(g.shape, np.inf)
    for lv in np.arange(0.05, top + 1e-6, 0.05):
        wet = terrain.flood(floodable, z_water + lv, w_dn)
        thr[wet & np.isinf(thr) & ~w_dn] = lv

    # ------------------------------------------------------------ sheet
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale
    x0, y0 = m + 30.0, m + 40.0
    pw, ph = (c1 - c0) * s, (r1 - r0) * s
    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(contour="PLAN · CONTOURS 1 M · 0.035 mm", contour5="PLAN · CONTOURS 5 M · 0.05 mm",
             water="PLAN · RIVER TODAY (LIDAR) · 0.12 mm", dots=f"PLAN · LAND DROWNED BY +{top:g} M, DOTS · {args.dot} mm",
             edges="PLAN · FUTURE WATER EDGES · 0.06–0.2 mm", dam="PLAN · DAM AND LOCK · 0.35 mm",
             cuts="PLAN · SECTION LINES · 0.1 mm", s14="SECTIONS · GROUND 2014, DASHED · 0.15 mm",
             s20="SECTIONS · GROUND 2020 · 0.15 mm", sw="SECTIONS · WATER TODAY AND FUTURE · 0.06–0.2 mm",
             key="KEY TO WATER EDGES · 0.06–0.2 mm", scale="SCALE BARS AND NORTH ARROW", text="TEXT")
    for name in L.values():
        svg.layer(name)
    clip = svg.clip_rect(x0, y0, pw, ph)

    def plan(key):
        svg.layer(L[key])
        svg.open_group(clip_path=clip)

    for lv, lines in terrain.smooth_contours(dem[r0:r1:2, c0:c1:2], np.arange(216, 233, 1.0)).items():
        index = lv % 5 == 0
        plan("contour5" if index else "contour")
        svg.path(" ".join(points_to_path(x0 + ln[::2, 0] * 2 * s, y0 + ln[::2, 1] * 2 * s) for ln in lines if len(ln) > 8),
                 fill="none", stroke="#a6a6a6", stroke_width=0.05 if index else 0.035, opacity=0.6 if index else 0.4)
        svg.close_group()

    rng = np.random.default_rng(args.seed)
    soon = np.where(np.isfinite(thr), 1 - thr / (top + 0.25), 0.0)
    n = rng.poisson(soon * (st * st) / args.m2_per_dot)
    rr, cc = np.nonzero(n)
    reps = n[rr, cc]
    ry = np.repeat(rr, reps) + rng.random(reps.sum())
    rx = np.repeat(cc, reps) + rng.random(reps.sum())
    plan("dots")
    svg.path("".join(f"M{x0 + xx * st * s:.2f},{y0 + yy * st * s:.2f}h0" for xx, yy in zip(rx, ry)), fill="none",
             stroke=INK, stroke_width=args.dot, stroke_linecap="round", opacity=0.85)
    svg.close_group()

    def outline(mask, key, **style):
        path = " ".join(points_to_path(x0 + (ln[:, 0] + 0.5) * st * s, y0 + (ln[:, 1] + 0.5) * st * s)
                        for ln in terrain.mask_lines(ndimage.binary_opening(mask, iterations=1)) if len(ln) > 6)
        plan(key)
        svg.path(path, fill="none", stroke=INK, **style)
        svg.close_group()

    outline(w_dn | w_up, "water", stroke_width=0.12)
    styles = {}
    for k, rise in enumerate(args.rises):
        styles[rise] = dict(stroke_width=round(0.06 + 0.045 * k, 3),
                            stroke_dasharray=["0.15 0.5", "0.8 0.5", "2 0.6", None][min(k, 3)],
                            stroke_linecap="round")
        outline(edges[rise] & ~w_dn, "edges", **{k_: v for k_, v in styles[rise].items() if v is not None})
    plan("dam")
    yd = y0 + (args.dam_row - r0) * s
    svg.line(x0 + (args.dam_cols[0] - c0) * s, yd, x0 + (args.dam_cols[1] - c0) * s, yd, stroke=INK, stroke_width=0.35)
    svg.close_group()

    svg.layer(L["text"])
    for text, row, col, ang in args.label:
        lr, lc = float(row) - r0, float(col) - c0
        if 0 <= lr < r1 - r0 and 0 <= lc < c1 - c0:
            svg.text(x0 + lc * s, y0 + lr * s + 0.6, text, size=1.7, fill=INK, anchor="middle", spacing=0.2,
                     rotate=float(ang) or None, opacity=0.8)
    svg.layer(L["scale"])
    scale_bar(svg, x0, y0 + ph + 6, 200 * s, 200, "m", color=INK)
    north_arrow(svg, x0 + 200 * s + 12, y0 + ph + 6.5, r=2.5, color=INK,
                angle=true_north_angle(transform, crs, (r0 + r1) / 2, (c0 + c1) / 2))

    # ------------------------------------------------------------ sections
    d14, _, _ = terrain.load_dem(args.dem_2014)
    d20, _, _ = terrain.load_dem(args.dem_2020)
    sx = x0 + pw + 28
    sy = y0 + 4
    gap = args.sec_gap
    ca, cb = args.sec_cols
    for k, row in enumerate(args.sec_rows):
        z14 = d14[row, ca:cb].astype(float)
        z20 = d20[row, ca:cb].astype(float)
        both = np.isfinite(z14) & np.isfinite(z20)
        z_top = args.sec_ztop
        sec = Section(sx, sy + k * gap, args.sec_h, args.sec_v, z_top)
        chn = np.arange(cb - ca, dtype=float)
        svg.layer(L["s14"])
        svg.path(sec.path(chn, np.where(both, z14, np.nan)), fill="none", stroke=INK, stroke_width=0.15,
                 stroke_dasharray="0.8 0.5")
        svg.layer(L["s20"])
        svg.path(sec.path(chn, np.where(both, z20, np.nan)), fill="none", stroke=INK, stroke_width=0.15)
        # Water today and the future levels, between the banks.
        svg.layer(L["sw"])
        ground = np.where(np.isfinite(z20), z20, z14)
        for rise in [0.0] + list(args.rises):
            lv = z_water + rise
            wet = np.nan_to_num(ground, nan=1e9) <= lv
            lab, nl = ndimage.label(wet)
            if not nl:
                continue
            biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            idx = np.nonzero(lab == biggest)[0]
            st_ = styles.get(rise, dict(stroke_width=0.2))
            svg.line(sec.X(idx.min()), sec.Y(lv), sec.X(idx.max() + 1), sec.Y(lv), stroke=INK,
                     **{k_: v for k_, v in st_.items() if v is not None})
        diff = np.abs(z20 - z14)[both & (ground > z_water + 1.5)]
        svg.layer(L["text"])
        svg.text(sec.X(0), sec.Y(z_top) - 1.5, f"{k + 1}" + (f" · 2014–2020 MEDIAN CHANGE ON THE BANKS "
                 f"{np.median(diff) * 100:.0f} CM" if len(diff) else ""), size=1.6, fill=INK, spacing=0.15)
        # The cut on the plan.
        if r0 <= row < r1:
            plan("cuts")
            yc = y0 + (row - r0) * s
            svg.line(x0 + (ca - c0) * s, yc, x0 + (cb - c0) * s, yc, stroke=INK, stroke_width=0.1,
                     stroke_dasharray="1.5 0.6 0.3 0.6")
            svg.close_group()
            svg.layer(L["text"])
            svg.text(x0 + (cb - c0) * s + 1.5, yc + 0.6, f"{k + 1}", size=1.8, fill=INK)
            svg.text(sec.X(cb - ca) + 2, sec.Y(z_water) + 0.6, f"{k + 1}", size=2.2, fill=INK)
    svg.layer(L["scale"])
    yb = sy + len(args.sec_rows) * gap + 2
    scale_bar(svg, sx, yb, 100 * 1000 / args.sec_h, 100, "m", color=INK)
    svg.line(sx + 100 * 1000 / args.sec_h + 25, yb + 0.8, sx + 100 * 1000 / args.sec_h + 25,
             yb + 0.8 - 5 * 1000 / args.sec_v, stroke=INK, stroke_width=0.15)
    svg.text(sx + 100 * 1000 / args.sec_h + 27, yb + 0.8, "5 m", size=1.8, fill=INK)

    # Key: each future edge with the years it takes at the two rates.
    kx, ky = sx, yb + 18
    svg.layer(L["text"])
    svg.text(kx, ky - 4, f"WATER BELOW THE DAM TODAY +{z_water:.2f} M · RISE RELATIVE TO THE LAND", size=1.8, fill=INK,
             spacing=0.15)
    for k, rise in enumerate(args.rises):
        yk = ky + k * 5
        svg.layer(L["key"])
        st_ = styles[rise]
        svg.line(kx, yk, kx + 14, yk, stroke=INK, **{k_: v for k_, v in st_.items() if v is not None})
        svg.layer(L["text"])
        y_hi = rise * 1000 / rate_high
        y_lo = rise * 1000 / rate_low
        svg.text(kx + 17, yk + 0.6, f"+{rise:g} M · IN ABOUT {y_hi:,.0f} YEARS AT {rate_high:g} MM/YR, "
                 f"{y_lo:,.0f} YEARS AT {rate_low:.2f} MM/YR", size=1.6, fill=INK)

    annot(m + 8, m + 12, "RED RIVER AT LOCKPORT — THE SLOW FUTURE", size=5.2, fill=INK, spacing=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"distance along tilt {along:.0f} km -> {rate_low:.3f} mm/yr; upper {rate_high} mm/yr; water {z_water:.2f}; "
          f"dots {reps.sum():,}; drowned at +{top} m: {np.isfinite(thr).sum() * st * st / 1e4:.1f} ha")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dem-2014", default="data/lockport/dem/lockport_MB-Cooks_Creek_dtm_1m.tif")
    ap.add_argument("--dem-2020", default="data/lockport/dem/lockport_MB-Red_River_2020_dtm_1m.tif")
    ap.add_argument("--site", type=float, nargs=2, default=[50.083, -96.935], help="lat lon of the dam")
    ap.add_argument("--outlet", type=float, nargs=2, default=[53.70, -97.85],
                    help="lat lon of Lake Winnipeg's outlet (Warren Landing)")
    ap.add_argument("--tilt", type=float, default=1e-9, help="tilt rate, radians per year (Tackman et al. 1999)")
    ap.add_argument("--tilt-bearing", type=float, default=43.0)
    ap.add_argument("--shore-rate", type=float, default=2.0, help="mm/yr: 60 cm in 300 years on the south shore")
    ap.add_argument("--rises", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--window", type=int, nargs=4, default=[1150, 2250, 900, 2300])
    ap.add_argument("--cell", type=int, default=2)
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--m2-per-dot", type=float, default=2.5)
    ap.add_argument("--dot", type=float, default=0.18)
    ap.add_argument("--scale", type=int, default=3000)
    ap.add_argument("--sec-rows", type=int, nargs="+", default=[1300, 1500, 1700, 1840, 2050])
    ap.add_argument("--sec-cols", type=int, nargs=2, default=[1150, 1750])
    ap.add_argument("--sec-h", type=int, default=2000)
    ap.add_argument("--sec-v", type=int, default=250)
    ap.add_argument("--sec-ztop", type=float, default=232.0)
    ap.add_argument("--sec-gap", type=float, default=68.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"))
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--annotate", action="store_true", help="also print a title")
    ap.add_argument("--out", default="outputs/sheets/07_slow_future.svg")
    args = ap.parse_args()
    if args.label is None:
        args.label = [("LOCK", 2200, 1272, 0), ("FLOODWAY OUTLET", 1395, 2010, 0), ("RED RIVER", 1600, 1400, -70)]
    build(args)


if __name__ == "__main__":
    main()
