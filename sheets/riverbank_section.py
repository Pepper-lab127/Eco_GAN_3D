"""Sheet 04b - Riverbank section.

A tall section through the wooded east bank of the Red River, on the same
line as the winter section (about 300 m below St. Andrews Lock and Dam), at
true scale (no vertical exaggeration):

  - sky: every day's strongest gust recorded at Winnipeg airport 2002-2008
    (ECCC daily data, gusts over about 31 km/h), each a streak whose length
    is the gust's west-east component, a dot at its downwind end
  - riparian wood: canopy heights from the lidar (DSM minus DTM) stippled,
    trunks at each crown top
  - roots: a branching DIAGRAM below each trunk, sized from the tree's height
  - water: every open-water day's level below the dam (05OJ010, 2002-2008)
  - lake clay, till and limestone aquifer: awaiting well logs

    python -m sheets.riverbank_section
"""

import argparse
import datetime as dt
import math
import os

import numpy as np

from drawing import climate, geology, hydrometric, terrain
from drawing.keyplan import key_plan
from drawing.section import Section, datum_grid, profile
from drawing.svg import PAPER, Svg, scale_bar


INK = "#111111"


def crown_tops(height, min_height, spacing):
    """Indices of local maxima of canopy height at least `spacing` m apart."""
    order = np.argsort(-np.nan_to_num(height))
    tops = []
    for i in order:
        if not height[i] >= min_height:
            break
        if all(abs(i - j) >= spacing for j in tops):
            tops.append(int(i))
    return sorted(tops)


def roots(rng, x, z, depth, spread, level=0, max_level=4):
    """Branching root DIAGRAM: list of (x0, z0, x1, z1, level) segments."""
    segs = []
    n = 3 if level == 0 else 2
    for k in range(n):
        ang = (k - (n - 1) / 2) * (1.1 if level == 0 else 0.8) + rng.normal(0, 0.25)
        length = (depth if level == 0 else depth * 0.6) * rng.uniform(0.6, 1.0)
        x1 = x + math.sin(ang) * spread / (level + 1) * rng.uniform(0.5, 1.0)
        z1 = z - abs(math.cos(ang)) * length
        segs.append((x, z, x1, z1, level))
        if level < max_level:
            segs += roots(rng, x1, z1, depth * 0.55, spread * 0.7, level + 1, max_level)
    return segs


def build(args):
    dtm, key_transform, key_crs = terrain.load_dem(args.dem)
    dsm, _, _ = terrain.load_dem(args.dsm)
    ch, ground = profile(dtm, args.row, *args.cols)
    _, top = profile(dsm, args.row, *args.cols)
    height = np.where((top - ground > 1.5) & (top - ground < 45), top - ground, np.nan)
    width = len(ch)

    Wp, Hp = PAPER[args.paper]
    m = 14.0
    sec = Section(m + 40, m + 70, args.scale, args.scale, args.z_top)
    svg = Svg(Wp, Hp)
    L = dict(frame="FRAME · 0.15 mm", grid="DATUM GRID, DOTTED · 0.03–0.05 mm",
             geo_clay="GEOLOGY · LAKE CLAY, VARVES · 0.08 mm", geo_till="GEOLOGY · TILL, CLASTS · 0.08 mm",
             geo_shale="GEOLOGY · SHALE, PARTINGS · 0.08 mm", geo_carbonate="GEOLOGY · CARBONATE, BEDS, JOINTS, VUGS · 0.08 mm",
             wind="SKY · CALMER HOURS, CRISP · 0.12 mm", wind_mid="SKY · WINDIER HOURS, SOFT BLUR",
             wind_strong="SKY · STRONGEST HOURS, BLURRED",
             water="WATER LEVELS · 0.05 mm", flood="1997 CREST (EST.), DASHED · 0.12 mm",
             canopy="CANOPY STIPPLE (LIDAR) · 0.2 mm", trunks="TRUNKS · 0.25 mm",
             roots="ROOTS (DIAGRAM) · 0.05–0.2 mm", ground="GROUND (LIDAR) · 0.3 mm",
             key_lines="KEY PLAN · RIVER, VALLEY EDGE, DAM · 0.06–0.35 mm", key_cut="KEY PLAN · CUT LINE · 0.3 mm",
             scale="SCALE BAR", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["frame"])
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)
    svg.layer(L["grid"])
    datum_grid(svg, sec, width, args.z_bottom, args.z_top, INK, chain=10.0)

    # Sky: hourly wind if on file, else each day's strongest gust. Each a
    # streak, its length the west-east part of the wind. With hourly wind,
    # height in the sky is time: the first hour at the top, the last lowest.
    days, station = climate.load_eccc_daily(args.climate)
    hourly, h_station = climate.load_eccc_hourly_wind(args.hourly) if args.hourly else ([], None)
    rng = np.random.default_rng(args.seed)
    ground_top = float(np.nanmax(top))
    sky_lo, sky_hi = ground_top + 4, args.z_top - 6
    if hourly:
        gusts = hourly
        t0, t1 = hourly[0][0], hourly[-1][0]
        span = (t1 - t0).total_seconds()

        def sky_z(t):
            return sky_hi - (t - t0).total_seconds() / span * (sky_hi - sky_lo)
        wind_m, sky_station = args.hourly_m, h_station
    else:
        gusts = [(d, v["gust_dir"] * 10, v["gust_spd"]) for d, v in sorted(days.items())
                 if v["gust_dir"] is not None and v["gust_spd"] is not None]

        def sky_z(t):
            return rng.uniform(sky_lo, sky_hi)
        wind_m, sky_station = args.wind_m, station
    # Each hour scatters points along its wind: more points for a stronger
    # wind, gathering toward the downwind end, loosely jittered, so the sky
    # reads as a density of air moving rather than as drawn lines.
    xs_pts, zs_pts, sp_pts = [], [], []
    for t, deg, spd in gusts:
        toward = math.radians(deg + 180)               # wind blows toward
        u = math.sin(toward) * spd * wind_m             # + toward east (right)
        w = math.cos(toward) * spd * wind_m * 0.08      # north part: a slight rise
        x = rng.uniform(0, width)
        z = sky_z(t)
        n = max(1, int(round(spd * args.points_per_kmh)))
        f = rng.beta(2.2, 1.0, n)                       # fraction along the wind, denser downwind
        xs_pts.append(x + f * u + rng.normal(0, args.jitter_x, n))
        zs_pts.append(z + f * w + rng.normal(0, args.jitter_z, n))
        sp_pts.append(np.full(n, spd))
    xs_pts = np.concatenate(xs_pts) % width             # wrap across the frame
    zs_pts = np.concatenate(zs_pts)
    sp_pts = np.concatenate(sp_pts)
    keep = (zs_pts > ground_top + 1) & (zs_pts < args.z_top)
    clip = svg.clip_rect(sec.X(0), sec.Y(args.z_top), width * sec.sh, (args.z_top - args.z_bottom) * sec.sv)
    # Tiers by wind speed: calm hours crisp, the strongest hours blurred into haze.
    all_speeds = np.array([g[2] for g in gusts])
    cuts = np.percentile(all_speeds, args.blur_tiers)
    # Blurred points are drawn larger and darker, since blurring spreads their ink.
    tiers = [(0, cuts[0], 0.0, 1.0, 0.75), (cuts[0], cuts[1], args.blur[0], 2.4, 0.9),
             (cuts[1], np.inf, args.blur[1], 4.5, 1.0)]
    for (lo, hi, blur, size, op), key in zip(tiers, ("wind", "wind_mid", "wind_strong")):
        sel = keep & (sp_pts >= lo) & (sp_pts < hi)
        svg.layer(L[key])
        svg.open_group(clip_path=clip)
        if blur:
            svg.open_group(filter=svg.blur(blur))
        svg.path("".join(f"M{sec.X(x):.2f},{sec.Y(z):.2f}h0" for x, z in zip(xs_pts[sel], zs_pts[sel])),
                 fill="none", stroke=INK, stroke_width=args.wind_dot * size, stroke_linecap="round", opacity=op)
        if blur:
            svg.close_group()
        svg.close_group()
        L[key + "_range"] = (lo, hi)

    # Water below the dam: every open-water day's level, from the bank out to the frame.
    dl, lv, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
    dq, q, sq = hydrometric.load_wsc_daily(args.level_csv, "discharge")
    svg.layer(L["water"])
    open_days = [(d, z) for d, z in zip(dl, lv) if z == z and 5 <= d.month <= 10]
    for d, z in open_days:
        edge = np.argmax(ground > z) if (ground > z).any() else width
        svg.line(sec.X(0), sec.Y(z), sec.X(edge), sec.Y(z), stroke=INK, stroke_width=0.05, opacity=0.15)
    rating, rmse, _ = hydrometric.fit_rating(dq, q, sq, dl, lv)
    q97 = max(v for d, v in zip(dq, q) if d.year == 1997 and v == v)
    z97 = rating(q97)
    edge = int(np.argmax(ground > z97))
    svg.layer(L["flood"])
    svg.line(sec.X(0), sec.Y(z97), sec.X(edge), sec.Y(z97), stroke=INK, stroke_width=0.12, stroke_dasharray="1 0.8")

    # Canopy: stipple between crown base and the lidar surface; trunks at crown tops.
    tops = crown_tops(height, args.min_tree, args.tree_spacing)
    pts = []
    for i in range(width):
        h = height[i]
        if not h == h:
            continue
        base = ground[i] + h * args.crown_base
        n = int(max(0, top[i] - base) * args.stipple)
        for _ in range(n):
            pts.append((i + rng.random(), rng.uniform(base, top[i])))
    svg.layer(L["canopy"])
    svg.path("".join(f"M{sec.X(x):.2f},{sec.Y(z):.2f}h0" for x, z in pts), fill="none", stroke=INK,
             stroke_width=0.2, stroke_linecap="round", opacity=0.75)
    svg.layer(L["trunks"])
    for i in tops:
        svg.line(sec.X(i + 0.5), sec.Y(ground[i]), sec.X(i + 0.5), sec.Y(ground[i] + height[i] * 0.85),
                 stroke=INK, stroke_width=0.25)

    # Roots: diagram, depth and spread from each tree's height.
    svg.layer(L["roots"])
    for i in tops:
        for x0_, z0_, x1_, z1_, lvl in roots(rng, i + 0.5, ground[i], min(3.0, height[i] * 0.15),
                                             height[i] * 0.45):
            svg.line(sec.X(x0_), sec.Y(z0_), sec.X(x1_), sec.Y(z1_), stroke=INK,
                     stroke_width=round(0.2 * 0.6 ** lvl, 3), opacity=0.8)

    svg.layer(L["ground"])
    svg.path(sec.path(ch, ground), fill="none", stroke=INK, stroke_width=0.3, stroke_linejoin="round")

    # Labels.
    svg.layer(L["text"])
    z_med = float(np.median([z for _, z in open_days]))
    svg.text(sec.X(3), sec.Y(z_med) - 1.5, f"OPEN WATER, MAY–OCT {dl[0].year}–{dl[-1].year} · MEDIAN +{z_med:.2f}",
             size=1.8, fill=INK)
    svg.text(sec.X(3), sec.Y(z97) - 1.5, f"1997 CREST, EST. +{z97:.2f} (±{rmse:.1f} M)", size=1.8, fill=INK)
    if tops:
        tallest = max(tops, key=lambda i: height[i])
        svg.text(sec.X(tallest + 2), sec.Y(top[tallest]) - 2, f"CANOPY {height[tallest]:.1f} M (LIDAR)", size=1.8, fill=INK)
        svg.text(sec.X(tallest + 2), sec.Y(ground[tallest] - 2.5), "ROOTS · DIAGRAM", size=1.8, fill=INK)
    if hourly:
        svg.text(sec.X(width / 2), sec.Y(args.z_top - 3), f"SKY · {len(gusts):,} HOURS OF WIND, "
                 f"{t0.strftime('%b %d %Y').upper()} – {t1.strftime('%b %d %Y').upper()}", size=2.0, fill=INK,
                 anchor="middle", spacing=0.2)
        # Gaps of more than a week in the record are labelled, not left as calm.
        times = [g[0] for g in hourly]
        for ta, tb in zip(times, times[1:]):
            if (tb - ta).days > 7:
                zm = (sky_z(ta) + sky_z(tb)) / 2
                svg.text(sec.X(width / 2), sec.Y(zm) + 0.6, f"NO WIND RECORD ON FILE · {ta.strftime('%b %d').upper()} – "
                         f"{tb.strftime('%b %d').upper()}", size=1.8, fill=INK, anchor="middle", spacing=0.2, opacity=0.6)
        mo = dt.datetime(t0.year, t0.month, 1)
        while mo <= t1:
            if mo >= t0:
                svg.text(sec.X(width) + 12, sec.Y(sky_z(mo)) + 0.6, mo.strftime("%b").upper(), size=1.8, fill=INK)
                svg.line(sec.X(width) + 7, sec.Y(sky_z(mo)), sec.X(width) + 11, sec.Y(sky_z(mo)), stroke=INK,
                         stroke_width=0.1)
            mo = dt.datetime(mo.year + (mo.month == 12), mo.month % 12 + 1, 1)
    else:
        svg.text(sec.X(width / 2), sec.Y(args.z_top - 3), f"SKY · {len(gusts)} DAYS OF STRONG GUSTS, "
                 f"{min(days).year}–{max(days).year}", size=2.0, fill=INK, anchor="middle", spacing=0.2)
    # Geology from the two well logs; blank under the unsurveyed riverbed.
    cols = np.arange(*args.cols)
    river = ground <= float(np.nanmin(ground)) + 0.15
    bed_row = np.full(len(ch), np.nan)
    if os.path.exists(args.bed):
        bed_dem, _, _ = terrain.load_dem(args.bed)
        bed_row = bed_dem[args.row, args.cols[0]:args.cols[1]]
    bed_cap = np.where(river, np.where(np.isfinite(bed_row), bed_row, float(np.nanmin(ground)) - 6.0), np.inf)
    if np.isfinite(np.where(river, bed_row, np.nan)).any():
        svg.layer(L["ground"])
        svg.path(sec.path(ch, np.where(river, bed_row, np.nan)), fill="none", stroke=INK, stroke_width=0.2,
                 stroke_dasharray="1.2 0.5")
    geology.draw_layers(svg, sec, ch, cols, ground, lambda c: float(dtm[args.row, int(c)]), args.z_bottom, INK,
                        weight=0.08, top_cap=bed_cap, layer_names={k: L[f"geo_{k}"] for k in geology.LAYERS})
    svg.layer(L["text"])

    tx, ty = m + 8, m + 12
    svg.text(tx, ty, "RED RIVER AT LOCKPORT — RIVERBANK SECTION B–B′", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx, ty + 7, f"EAST BANK, ABOUT 300 M BELOW ST. ANDREWS LOCK AND DAM · 1:{args.scale:,} (TRUE SCALE)",
             size=2.4, fill=INK, spacing=0.3)
    notes = [
        (f"Sky: each hour's wind at {sky_station.title()} scatters points along its west–east travel ({wind_m:g} m per km/h), "
         f"{args.points_per_kmh:g} points per km/h, gathering downwind; hours above the {args.blur_tiers[0]:g}th and "
         f"{args.blur_tiers[1]:g}th percentile of speed are increasingly blurred. Height in the sky is "
         "time (months at right); position across is scattered." if hourly else
         f"Sky: each day's strongest gust at {sky_station.title()} (recorded when over about 31 km/h) scatters points along its "
         f"west–east travel, {wind_m:g} m per km/h; positions in the sky are scattered, not measured."),
        "Ground and canopy heights: 1 m lidar along the section.",
        "Canopy: stippled from the lidar surface down to an assumed crown base; trunks at each crown top.",
        "Roots: a DIAGRAM sized from each tree's height (to 3 m deep); species not identified (cottonwood, elm and ash "
        "are typical of this bank).",
        f"Water: every May–October day's level below the dam ({hydrometric.station_id(args.level_csv)}); the 1997 crest "
        "is estimated from flow (see sheet 05).",
        "Geology: two drillers' logs either side of the river (Groundwater Information Network), hung from the lidar ground "
        "and joined in straight lines; blank under the unsurveyed riverbed.",
        "SOURCES: NRCan HRDEM lidar; Water Survey of Canada; Environment and Climate Change Canada; NRCan Groundwater Information Network.",
    ]
    for j, note in enumerate(notes):
        svg.text(tx, ty + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 5 else 0.65)

    svg.layer(L["scale"])
    scale_bar(svg, sec.x0, Hp - m - 9, 20 * sec.sh, 20, "m", color=INK)

    # Key plan: where the section is cut. The other section's line is dashed.
    from sheets.calendar_cut import true_north_angle
    kw = args.key_window
    kx, ky = Wp - m - 122, Hp - m - 125
    key_plan(svg, dtm, kx, ky, kw, args.row, args.cols, args.dam_row, args.dam_cols, INK,
             scale=args.key_scale, angle=true_north_angle(key_transform, key_crs, (kw[0] + kw[1]) / 2, (kw[2] + kw[3]) / 2),
             other_cuts=[(1840, (1150, 1850))], layers=dict(lines=L["key_lines"], cut=L["key_cut"], text=L["text"]),
             label=("B", "B′"), names=[("DAM", 2105, 1700, 0), ("LOCK", 2240, 1200, -90), ("FLOODWAY", 2700, 2470, -90), ("RED RIVER", 2700, 1250, -60)])
    # The cut's letters at both ends of the section.
    svg.layer(L["text"])
    for c, letter, anchor in ((0, ("B", "B′")[0], "end"), (len(ch), ("B", "B′")[1], "start")):
        svg.text(sec.X(c) + (-6 if anchor == "end" else 6), sec.Y(args.z_top) - 4, letter, size=4.5,
                 fill=INK, anchor=anchor)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  ({len(gusts)} {'hours' if hourly else 'gust days'}, {len(tops)} trees, {len(pts)} canopy dots, "
          f"{len(open_days)} water days, 1997 est {z97:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--climate", default="data/lockport/climate/en_climate_daily_MB_5023222_*.csv")
    ap.add_argument("--row", type=int, default=1840)
    ap.add_argument("--cols", type=int, nargs=2, default=[1552, 1675], help="DEM columns, water's edge to bank top")
    ap.add_argument("--scale", type=int, default=250)
    ap.add_argument("--z-top", type=float, default=335.0)
    ap.add_argument("--z-bottom", type=float, default=195.0)
    ap.add_argument("--hourly", default="data/lockport/climate/en_climate_hourly_MB_*.csv",
                    help="ECCC hourly files for the sky (empty: use daily gusts)")
    ap.add_argument("--wind-m", type=float, default=0.5, help="m of streak per km/h of gust (daily gusts)")
    ap.add_argument("--hourly-m", type=float, default=0.25, help="m of wind per km/h (hourly wind)")
    ap.add_argument("--points-per-kmh", type=float, default=0.3, help="sky points per km/h of wind, per record")
    ap.add_argument("--jitter-x", type=float, default=1.2, help="m of horizontal scatter of sky points")
    ap.add_argument("--jitter-z", type=float, default=0.35, help="m of vertical scatter of sky points")
    ap.add_argument("--wind-dot", type=float, default=0.12, help="sky point diameter, mm")
    ap.add_argument("--blur-tiers", type=float, nargs=2, default=[60, 90],
                    help="wind-speed percentiles where blurring starts and strengthens")
    ap.add_argument("--blur", type=float, nargs=2, default=[0.25, 0.7], help="blur (mm) of the two windier tiers")
    ap.add_argument("--min-tree", type=float, default=6.0, help="m canopy height counted as a tree top")
    ap.add_argument("--tree-spacing", type=int, default=6, help="m between tree tops")
    ap.add_argument("--crown-base", type=float, default=0.4, help="crown base as a share of tree height")
    ap.add_argument("--stipple", type=float, default=6.0, help="canopy dots per m of crown per m")
    ap.add_argument("--seed", type=int, default=1840)
    ap.add_argument("--key-window", type=int, nargs=4, default=[1150, 2950, 700, 2900],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"), help="DEM area shown in the key plan")
    ap.add_argument("--key-scale", type=int, default=20000)
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD_P")
    ap.add_argument("--out", default="outputs/sheets/04b_riverbank_section.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
