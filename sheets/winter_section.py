"""Sheet 04a - Winter section.

West to east across the Red River about 300 m below St. Andrews Lock and Dam,
white on black over a datum grid:

  - ground and winter tree canopy from the 1 m lidar (DTM, DSM minus DTM)
  - the winter water surface below the dam, every December to early-March day the gauge
    recorded (05OJ010, 2002-2008)
  - the ice: each winter's thickness estimated day by day from the cold
    (Stefan's law on Winnipeg's daily mean temperature), hung from that day's
    water level; the coldest winter drawn as daily growth lines
  - the water column, which the lidar cannot see into: drawn fading, with no
    bed, until a surveyed cross-section is available

A margin hangs one tick per day of ice thickness for every winter, and a
detail below enlarges the coldest winter's ice.

    python -m sheets.winter_section
"""

import argparse
import datetime as dt
import os

import numpy as np

from drawing import climate, hydrometric, terrain
from drawing.keyplan import key_plan
from drawing.section import Section, datum_grid, profile, water_edges
from drawing.svg import PAPER, Svg, scale_bar


INK = "#f2f2ee"
GROUND = "#0b0b0b"


def build(args):
    dtm, key_transform, key_crs = terrain.load_dem(args.dem)
    dsm, _, _ = terrain.load_dem(args.dsm)
    row, c0, c1 = args.row, args.cols[0], args.cols[1]
    ch, ground = profile(dtm, row, c0, c1)
    _, top = profile(dsm, row, c0, c1)
    canopy = np.where((top - ground > args.min_canopy) & (top - ground < 45), top, np.nan)

    # The lidar's flattened water: no ground is drawn there (the bed is unseen).
    lidar_water = float(np.nanmedian(ground[np.abs(ground - np.nanmin(ground)) < 0.05]))
    in_channel = ground <= lidar_water + 0.15
    mid = int(np.median(np.nonzero(in_channel)[0]))
    for_edges = np.where(in_channel, -np.inf, ground)

    # Winter water levels below the dam, December to Mar 10 (before the melt raises them).
    dl, lv, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
    level = {d: v for d, v in zip(dl, lv) if v == v}
    days, station_wx = climate.load_eccc_daily(args.climate)

    winters = []
    for y in range(args.first_winter, args.last_winter + 1):
        start, end = dt.date(y, 10, 1), dt.date(y + 1, 4, 30)
        afdd = climate.freezing_degree_days(days, start, end)
        ice = {d: float(climate.stefan_ice(a, args.alpha)) for d, a in afdd.items()}
        lv_days = {d: level[d] for d in level if dt.date(y, 12, 1) <= d <= dt.date(y + 1, 3, 10)}
        peak = max(ice, key=ice.get)
        winters.append(dict(year=y, ice=ice, levels=lv_days, peak=peak, afdd=afdd[peak],
                            usable=len(lv_days) >= 60))
    usable = [w for w in winters if w["usable"]]
    coldest = max(usable, key=lambda w: w["afdd"])

    # Layout: Arch D landscape.
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    sec = Section(m + 34, m + 62, args.h_scale, args.v_scale, args.z_top)
    width = len(ch)
    svg = Svg(Wp, Hp, background=GROUND)

    L = dict(frame="FRAME · 0.15 mm", grid="DATUM GRID, DOTTED · 0.06–0.1 mm",
             canopy="WINTER CANOPY (LIDAR) · 0.05 mm", water="WINTER WATER SURFACES · 0.05 mm",
             column="WATER COLUMN, FADING · 0.06 mm", growth="ICE · DAILY GROWTH LINES · 0.04 mm",
             ice="ICE · EACH WINTER'S THICKEST · 0.1 mm", ground="GROUND (LIDAR) · 0.25 mm",
             margin="MARGIN · ICE TICKS · 0.15 mm", detail="DETAIL · ICE GROWTH LINES · 0.05–0.12 mm",
             key_lines="KEY PLAN · RIVER, VALLEY EDGE, DAM · 0.06–0.35 mm", key_cut="KEY PLAN · CUT LINE · 0.3 mm",
             scale="SCALE BARS", text="TEXT")
    for name in L.values():
        svg.layer(name)

    svg.layer(L["frame"])
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)

    svg.layer(L["grid"])
    datum_grid(svg, sec, width, args.z_bottom, args.z_top, INK)

    # Winter canopy: one hairline per metre from ground to the lidar surface.
    svg.layer(L["canopy"])
    d = " ".join(f"M{sec.X(c):.2f},{sec.Y(g):.2f}V{sec.Y(t):.2f}"
                 for c, g, t in zip(ch, ground, canopy) if t == t)
    svg.path(d, fill="none", stroke=INK, stroke_width=0.05, opacity=0.55)

    # Water surfaces: every recorded winter day, edge to edge.
    svg.layer(L["water"])
    for w in usable:
        for d_, z in sorted(w["levels"].items()):
            a, b = water_edges(for_edges, z, mid)
            svg.line(sec.X(a), sec.Y(z), sec.X(b + 1), sec.Y(z), stroke=INK, stroke_width=0.05, opacity=0.18)

    # Water column under the ice: hairlines fading with depth, no bed.
    svg.layer(L["column"])
    a, b = water_edges(for_edges, lidar_water, mid)
    depth = args.column_depth
    steps = 12
    for c in np.arange(a + 2, b, args.column_spacing):
        for k in range(steps):
            z0 = lidar_water - 0.8 - depth * k / steps
            z1 = lidar_water - 0.8 - depth * (k + 1) / steps
            svg.line(sec.X(c), sec.Y(z0), sec.X(c), sec.Y(z1), stroke=INK, stroke_width=0.06,
                     opacity=round(0.45 * (1 - k / steps), 3))

    # Ice: each winter's thickest day as an outlined slab under that day's water level.
    svg.layer(L["ice"])
    for w in usable:
        pk = max((d_ for d_ in w["levels"]), key=lambda d_: w["ice"][d_])
        z = w["levels"][pk]
        hgt = w["ice"][pk]
        a, b = water_edges(for_edges, z, mid)
        svg.rect(sec.X(a), sec.Y(z), (b + 1 - a) * sec.sh, hgt * sec.sv, fill=INK if w is coldest else "none",
                 fill_opacity=0.12, stroke=INK, stroke_width=0.1, opacity=1 if w is coldest else 0.6)
        w["slab"] = (pk, z, hgt, b)

    # The coldest winter's daily growth lines inside its slab.
    svg.layer(L["growth"])
    for d_, z in sorted(coldest["levels"].items()):
        a, b = water_edges(for_edges, z, mid)
        zz = z - coldest["ice"][d_]
        svg.line(sec.X(a), sec.Y(zz), sec.X(b + 1), sec.Y(zz), stroke=INK, stroke_width=0.04, opacity=0.5)

    # Ground, except where the lidar shows water.
    svg.layer(L["ground"])
    svg.path(sec.path(ch, np.where(in_channel, np.nan, ground)), fill="none", stroke=INK,
             stroke_width=0.25, stroke_linejoin="round")

    # Margin: one tick per day of ice, hanging from each winter's line.
    mx = sec.X(width) + 30
    mw = Wp - m - mx - 12
    row_h = 30.0
    my = sec.y0 + 8
    svg.layer(L["text"])
    svg.text(mx, my - 8, "ICE THICKNESS, ONE TICK PER DAY", size=2.4, fill=INK, spacing=0.25)
    svg.text(mx, my - 4.5, f"OCT 1 – APR 30 · TICK LENGTH 1:{args.margin_scale:,}", size=1.8, fill=INK, opacity=0.75)
    for i, w in enumerate(winters):
        y = my + i * row_h
        n_days = len(w["ice"])
        svg.layer(L["margin"])
        svg.line(mx, y, mx + mw, y, stroke=INK, stroke_width=0.15)
        ticks = " ".join(f"M{mx + k / (n_days - 1) * mw:.2f},{y:.2f}v{h * 1000 / args.margin_scale:.2f}"
                         for k, (d_, h) in enumerate(sorted(w["ice"].items())) if h > 0)
        svg.path(ticks, fill="none", stroke=INK, stroke_width=0.15 if w is coldest else 0.1,
                 opacity=1 if w["usable"] else 0.55)
        svg.layer(L["text"])
        pk = w["peak"]
        svg.text(mx, y - 1.4, f"{w['year']}–{str(w['year'] + 1)[2:]}", size=2.0, fill=INK)
        svg.text(mx + mw, y - 1.4, f"{w['ice'][pk]:.2f} m · {w['afdd']:,.0f} °C·DAYS"
                 + ("" if w["usable"] else " · NO WATER LEVELS"), size=1.7, fill=INK, anchor="end", opacity=0.8)
    for mon, day in ((11, 1), (12, 1), (1, 1), (2, 1), (3, 1), (4, 1)):
        w0 = winters[0]
        dates = sorted(w0["ice"])
        k = dates.index(dt.date(w0["year"] + (mon < 10), mon, day))
        x = mx + k / (len(dates) - 1) * mw
        svg.text(x, my + len(winters) * row_h - 4, dt.date(2001, mon, 1).strftime("%b").upper(),
                 size=1.7, fill=INK, anchor="middle", opacity=0.75)

    # Detail: the coldest winter's ice, enlarged, across the middle of the channel.
    det = Section(sec.x0, sec.Y(args.z_bottom) + 40, args.detail_h_scale, args.detail_v_scale,
                  max(coldest["levels"].values()) + 0.3)
    span = args.detail_span
    svg.layer(L["detail"])
    for d_, z in sorted(coldest["levels"].items()):
        under = z - coldest["ice"][d_]
        first = d_.day == 1
        for zz, op in ((z, 0.25), (under, 0.7)):
            svg.line(det.X(0), det.Y(zz), det.X(span), det.Y(zz), stroke=INK,
                     stroke_width=0.12 if first else 0.05, opacity=1 if first else op)
        if first:
            svg.layer(L["text"])
            svg.text(det.X(span) + 3, det.Y(under) + 0.7,
                     f"{d_.strftime('%b %d').upper()} · {coldest['ice'][d_]:.2f} m", size=1.7, fill=INK)
            svg.layer(L["detail"])
    z_lo = min(z - coldest["ice"][d_] for d_, z in coldest["levels"].items()) - 0.2
    svg.layer(L["text"])
    svg.text(det.X(0), det.Y(det.z_top) - 4, f"DETAIL · ICE OF {coldest['year']}–{str(coldest['year'] + 1)[2:]}, "
             f"THE COLDEST WINTER · H 1:{args.detail_h_scale:,} · V 1:{args.detail_v_scale:,}", size=2.0, fill=INK,
             spacing=0.2)
    svg.text(det.X(0), det.Y(z_lo) + 4, "Faint: the water surface each day. Strong: the underside of the ice "
             "each day, estimated; heavier lines on the first of each month.", size=1.7, fill=INK, opacity=0.8)

    # Labels in the section.
    pk, z, hgt, b = coldest["slab"]
    svg.text(sec.X(b + 1) + 3, sec.Y(z - hgt / 2) + 0.6, f"ICE {coldest['year']}–{str(coldest['year'] + 1)[2:]} · "
             f"{hgt:.2f} m · {pk.strftime('%b %d').upper()}", size=1.8, fill=INK)
    a, b = water_edges(for_edges, lidar_water, mid)
    svg.text(sec.X((a + b) / 2), sec.Y(lidar_water - 0.8 - depth * 0.55), "WATER COLUMN · DEPTH NOT SURVEYED",
             size=1.8, fill=INK, anchor="middle", spacing=0.2, opacity=0.8)
    svg.text(sec.X(width / 2), sec.Y(args.z_bottom + 3), "LAKE AGASSIZ CLAY, TILL AND LIMESTONE AQUIFER: "
             "AWAITING WELL LOGS", size=1.8, fill=INK, anchor="middle", spacing=0.2, opacity=0.55)
    for c, text in ((args.label_west, "WEST BANK"), (args.label_east, "EAST BANK · RIPARIAN WOOD")):
        svg.text(sec.X(c), sec.Y(args.z_top - 2), text, size=1.8, fill=INK, anchor="middle", spacing=0.2)

    # Title block.
    tx, ty = m + 8, m + 12
    svg.text(tx, ty, "RED RIVER AT LOCKPORT — WINTER SECTION A–A′", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx, ty + 7, f"WEST TO EAST, ABOUT 300 M BELOW ST. ANDREWS LOCK AND DAM · H 1:{args.h_scale:,} · "
             f"V 1:{args.v_scale:,} (HEIGHTS ×{args.h_scale / args.v_scale:.3g})", size=2.4, fill=INK, spacing=0.3)
    yrs = f"{usable[0]['year']}–{usable[-1]['year'] + 1}"
    notes = [
        f"Ground and leafless canopy: 1 m lidar along the section (DTM; DSM minus DTM). Water surfaces: every December to early-March day "
        f"recorded below the dam at {hydrometric.station_id(args.level_csv)}, winters {yrs}.",
        f"Ice: ESTIMATED day by day from the cold by Stefan's law, thickness = {args.alpha} × √(accumulated freezing degree-days), "
        f"a typical value for rivers under snow (range about 0.014–0.017); daily mean temperature at {station_wx.title()}.",
        "Each winter's thickest ice is outlined under that day's water level; the coldest winter is filled, with a line for the "
        "underside of its ice on each day.",
        "The lidar stops at the water surface: the water column is drawn fading, with no bed, until a surveyed cross-section "
        "is available. Geology awaits well logs.",
        "SOURCES: NRCan HRDEM lidar; Water Survey of Canada; Environment and Climate Change Canada daily climate data.",
    ]
    for j, note in enumerate(notes):
        svg.text(tx, ty + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 4 else 0.65)

    svg.layer(L["scale"])
    sb_y = Hp - m - 9
    scale_bar(svg, sec.x0, sb_y, 100 * sec.sh, 100, "m (horizontal)", color=INK)
    vx = sec.x0 + 100 * sec.sh + 45
    svg.line(vx, sb_y + 0.8, vx, sb_y + 0.8 - 5 * sec.sv, stroke=INK, stroke_width=0.15)
    svg.text(vx + 2, sb_y + 0.8, "5 m (vertical)", size=1.8, fill=INK)

    # Key plan: where the section is cut. The other section's line is dashed.
    from sheets.calendar_cut import true_north_angle
    kw = args.key_window
    kx, ky = Wp - m - 122, Hp - m - 125
    key_plan(svg, dtm, kx, ky, kw, args.row, args.cols, args.dam_row, args.dam_cols, INK,
             scale=args.key_scale, angle=true_north_angle(key_transform, key_crs, (kw[0] + kw[1]) / 2, (kw[2] + kw[3]) / 2),
             other_cuts=[(1840, (1552, 1675))], layers=dict(lines=L["key_lines"], cut=L["key_cut"], text=L["text"]),
             label=("A", "A′"), names=[("DAM", 2105, 1700, 0), ("LOCK", 2240, 1200, -90), ("FLOODWAY", 2700, 2470, -90), ("RED RIVER", 2700, 1250, -60)])
    # The cut's letters at both ends of the section.
    svg.layer(L["text"])
    for c, letter, anchor in ((0, ("A", "A′")[0], "end"), (len(ch), ("A", "A′")[1], "start")):
        svg.text(sec.X(c) + (-6 if anchor == "end" else 6), sec.Y(args.z_top) + 1, letter, size=4.5,
                 fill=INK, anchor=anchor)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  (lidar water {lidar_water:.2f} m, channel cols {a}-{b}; winters with levels: "
          f"{[w['year'] for w in usable]}; coldest {coldest['year']} ice {coldest['ice'][coldest['peak']]:.2f} m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--climate", default="data/lockport/climate/en_climate_daily_MB_5023222_*.csv")
    ap.add_argument("--row", type=int, default=1840, help="DEM row of the section line")
    ap.add_argument("--cols", type=int, nargs=2, default=[1150, 1850], help="DEM columns, west to east")
    ap.add_argument("--first-winter", type=int, default=2002)
    ap.add_argument("--last-winter", type=int, default=2007)
    ap.add_argument("--alpha", type=float, default=0.015, help="Stefan coefficient, m/(°C day)^0.5")
    ap.add_argument("--min-canopy", type=float, default=1.5, help="m above ground counted as canopy")
    ap.add_argument("--column-depth", type=float, default=6.0, help="m the water column fades over")
    ap.add_argument("--column-spacing", type=float, default=2.0, help="m between water-column hairlines")
    ap.add_argument("--h-scale", type=int, default=1000)
    ap.add_argument("--v-scale", type=int, default=150)
    ap.add_argument("--z-top", type=float, default=248.0)
    ap.add_argument("--z-bottom", type=float, default=195.0)
    ap.add_argument("--margin-scale", type=int, default=40, help="ice tick scale 1:N")
    ap.add_argument("--detail-h-scale", type=int, default=1000)
    ap.add_argument("--detail-v-scale", type=int, default=20)
    ap.add_argument("--detail-span", type=float, default=380.0, help="m of channel width shown in the detail")
    ap.add_argument("--label-west", type=float, default=150.0, help="chainage of the west bank label")
    ap.add_argument("--label-east", type=float, default=470.0, help="chainage of the east bank label")
    ap.add_argument("--key-window", type=int, nargs=4, default=[1150, 2950, 700, 2900],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"), help="DEM area shown in the key plan")
    ap.add_argument("--key-scale", type=int, default=20000)
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--out", default="outputs/sheets/04a_winter_section.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
