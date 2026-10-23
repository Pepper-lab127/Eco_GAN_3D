"""Sheet 02 - Calendar cut.

The Red River at St. Andrews Lock and Dam, cut into twelve bands, one per
month, north at the top. Each band is the 1 m lidar terrain at that position,
flooded with that month's record from the gauge above the dam:

  - paper white: ground under water at the lowest level recorded that month
  - white lines: the highest level reached that month, one line per year

Bands are shifted sideways by the month's median level, so the stack's
jagged edge reads as the river's yearly rise and fall. Months the gauge does
not report are drawn pale, with no lines. The dam holds the river at two
levels, so each side is flooded from its own gauge (above: 05OJ021, open
water only; below: 05OJ010, year-round) and no flood crosses the dam. By
default the months start in December, so the ice months fall below the dam,
where the year-round gauge recorded them.

    python -m sheets.calendar_cut
"""

import argparse
import math
import os

import numpy as np
from rasterio.warp import transform as warp_points

from drawing import hydrometric, terrain
from drawing.svg import PAPER, Svg, flag, north_arrow, points_to_path, scale_bar


MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
INK = "#111111"


def true_north_angle(transform, crs, row, col):
    """Degrees clockwise from the top of the grid to true north."""
    x, y = transform * (col, row)
    lon, lat = warp_points(crs, "EPSG:4326", [x], [y])
    x2, y2 = warp_points("EPSG:4326", crs, [lon[0]], [lat[0] + 0.01])
    return math.degrees(math.atan2(x2[0] - x, y2[0] - y))


def build(args):
    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    r1 = min(r1, dem.shape[0])
    c1 = min(c1, dem.shape[1])
    a = dem[r0:r1, c0:c1]
    H, W = a.shape
    dam = args.dam_row - r0                 # window row of the dam crest

    # The lidar's flattened river surface is water whatever the gauge says,
    # and it is where every flood starts; the dam splits it into two reaches.
    above, water_up, water_down = terrain.river_reaches(a, dam, args.water_below)
    lidar_water = (float(np.nanmin(a[water_up])), float(np.nanmax(a[water_up])))
    lidar_down = float(np.nanmedian(a[water_down])) if water_down.any() else None
    # Each reach floods from its own lidar water with its own gauge, and the
    # flood stops at the dam.
    def gauge(path):
        if not path or not os.path.exists(path):
            return dict(station=None, years=None, stats={})
        d, v, _ = hydrometric.load_wsc_daily(path, "level")
        st = hydrometric.monthly_levels(d, v, min_days=args.min_days)
        return dict(station=hydrometric.station_id(path), years=(d[0].year, d[-1].year),
                    stats={mo: x for mo, x in st.items() if len(x["max"]) >= args.min_years})

    reaches = {
        "above": dict(rows=above[:, 0], water=water_up, cache={},
                      floodable=np.where(above, a, np.inf), **gauge(args.csv)),
        "below": dict(rows=~above[:, 0], water=water_down, cache={},
                      floodable=np.where(~above, a, np.inf), **gauge(args.csv_below)),
    }

    def wet(reach, level):
        level = round(level, 2)
        if level not in reach["cache"]:
            reach["cache"][level] = terrain.flood(reach["floodable"], level, reach["water"])
        return reach["cache"][level]

    hs = terrain.hillshade(a, z=args.exaggeration)
    lo, hi = np.nanpercentile(a, [1, 99])
    height = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)
    dark = 0.10 + 0.50 * hs + 0.12 * height
    pale = 0.80 + 0.14 * hs

    # Layout.
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale                 # mm per metre (one lidar cell)
    band_rows = np.linspace(0, H, 13).astype(int)
    gap = 1.2
    band_h = (H / 12) * s
    x_left = m + 34.0
    y_top = m + 50.0
    lowest_median = min(x["median"] for r in reaches.values() for x in r["stats"].values())
    order = [(args.start_month - 1 + k) % 12 + 1 for k in range(12)]

    def band(i):
        """Rows, the reach covering most of them, and that reach's stats."""
        rs, re = band_rows[i], band_rows[i + 1]
        main = reaches["below" if reaches["below"]["rows"][rs:re].mean() > 0.5 else "above"]
        return rs, re, main, main["stats"].get(order[i])

    def band_x(i):
        st = band(i)[3]
        return x_left + ((st["median"] - lowest_median) * args.shift if st else 0.0)

    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a, **k: None)      # titles and notes: only with --annotate
    # Layers by line weight, declared bottom to top so they stack in this order.
    L = dict(frame="FRAME · 0.15 mm", terrain="BANDS · LIDAR TERRAIN (IMAGES)",
             outline="BANDS · NO-RECORD OUTLINE, DASHED · 0.1 mm",
             years="BANDS · YEAR LINES, WHITE · 0.06 mm", record="BANDS · RECORD LINES, WHITE · 0.14 mm",
             dam="BANDS · DAM, WHITE DASHED · 0.15 mm", leaders="LEADERS AND FLAGS · 0.08–0.1 mm",
             scale="SCALE BAR AND NORTH ARROW", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["frame"])
    if args.annotate:
        svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)

    for i, month in enumerate(order):
        rs, re, main, st = band(i)
        by = y_top + i * (band_h + gap)
        bh = (re - rs) * s
        bx = band_x(i)
        recorded = [r for r in reaches.values() if month in r["stats"] and r["rows"][rs:re].any()]

        if not recorded:
            tone = np.where((water_up | water_down)[rs:re], 1.0, pale[rs:re])
            svg.layer(L["terrain"])
            svg.image(bx, by, W * s, bh, terrain.png_data_uri((tone * 255).astype(np.uint8)))
            svg.layer(L["outline"])
            svg.rect(bx, by, W * s, bh, fill="none", stroke=INK, stroke_width=0.1,
                     stroke_dasharray="0.4 0.8")
            svg.layer(L["text"])
            svg.text(bx + W * s + 4, by + bh / 2 + 0.8, MONTHS[month - 1], size=2.6,
                     fill=INK, spacing=0.3, opacity=0.6)
            svg.text(bx + W * s + 14, by + bh / 2 + 0.8, "NO RECORD", size=1.7, fill=INK, opacity=0.6)
            continue

        always = water_up | water_down
        for r in recorded:
            always = always | wet(r, min(r["stats"][month]["min"].values()))
        tone = np.where(always[rs:re], 1.0, dark[rs:re])
        svg.layer(L["terrain"])
        svg.image(bx, by, W * s, bh, terrain.png_data_uri((tone * 255).astype(np.uint8)))

        # One line per year at that year's highest level this month.
        for r in recorded:
            stm = r["stats"][month]
            top = max(stm["max"].values())
            for year, level in sorted(stm["max"].items()):
                is_record = level == top
                for line in terrain.mask_lines(wet(r, level)[rs:re]):
                    pts = line[::2] if len(line) > 4 else line
                    xs = bx + (pts[:, 0] + 0.5) * s
                    # Drop the edge the dam cuts; it is not a water line.
                    rows = pts[:, 1] + rs
                    cut = rows <= dam + 0.75 if r is reaches["above"] else rows >= dam + 0.25
                    ys = np.where(cut, np.nan, by + (pts[:, 1] + 0.5) * s)
                    svg.layer(L["record" if is_record else "years"])
                    svg.path(points_to_path(xs, ys), fill="none", stroke="#ffffff",
                             stroke_width=0.14 if is_record else 0.06,
                             opacity=1.0 if is_record else 0.75, stroke_linejoin="round")

        if st is None:
            continue
        # Left: median level, as on p.238's elevation column.
        svg.layer(L["leaders"])
        svg.line(bx - 3, by + bh / 2, bx - 0.8, by + bh / 2, stroke=INK, stroke_width=0.1)
        svg.layer(L["text"])
        svg.text(bx - 4, by + bh / 2 + 0.7, f"+{st['median']:.2f}", size=1.9, fill=INK, anchor="end")

        # Right: month, its range and which gauge.
        top_year = max(st["max"], key=st["max"].get)
        label_x = bx + W * s
        flag(svg, label_x, by + bh / 2, MONTHS[month - 1], dx=3, color=INK, size=2.6,
             line_layer=L["leaders"], text_layer=L["text"])
        svg.text(label_x + 17, by + bh / 2 + 0.7,
                 f"{len(st['max'])} YRS · LOW {min(st['min'].values()):.2f} · "
                 f"HIGH {st['max'][top_year]:.2f} ({top_year}) · {main['station']}",
                 size=1.7, fill=INK)

    # The dam: a dashed line across its band.
    if 0 <= dam < H:
        i = int(np.searchsorted(band_rows, dam, side="right") - 1)
        bx = band_x(i)
        by = y_top + i * (band_h + gap)
        y = by + (dam - band_rows[i] + 0.5) * s
        svg.layer(L["dam"])
        svg.line(bx, y, bx + W * s, y, stroke="#ffffff", stroke_width=0.15, stroke_dasharray="1.2 0.8")
        svg.layer(L["leaders"])
        svg.line(bx, y, bx - 3, y, stroke=INK, stroke_width=0.1)
        svg.layer(L["text"])
        svg.text(bx - 4, y + 0.6, "DAM", size=1.7, fill=INK, anchor="end", spacing=0.2)

    # Place names, centred on a DEM row/col, rotated by angle (degrees).
    svg.layer(L["text"])
    for text, row, col, angle in args.label or []:
        lr, lc = float(row) - r0, float(col) - c0
        if not (0 <= lr < H and 0 <= lc < W):
            continue
        j = int(np.searchsorted(band_rows, lr, side="right") - 1)
        lx = band_x(j) + lc * s
        ly = y_top + j * (band_h + gap) + (lr - band_rows[j]) * s
        ink = "#ffffff" if band(j)[3] else INK
        svg.text(lx, ly + 0.6, text, size=1.7, fill=ink, anchor="middle", spacing=0.2,
                 rotate=float(angle) or None)

    # Title block.
    svg.layer(L["text"])
    tx, ty = m + 8, m + 12
    annot(tx, ty, "RED RIVER AT ST. ANDREWS LOCK AND DAM", size=5.2, fill=INK, spacing=0.4)
    annot(tx, ty + 7, f"CALENDAR CUT — MONTHLY WATER LEVELS ON 1 M LIDAR TERRAIN, 1:{args.scale:,}",
             size=2.4, fill=INK, spacing=0.3)
    up, dn = reaches["above"], reaches["below"]

    def source(r, where):
        return (f"station {r['station']} {where} the dam, daily water level {r['years'][0]}–{r['years'][1]}"
                if r["station"] else f"no gauge {where} the dam (lidar water surface only)")

    notes = [
        f"The river at the dam cut into twelve bands, one per month from {MONTHS[args.start_month - 1].title()}, north at the top. "
        "Each side of the dam is flooded from its own gauge, and no flood crosses the dam.",
        "Paper white: ground under water at the lowest level recorded that month. White lines: the highest level reached that month, one per year;",
        "the heavier line is the month's record. Bands shift right with the month's median level (left column, metres).",
        f"Water spreads only through connected ground below each level. Heights exaggerated {args.exaggeration:g}× in the shading only.",
        f"SOURCES: Water Survey of Canada {source(up, 'above')}; {source(dn, 'below')}; "
        "NRCan HRDEM lidar DTM (MB-Red_River_2020, MB-Cooks_Creek 2014).",
        f"Gauge levels applied without datum adjustment; the lidar river surface above the dam ({lidar_water[0]:.2f}–{lidar_water[1]:.2f} m)"
        + (f" and below it ({lidar_down:.2f} m)" if lidar_down else "") + " sit within the gauges' ranges. The two records cover different years.",
        "The lidar cannot see under water: where it shows river, the bed and lower banks are unknown and are drawn as water in every month.",
    ]
    for j, note in enumerate(notes):
        annot(tx, ty + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 4 else 0.65)

    # Scale bar and north arrow.
    svg.layer(L["scale"])
    sb_y = Hp - m - 9
    scale_bar(svg, x_left, sb_y, 500 * s, 500, "m", color=INK)
    north_arrow(svg, x_left + 500 * s + 14, sb_y + 0.5, r=3.0, color=INK,
                angle=true_north_angle(transform, crs, r0 + H / 2, c0 + W / 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  (months above dam {len(up['stats'])}, below {len(dn['stats'])}; "
          f"water above dam {lidar_water[0]:.2f}-{lidar_water[1]:.2f} m, below {lidar_down:.2f} m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--csv", default="data/lockport/daily.csv", help="gauge above the dam")
    ap.add_argument("--csv-below", default="data/lockport/daily_05OJ010.csv",
                    help="gauge below the dam (missing file: lidar water only)")
    # Rows/cols of the DEM: ~450 m below the dam to ~1.36 km above it, west
    # edge to the floodway, taking in the lock and the roads at the dam. With
    # twelve bands the dam falls on the top edge of the fourth band.
    ap.add_argument("--window", type=int, nargs=4, default=[1682, 3496, 0, 2600],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--dam-row", type=int, default=2135, help="DEM row of the dam crest")
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"),
                    help="place name centred on a DEM row/col; repeat for more "
                         "(default: the lock and Henderson Highway)")
    ap.add_argument("--start-month", type=int, default=12, choices=range(1, 13),
                    help="month of the top band")
    # The 2020 survey flattened the river at ~219.75 m, the 2014 one at the
    # summer pool, ~222.6 m; both appear inside the merged channel.
    ap.add_argument("--water-below", type=float, default=223.2,
                    help="flat lidar surfaces lower than this (m) are treated as the river")
    ap.add_argument("--scale", type=int, default=6500, help="drawing scale 1:N")
    ap.add_argument("--shift", type=float, default=6.0, help="mm of sideways shift per metre of median level")
    ap.add_argument("--exaggeration", type=float, default=3.0)
    ap.add_argument("--min-days", type=int, default=10, help="days a month needs to count for a year")
    ap.add_argument("--min-years", type=int, default=5, help="years a month needs to be drawn")
    ap.add_argument("--paper", choices=list(PAPER), default="A2")
    ap.add_argument("--annotate", action="store_true",
                    help="also print the title, notes, legend, sources and page border; the reading guide covers them")
    ap.add_argument("--out", default="outputs/sheets/02_calendar_cut.svg")
    args = ap.parse_args()
    if args.label is None:
        args.label = [
            ("LOCK", 2200, 1272, 0),
            ("HENDERSON HWY", 2362, 1952, -90),   # raised road ~300 m east of the river
        ]
    build(args)


if __name__ == "__main__":
    main()
