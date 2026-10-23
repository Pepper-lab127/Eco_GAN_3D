"""Sheet 02 - Calendar cut.

The reach of the Red River above St. Andrews Lock and Dam, cut into twelve
bands, one per month of the water year (November at the top, by the dam).
Each band is the 1 m lidar terrain at that position, flooded with that
month's record from the gauge above the dam:

  - paper white: ground under water at the lowest level recorded that month
  - white lines: the highest level reached that month, one line per year

Bands are shifted sideways by the month's median level, so the stack's
jagged edge reads as the river's yearly rise and fall. Months the gauge does
not report (ice season) are drawn pale, with no lines.

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
WATER_YEAR = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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

    # The lidar's flattened river surface is water whatever the gauge says,
    # and it is where every flood starts.
    water = terrain.flat_water(a, below=args.water_below, min_cells=1000, largest_only=True)
    lidar_water = (float(np.nanmin(a[water])), float(np.nanmax(a[water])))

    dates, values, _ = hydrometric.load_wsc_daily(args.csv, "level")
    station = hydrometric.station_id(args.csv)
    stats = hydrometric.monthly_levels(dates, values, min_days=args.min_days)
    stats = {m: s for m, s in stats.items() if len(s["max"]) >= args.min_years}

    cache = {}

    def wet(level):
        level = round(level, 2)
        if level not in cache:
            cache[level] = terrain.flood(a, level, water)
        return cache[level]

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
    medians = [stats[mo]["median"] for mo in stats]
    lowest_median = min(medians)

    svg = Svg(Wp, Hp)
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)

    record_high = max((v, y, mo) for mo, st in stats.items() for y, v in st["max"].items())

    for i, month in enumerate(WATER_YEAR):
        rs, re = band_rows[i], band_rows[i + 1]
        st = stats.get(month)
        by = y_top + i * (band_h + gap)
        bh = (re - rs) * s
        bx = x_left + ((st["median"] - lowest_median) * args.shift if st else 0.0)

        if st is None:
            tone = pale[rs:re]
            svg.image(bx, by, W * s, bh, terrain.png_data_uri((tone * 255).astype(np.uint8)))
            svg.rect(bx, by, W * s, bh, fill="none", stroke=INK, stroke_width=0.1,
                     stroke_dasharray="0.4 0.8")
            svg.text(bx + W * s + 4, by + bh / 2 + 0.8, MONTHS[month - 1], size=2.6,
                     fill=INK, spacing=0.3, opacity=0.6)
            svg.text(bx + W * s + 14, by + bh / 2 + 0.8, "NO RECORD — GAUGE OUT UNDER ICE",
                     size=1.7, fill=INK, opacity=0.6)
            continue

        low = min(st["min"].values())
        always = wet(low)[rs:re]
        tone = np.where(always, 1.0, dark[rs:re])
        svg.image(bx, by, W * s, bh, terrain.png_data_uri((tone * 255).astype(np.uint8)))

        # One line per year at that year's highest level this month.
        for year, level in sorted(st["max"].items()):
            is_record = (level, year, month) == record_high or level == max(st["max"].values())
            for line in terrain.mask_lines(wet(level)[rs:re]):
                pts = line[::2] if len(line) > 4 else line
                xs = bx + (pts[:, 0] + 0.5) * s
                ys = by + (pts[:, 1] + 0.5) * s
                svg.path(points_to_path(xs, ys), fill="none", stroke="#ffffff",
                         stroke_width=0.14 if is_record else 0.06,
                         opacity=1.0 if is_record else 0.75, stroke_linejoin="round")

        # Left: median level, as on p.238's elevation column.
        svg.line(bx - 3, by + bh / 2, bx - 0.8, by + bh / 2, stroke=INK, stroke_width=0.1)
        svg.text(bx - 4, by + bh / 2 + 0.7, f"+{st['median']:.2f}", size=1.9, fill=INK, anchor="end")

        # Right: month and its range.
        top_year = max(st["max"], key=st["max"].get)
        label_x = bx + W * s
        flag(svg, label_x, by + bh / 2, MONTHS[month - 1], dx=3, color=INK, size=2.6)
        svg.text(label_x + 17, by + bh / 2 + 0.7,
                 f"{len(st['max'])} YRS · LOW {low:.2f} · HIGH {st['max'][top_year]:.2f} ({top_year})",
                 size=1.7, fill=INK)

    # Title block.
    tx, ty = m + 8, m + 12
    svg.text(tx, ty, "RED RIVER ABOVE ST. ANDREWS LOCK AND DAM", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx, ty + 7, f"CALENDAR CUT — MONTHLY WATER LEVELS ON 1 M LIDAR TERRAIN, 1:{args.scale:,}",
             size=2.4, fill=INK, spacing=0.3)
    notes = [
        "The reach above the dam cut into twelve bands, one per month of the water year, November at the top beside the dam.",
        "Paper white: ground under water at the lowest level recorded that month. White lines: the highest level reached that month, one per year;",
        "the heavier line is the month's record. Bands shift right with the month's median level (left column, metres).",
        f"Water spreads only through connected ground below each level. Heights exaggerated {args.exaggeration:g}× in the shading only.",
        f"SOURCES: Water Survey of Canada station {station} daily water level {dates[0].year}–{dates[-1].year}; "
        "NRCan HRDEM lidar DTM (MB-Red_River_2020, MB-Cooks_Creek 2014).",
        f"Gauge levels applied without datum adjustment; the lidar river surface ({lidar_water[0]:.2f}–{lidar_water[1]:.2f} m) sits within the gauge's range.",
        "The lidar cannot see under water: where it shows river, the bed and lower banks are unknown and are drawn as water in every month.",
    ]
    for j, note in enumerate(notes):
        svg.text(tx, ty + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 4 else 0.65)

    # Scale bar and north arrow.
    sb_y = Hp - m - 9
    scale_bar(svg, x_left, sb_y, 500 * s, 500, "m", color=INK)
    north_arrow(svg, x_left + 500 * s + 14, sb_y + 0.5, r=3.0, color=INK,
                angle=true_north_angle(transform, crs, r0 + H / 2, c0 + W / 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  ({len(stats)} recorded months, {len(cache)} flood levels, "
          f"lidar water {lidar_water[0]:.2f}-{lidar_water[1]:.2f} m, {water.mean():.1%} of window)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--csv", default="data/lockport/daily.csv")
    # Rows/cols of the DEM: from just upstream of the dam to the south edge,
    # west edge to the creek east of the river.
    ap.add_argument("--window", type=int, nargs=4, default=[2200, 4014, 0, 2500],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"))
    # The 2020 survey flattened the river at ~219.75 m, the 2014 one at the
    # summer pool, ~222.6 m; both appear inside the merged channel.
    ap.add_argument("--water-below", type=float, default=223.2,
                    help="flat lidar surfaces lower than this (m) are treated as the river")
    ap.add_argument("--scale", type=int, default=6000, help="drawing scale 1:N")
    ap.add_argument("--shift", type=float, default=8.0, help="mm of sideways shift per metre of median level")
    ap.add_argument("--exaggeration", type=float, default=3.0)
    ap.add_argument("--min-days", type=int, default=10, help="days a month needs to count for a year")
    ap.add_argument("--min-years", type=int, default=5, help="years a month needs to be drawn")
    ap.add_argument("--paper", choices=list(PAPER), default="A2")
    ap.add_argument("--out", default="outputs/sheets/02_calendar_cut.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
