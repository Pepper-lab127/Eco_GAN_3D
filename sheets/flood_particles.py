"""Sheet 05 - The 1997 flood as particles.

Every day of the flood, the water that passed Lockport that day (gauge
05OJ010, daily flow) is drawn as dots, one dot per fixed volume. Each day's
dots gather along that day's water's edge above the dam (gauge 05OJ021,
daily level, on the 1 m lidar terrain), thinning away from the edge. Where
the edge lingered day after day the dots pile up; the crest is the outermost
front. Like the Lituya Bay wave-progression plan, the days are superimposed in
one plan, and a tick for each day runs down the margin, its length the flow.

Below the dam no 1997 level was recorded, so no dots are placed there.

    python -m sheets.flood_particles
"""

import argparse
import datetime as dt
import os

import numpy as np
from scipy import ndimage

from drawing import hydrometric, terrain
from drawing.svg import PAPER, Svg, north_arrow, points_to_path, scale_bar
from sheets.calendar_cut import true_north_angle


INK = "#111111"


def daily(path, param, start, end):
    """{date: value} for start..end, linearly filling short gaps."""
    dates, values, _ = hydrometric.load_wsc_daily(path, param)
    known = {d: v for d, v in zip(dates, values) if v == v and start - dt.timedelta(30) <= d <= end + dt.timedelta(30)}
    days = [start + dt.timedelta(k) for k in range((end - start).days + 1)]
    xs = np.array([d.toordinal() for d in sorted(known)])
    ys = np.array([known[d] for d in sorted(known)])
    return {d: float(np.interp(d.toordinal(), xs, ys)) for d in days}, set(days) - set(known)


def build(args):
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    a = dem[r0:min(r1, dem.shape[0]), c0:min(c1, dem.shape[1])]
    H, W = a.shape
    dam = args.dam_row - r0
    above, water_up, water_down = terrain.river_reaches(a, dam, args.water_below)
    floodable = np.where(above, a, np.inf)

    level, level_filled = daily(args.level_csv, "level", start, end)
    flow, flow_filled = daily(args.flow_csv, "discharge", start, end)
    days = sorted(level)
    crest_day = max(days, key=level.get)
    peak_day = max(days, key=flow.get)

    # Dots: each day's volume, gathered along that day's water's edge.
    rng = np.random.default_rng(args.seed)
    xs_all, ys_all, extents = [], [], {}
    for d in days:
        wet = terrain.flood(floodable, level[d], water_up)
        dry = above & ~wet
        if d in (start, crest_day):
            extents[d] = wet
        dist = ndimage.distance_transform_edt(~dry)       # metres to the nearest dry ground
        weight = np.where(wet, np.exp(-dist / args.front), 0.0).ravel()
        n = int(round(flow[d] * 86400 / args.volume))
        if n == 0 or weight.sum() == 0:
            continue
        cells = rng.choice(weight.size, size=n, p=weight / weight.sum())
        rows, cols = np.divmod(cells, W)
        xs_all.append(cols + rng.random(n))
        ys_all.append(rows + rng.random(n))
    xs_all, ys_all = np.concatenate(xs_all), np.concatenate(ys_all)

    # Layout.
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale
    x0, y0 = m + 34.0, m + 50.0
    pw, ph = W * s, H * s

    svg = Svg(Wp, Hp)
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)
    clip = svg.clip_rect(x0, y0, pw, ph)
    svg.open_group(clip_path=clip)

    # Ground: faint contours of the river valley, one per metre.
    for lv, lines in terrain.smooth_contours(a, np.arange(218, 231, 1.0)).items():
        d = " ".join(points_to_path(x0 + (ln[::2, 0] + 0.5) * s, y0 + (ln[::2, 1] + 0.5) * s)
                     for ln in lines if len(ln) > 8)
        svg.path(d, fill="none", stroke="#8a8a8a", stroke_width=0.05,
                 opacity=0.9 if lv % 5 == 0 else 0.5)

    # The dots, as zero-length round-capped strokes in one path.
    d = "".join(f"M{x0 + x * s:.2f},{y0 + y * s:.2f}h0" for x, y in zip(xs_all, ys_all))
    svg.path(d, fill="none", stroke=INK, stroke_width=args.dot, stroke_linecap="round", opacity=0.85)

    # Hairlines: the flood's first and highest edges.
    for d, style in ((start, dict(stroke_dasharray="0.6 0.6", stroke_width=0.08)),
                     (crest_day, dict(stroke_width=0.12))):
        path = " ".join(
            points_to_path(x0 + (ln[:, 0] + 0.5) * s,
                           np.where(ln[:, 1] <= dam + 0.75, np.nan, y0 + (ln[:, 1] + 0.5) * s))
            for ln in terrain.mask_lines(extents[d]))
        svg.path(path, fill="none", stroke=INK, **style)

    # Dam and place names.
    if 0 <= dam < H:
        y = y0 + (dam + 0.5) * s
        svg.line(x0, y, x0 + pw, y, stroke=INK, stroke_width=0.12, stroke_dasharray="1.2 0.8")
    svg.close_group()
    if 0 <= dam < H:
        y = y0 + (dam + 0.5) * s
        svg.line(x0 - 3, y, x0 - 0.8, y, stroke=INK, stroke_width=0.1)
        svg.text(x0 - 4, y + 0.6, "DAM", size=1.7, fill=INK, anchor="end", spacing=0.2)
    for text, row, col, angle in args.label:
        lr, lc = float(row) - r0, float(col) - c0
        if 0 <= lr < H and 0 <= lc < W:
            svg.text(x0 + lc * s, y0 + lr * s + 0.6, text, size=1.7, fill=INK, anchor="middle",
                     spacing=0.2, rotate=float(angle) or None, opacity=0.8)

    # Timeline: a tick per day down the margin, its length the day's flow.
    tx = x0 + pw + 24
    tick_max = Wp - m - tx - 26
    qmax = max(flow.values())
    svg.text(tx, y0 - 6, "DAILY FLOW PASSING LOCKPORT", size=1.9, fill=INK, spacing=0.2)
    svg.text(tx, y0 - 3, f"STATION {hydrometric.station_id(args.flow_csv)} · ONE TICK PER DAY", size=1.6,
             fill=INK, opacity=0.7)
    svg.line(tx, y0, tx, y0 + ph, stroke=INK, stroke_width=0.12)
    for k, d in enumerate(days):
        y = y0 + k / (len(days) - 1) * ph
        filled = d in flow_filled
        svg.line(tx, y, tx + flow[d] / qmax * tick_max, y, stroke=INK,
                 stroke_width=0.25 if d == peak_day else 0.08,
                 stroke_dasharray="0.4 0.4" if filled else None)
        if d.weekday() == 0 or d in (start, end):
            svg.text(tx - 2, y + 0.6, d.strftime("%b %d").upper(), size=1.6, fill=INK, anchor="end")
        if d == peak_day:
            svg.text(tx + tick_max + 2, y + 0.6, f"{flow[d]:,.0f} m³/s", size=1.7, fill=INK)
        if d == crest_day and d != peak_day:
            svg.text(tx + flow[d] / qmax * tick_max + 2, y - 1.2, f"CREST {level[d]:.2f} m", size=1.6,
                     fill=INK, opacity=0.8)
    step = 1000
    svg.line(tx, y0 + ph + 5, tx + step / qmax * tick_max, y0 + ph + 5, stroke=INK, stroke_width=0.15)
    svg.text(tx, y0 + ph + 8, f"{step:,} m³/s", size=1.6, fill=INK)

    # Title block.
    n = len(xs_all)
    tx0, ty0 = m + 8, m + 12
    svg.text(tx0, ty0, "RED RIVER AT ST. ANDREWS LOCK AND DAM", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx0, ty0 + 7, f"THE 1997 FLOOD — EACH DOT {args.volume:,.0f} M³ OF WATER, "
             f"{start.strftime('%b %d').upper()} – {end.strftime('%b %d').upper()}, 1:{args.scale:,}",
             size=2.4, fill=INK, spacing=0.3)
    side = round(args.volume ** (1 / 3))
    notes = [
        f"Each day the water that passed Lockport is drawn as dots, one per {args.volume:,.0f} m³ (a cube about {side} m on a side): "
        f"{n:,} dots for {len(days)} days.",
        f"Each day's dots gather along that day's water's edge above the dam and thin away from it over about {args.front:g} m, "
        "so the dots pile up where the edge lingered.",
        f"Dashed line: the edge on {start.strftime('%b %d')} ({level[start]:.2f} m). Solid line: the crest, "
        f"{crest_day.strftime('%b %d')} ({level[crest_day]:.2f} m). Faint lines: ground contours every metre.",
        "Below the dam no 1997 water level was recorded, so no dots are placed there.",
        f"SOURCES: Water Survey of Canada stations {hydrometric.station_id(args.level_csv)} (daily water level, above the dam) and "
        f"{hydrometric.station_id(args.flow_csv)} (daily flow, below the dam); NRCan HRDEM lidar DTM.",
        "Water spreads only through connected ground below each level. The lidar cannot see under water; "
        "levels applied without datum adjustment."
        + (f" {len(level_filled | flow_filled)} missing days interpolated (dashed ticks)." if level_filled | flow_filled else ""),
    ]
    for j, note in enumerate(notes):
        svg.text(tx0, ty0 + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 4 else 0.65)

    sb_y = Hp - m - 9
    scale_bar(svg, x0, sb_y, 500 * s, 500, "m", color=INK)
    north_arrow(svg, x0 + 500 * s + 14, sb_y + 0.5, r=3.0, color=INK,
                angle=true_north_angle(transform, crs, r0 + H / 2, c0 + W / 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  ({n:,} dots, {len(days)} days, crest {crest_day} {level[crest_day]:.2f} m, "
          f"peak flow {peak_day} {flow[peak_day]:,.0f} m3/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily.csv", help="daily level above the dam")
    ap.add_argument("--flow-csv", default="data/lockport/daily_05OJ010.csv", help="daily flow")
    ap.add_argument("--start", default="1997-04-01")
    ap.add_argument("--end", default="1997-06-30")
    ap.add_argument("--volume", type=float, default=100_000, help="m³ of water per dot")
    ap.add_argument("--front", type=float, default=12.0, help="m over which each day's dots thin from the edge")
    ap.add_argument("--dot", type=float, default=0.16, help="dot diameter in mm")
    ap.add_argument("--seed", type=int, default=1997)
    ap.add_argument("--window", type=int, nargs=4, default=[1682, 3496, 0, 2600],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"))
    ap.add_argument("--scale", type=int, default=6500)
    ap.add_argument("--paper", choices=list(PAPER), default="A2")
    ap.add_argument("--out", default="outputs/sheets/05_flood_particles_1997.svg")
    args = ap.parse_args()
    if args.label is None:
        args.label = [("LOCK", 2200, 1272, 0), ("HENDERSON HWY", 2362, 1952, -90)]
    build(args)


if __name__ == "__main__":
    main()
