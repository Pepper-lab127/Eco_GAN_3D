"""Sheet 05 - The 1997 flood as particles.

Every day of the flood, the water that passed Lockport that day (gauge
05OJ010, daily flow) is drawn as dots, one dot per fixed volume. Each day's
dots gather along that day's water's edges, thinning away from them. Where an
edge lingered day after day the dots pile up; the crest is the outermost
front. Like the Lituya Bay wave-progression plan, the days are superimposed in
one plan, and a tick for each day runs down the margin, its length the flow.

The dam splits the river, so each part is flooded from its own water level:

  - above the dam: gauge 05OJ021, daily water level (recorded)
  - below the dam: estimated from 05OJ010's daily flow through the stage-flow
    relation fitted on its 2002-2008 days with both level and flow
  - the floodway: its own gauge if given (--floodway-csv, e.g. 05OJ014);
    otherwise only its lidar water surface is drawn

A barrier across the dam and lock keeps the floods apart; elsewhere water
spreads only through connected ground below each level.

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
    """{date: value} for start..end, linearly filling gaps; and the filled days."""
    dates, values, _ = hydrometric.load_wsc_daily(path, param)
    pad = dt.timedelta(30)
    known = {d: v for d, v in zip(dates, values) if v == v and start - pad <= d <= end + pad}
    if not known:
        raise SystemExit(f"{path}: no {param} between {start} and {end}")
    days = [start + dt.timedelta(k) for k in range((end - start).days + 1)]
    order = sorted(known)
    xs = np.array([d.toordinal() for d in order])
    ys = np.array([known[d] for d in order])
    return {d: float(np.interp(d.toordinal(), xs, ys)) for d in days}, set(days) - set(known)


def build(args):
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    a = dem[r0:min(r1, dem.shape[0]), c0:min(c1, dem.shape[1])]
    H, W = a.shape
    dam = args.dam_row - r0

    # Lidar water of each reach: the river above and below the dam, and the
    # floodway (the largest remaining flat water, away from the river).
    _, water_up, water_down = terrain.river_reaches(a, dam, args.water_below)
    near_river = ndimage.binary_dilation(water_up | water_down, iterations=25)
    water_fw = terrain.flat_water(np.where(near_river, np.nan, a), below=args.water_below,
                                  min_cells=1000, largest_only=True)

    # A barrier across the dam and lock keeps the reaches' floods apart.
    barrier = np.zeros(a.shape, bool)
    barrier[max(dam - 8, 0):dam + 9, args.dam_cols[0] - c0:args.dam_cols[1] - c0] = True
    floodable = np.where(barrier, np.inf, a)

    flow, flow_filled = daily(args.flow_csv, "discharge", start, end)
    days = sorted(flow)
    peak_day = max(days, key=flow.get)

    level_up, up_filled = daily(args.level_csv, "level", start, end)

    # Below the dam: the stage-flow relation of 05OJ010, applied to each day's flow.
    dq, q, sq = hydrometric.load_wsc_daily(args.flow_csv, "discharge")
    dh, h, _ = hydrometric.load_wsc_daily(args.flow_csv, "level")
    rating, rmse, (q_lo, q_hi) = hydrometric.fit_rating(dq, q, sq, dh, h)
    level_dn = {d: rating(flow[d]) for d in days}
    beyond = [d for d in days if flow[d] > q_hi]

    reaches = [
        dict(name="above the dam", water=water_up, level=level_up),
        dict(name="below the dam (estimated)", water=water_down, level=level_dn),
    ]
    if args.floodway_csv:
        level_fw, _ = daily(args.floodway_csv, "level", start, end)
        reaches.append(dict(name="floodway", water=water_fw, level=level_fw))
    for r in reaches:
        r["crest"] = max(days, key=r["level"].get)

    # Dots: each day's volume, gathered along that day's water's edges.
    rng = np.random.default_rng(args.seed)
    xs_all, ys_all, edges = [], [], {}
    for d in days:
        wet = water_up | water_down | water_fw
        for r in reaches:
            r_wet = terrain.flood(floodable, r["level"][d], r["water"])
            wet = wet | r_wet
            if d in (start, r["crest"]):
                edges[(r["name"], d)] = r_wet
        dist = ndimage.distance_transform_edt(wet)         # metres to the nearest dry ground
        weight = np.where(wet & ~barrier, np.exp(-dist / args.front), 0.0).ravel()
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
    svg.open_group(clip_path=svg.clip_rect(x0, y0, pw, ph))

    # Ground: faint contours of the river valley, one per metre.
    for lv, lines in terrain.smooth_contours(a, np.arange(216, 232, 1.0)).items():
        d = " ".join(points_to_path(x0 + (ln[::2, 0] + 0.5) * s, y0 + (ln[::2, 1] + 0.5) * s)
                     for ln in lines if len(ln) > 8)
        svg.path(d, fill="none", stroke="#8a8a8a", stroke_width=0.05,
                 opacity=0.9 if lv % 5 == 0 else 0.5)

    # A reach with no record: its lidar water outlined, dotted.
    if not args.floodway_csv and water_fw.any():
        path = " ".join(points_to_path(x0 + (ln[:, 0] + 0.5) * s, y0 + (ln[:, 1] + 0.5) * s)
                        for ln in terrain.mask_lines(water_fw))
        svg.path(path, fill="none", stroke=INK, stroke_width=0.1, stroke_dasharray="0.3 0.6")

    # The dots, as zero-length round-capped strokes in one path.
    d = "".join(f"M{x0 + x * s:.2f},{y0 + y * s:.2f}h0" for x, y in zip(xs_all, ys_all))
    svg.path(d, fill="none", stroke=INK, stroke_width=args.dot, stroke_linecap="round", opacity=0.85)

    # Hairlines: each reach's first and highest edges.
    for (name, d), wet in edges.items():
        style = dict(stroke_width=0.12) if d != start else dict(stroke_width=0.08, stroke_dasharray="0.6 0.6")
        path = " ".join(points_to_path(x0 + (ln[:, 0] + 0.5) * s, y0 + (ln[:, 1] + 0.5) * s)
                        for ln in terrain.mask_lines(wet & ~barrier))
        svg.path(path, fill="none", stroke=INK, **style)

    # The dam.
    if 0 <= dam < H:
        y = y0 + (dam + 0.5) * s
        svg.line(x0 + (args.dam_cols[0] - c0) * s, y, x0 + (args.dam_cols[1] - c0) * s, y,
                 stroke=INK, stroke_width=0.35)
    svg.close_group()
    for text, row, col, angle in args.label:
        lr, lc = float(row) - r0, float(col) - c0
        if 0 <= lr < H and 0 <= lc < W:
            svg.text(x0 + lc * s, y0 + lr * s + 0.6, text, size=1.7, fill=INK, anchor="middle",
                     spacing=0.2, rotate=float(angle) or None, opacity=0.8)

    # Timeline: a tick per day down the margin, its length the day's flow.
    tx = x0 + pw + 26
    tick_max = min(Wp - m - tx - 30, 90.0)
    qmax = max(flow.values())
    svg.text(tx, y0 - 6, "DAILY FLOW PASSING LOCKPORT", size=1.9, fill=INK, spacing=0.2)
    svg.text(tx, y0 - 3, f"STATION {hydrometric.station_id(args.flow_csv)} · ONE TICK PER DAY",
             size=1.6, fill=INK, opacity=0.7)
    svg.line(tx, y0, tx, y0 + ph, stroke=INK, stroke_width=0.12)
    crest_up = reaches[0]["crest"]
    for k, d in enumerate(days):
        y = y0 + k / (len(days) - 1) * ph
        svg.line(tx, y, tx + flow[d] / qmax * tick_max, y, stroke=INK,
                 stroke_width=0.25 if d == peak_day else 0.08,
                 stroke_dasharray="0.4 0.4" if d in flow_filled else None)
        if d.weekday() == 0 or d in (start, end):
            svg.text(tx - 2, y + 0.6, d.strftime("%b %d").upper(), size=1.6, fill=INK, anchor="end")
        if d == peak_day:
            svg.text(tx + tick_max + 2, y + 0.6, f"{flow[d]:,.0f} m³/s", size=1.7, fill=INK)
        if d == crest_up and d != peak_day:
            svg.text(tx + flow[d] / qmax * tick_max + 2, y - 1.2,
                     f"CREST ABOVE DAM {level_up[d]:.2f} m", size=1.6, fill=INK, opacity=0.8)
        if d == beyond[0] if beyond else False:
            svg.line(tx + q_hi / qmax * tick_max, y0, tx + q_hi / qmax * tick_max, y0 + ph,
                     stroke=INK, stroke_width=0.06, stroke_dasharray="0.5 0.8")
    if beyond:
        bx = tx + q_hi / qmax * tick_max
        svg.text(bx + 1, y0 + ph + 2.6, f"{q_hi:,.0f} m³/s: HIGHEST FLOW IN THE STAGE-FLOW FIT",
                 size=1.5, fill=INK, opacity=0.7)
    svg.line(tx, y0 + ph + 6, tx + 1000 / qmax * tick_max, y0 + ph + 6, stroke=INK, stroke_width=0.15)
    svg.text(tx, y0 + ph + 9, "1,000 m³/s", size=1.6, fill=INK)

    # Title block.
    n = len(xs_all)
    tx0, ty0 = m + 8, m + 12
    svg.text(tx0, ty0, "RED RIVER AT ST. ANDREWS LOCK AND DAM", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx0, ty0 + 7, f"THE 1997 FLOOD — EACH DOT {args.volume:,.0f} M³ OF WATER, "
             f"{start.strftime('%b %d').upper()} – {end.strftime('%b %d').upper()}, 1:{args.scale:,}",
             size=2.4, fill=INK, spacing=0.3)
    side = round(args.volume ** (1 / 3))
    crest_dn = reaches[1]["crest"]
    notes = [
        f"Each day the water that passed Lockport is drawn as dots, one per {args.volume:,.0f} m³ (a cube about {side} m on a side): "
        f"{n:,} dots for {len(days)} days.",
        f"Each day's dots gather along that day's water's edges and thin away from them over about {args.front:g} m, "
        "so the dots pile up where an edge lingered. Dashed lines: the first day's edges; solid lines: each reach's crest.",
        f"Above the dam: daily water level recorded at {hydrometric.station_id(args.level_csv)} "
        f"(crest {level_up[crest_up]:.2f} m on {crest_up.strftime('%b %d')}).",
        f"Below the dam: ESTIMATED from each day's flow through the stage-flow relation of {hydrometric.station_id(args.flow_csv)}, "
        f"fitted on its 2002–2008 days with both records (±{rmse:.1f} m, flows {q_lo:,.0f}–{q_hi:,.0f} m³/s); "
        f"crest {level_dn[crest_dn]:.2f} m on {crest_dn.strftime('%b %d')}.",
        (f"{len(beyond)} days exceeded the fitted range, up to {qmax:,.0f} m³/s: their levels below the dam are extrapolated."
         if beyond else "All days lie within the fitted range."),
        (f"Floodway: daily water level from {hydrometric.station_id(args.floodway_csv)}."
         if args.floodway_csv else
         "Floodway (dotted outline): no 1997 record on file, so only its lidar water surface is shown; "
         "it carried water in 1997 that this sheet cannot place."),
        "Faint lines: ground contours every metre. Heavy bar: the dam and lock, which no flood crosses here.",
        f"SOURCES: Water Survey of Canada stations {hydrometric.station_id(args.level_csv)}, "
        f"{hydrometric.station_id(args.flow_csv)}" + (f", {hydrometric.station_id(args.floodway_csv)}" if args.floodway_csv else "")
        + "; NRCan HRDEM lidar DTM. Levels applied without datum adjustment; the lidar cannot see under water.",
    ]
    for j, note in enumerate(notes):
        svg.text(tx0, ty0 + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 7 else 0.65)

    sb_y = Hp - m - 9
    scale_bar(svg, x0, sb_y, 500 * s, 500, "m", color=INK)
    north_arrow(svg, x0 + 500 * s + 14, sb_y + 0.5, r=3.0, color=INK,
                angle=true_north_angle(transform, crs, r0 + H / 2, c0 + W / 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  ({n:,} dots, {len(days)} days; crest above dam {crest_up} {level_up[crest_up]:.2f} m, "
          f"below dam est. {crest_dn} {level_dn[crest_dn]:.2f} m; rating ±{rmse:.2f} m on {q_lo:.0f}-{q_hi:.0f} m3/s, "
          f"{len(beyond)} days beyond; floodway water {water_fw.sum()} cells)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily.csv", help="daily level above the dam")
    ap.add_argument("--flow-csv", default="data/lockport/daily_05OJ010.csv",
                    help="daily flow, plus level for the stage-flow relation below the dam")
    ap.add_argument("--floodway-csv", help="daily level in the floodway (e.g. station 05OJ014)")
    ap.add_argument("--start", default="1997-04-01")
    ap.add_argument("--end", default="1997-06-30")
    ap.add_argument("--volume", type=float, default=100_000, help="m³ of water per dot")
    ap.add_argument("--front", type=float, default=12.0, help="m over which each day's dots thin from the edge")
    ap.add_argument("--dot", type=float, default=0.16, help="dot diameter in mm")
    ap.add_argument("--seed", type=int, default=1997)
    # Rows/cols of the DEM: the river from the floodway outlet north of the
    # dam to ~1.6 km above it, and the floodway channel to the east.
    ap.add_argument("--window", type=int, nargs=4, default=[1150, 3700, 0, 2900],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1240, 1700],
                    help="DEM columns the dam and lock span (the flood barrier)")
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"))
    ap.add_argument("--scale", type=int, default=6000)
    ap.add_argument("--paper", choices=list(PAPER), default="A1")
    ap.add_argument("--out", default="outputs/sheets/05_flood_particles_1997.svg")
    args = ap.parse_args()
    if args.label is None:
        args.label = [("LOCK", 2200, 1272, 0), ("HENDERSON HWY", 2362, 1952, -90),
                      ("FLOODWAY", 3150, 2375, -90)]
    build(args)


if __name__ == "__main__":
    main()
