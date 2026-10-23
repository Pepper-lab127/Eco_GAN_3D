"""Sheet 05 - A flood as particles (1997 by default; 1974 with the floodway).

Every day of the flood, the water that passed Lockport that day (gauge
05OJ010, daily flow) is drawn as dots, one dot per fixed volume. Each day's
dots gather along that day's water's edges, thinning away from them. Where an
edge lingered day after day the dots pile up; the crest is the outermost
front. Like the Lituya Bay wave-progression plan, the days are superimposed in
one plan, and a tick for each day runs down the margin, its length the flow.

The dam splits the river, so each part is flooded from its own water level:

  - above the dam: gauge 05OJ021's daily level where it exists (1996 on);
    otherwise estimated from James Avenue's level (05OJ015) through their
    relation in spring open water
  - below the dam: estimated from 05OJ010's daily flow through the stage-flow
    relation fitted on its 2002-2008 days with both level and flow
  - the floodway: with a daily flow record (--floodway-flow, e.g. 05OJ014,
    1969-1976) its share of the dots is spread along its channel bed, since
    its water's edge was not recorded; the rest go to the river

A barrier across the dam and lock keeps the floods apart; elsewhere water
spreads only through connected ground below each level.

    python -m sheets.flood_particles
    python -m sheets.flood_particles --start 1974-04-01 --end 1974-06-30 \
        --floodway-flow data/lockport/daily_05OJ014.csv
"""

import argparse
import datetime as dt
import os

import contourpy
import numpy as np
from scipy import ndimage

from drawing import hydrometric, terrain
from drawing.svg import PAPER, Svg, north_arrow, points_to_path, scale_bar
from sheets.calendar_cut import true_north_angle


INK = "#111111"


def daily(path, param, start, end, required=True):
    """{date: value} for start..end, linearly filling gaps; and the filled days.
    With required=False, a record with nothing in the period returns ({}, set())."""
    dates, values, _ = hydrometric.load_wsc_daily(path, param)
    pad = dt.timedelta(30)
    known = {d: v for d, v in zip(dates, values) if v == v and start - pad <= d <= end + pad}
    if not known:
        if not required:
            return {}, set()
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

    # Lidar water of the river above and below the dam.
    _, water_up, water_down = terrain.river_reaches(a, dam, args.water_below)

    # The floodway channel bed: ground within a metre or two of its lowest
    # point, traced from a cell inside it (the lidar shows it dry).
    sr, sc = args.floodway_seed[0] - r0, args.floodway_seed[1] - c0
    box = a[max(sr - 30, 0):sr + 30, max(sc - 60, 0):sc + 60]
    k = np.unravel_index(np.nanargmin(box), box.shape)
    fw_seed = (max(sr - 30, 0) + k[0], max(sc - 60, 0) + k[1])
    fw_bed = float(a[fw_seed])
    channel = terrain.flood(a, fw_bed + args.floodway_bed, fw_seed)

    # A barrier across the dam and lock keeps the reaches' floods apart, and
    # the river's floods stay out of the floodway channel.
    barrier = np.zeros(a.shape, bool)
    barrier[max(dam - 2, 0):dam + 3, args.dam_cols[0] - c0:args.dam_cols[1] - c0] = True
    floodable = np.where(barrier | channel, np.inf, a)

    flow, flow_filled = daily(args.flow_csv, "discharge", start, end)
    days = sorted(flow)
    peak_day = max(days, key=flow.get)
    fw_flow = {}
    if args.floodway_flow:
        fw_flow, _ = daily(args.floodway_flow, "discharge", start, end)
        fw_flow = {d: max(0.0, min(v, flow[d])) for d, v in fw_flow.items()}

    # Above the dam: recorded if the gauge covers the period, otherwise
    # estimated from James Avenue through their spring open-water relation.
    level_up, up_filled = daily(args.level_csv, "level", start, end, required=False)
    up_est = None
    if not level_up:
        if not args.james_csv:
            raise SystemExit(f"{args.level_csv} has no level for {start.year}; pass --james-csv to estimate it")
        du, lu, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
        dj, lj, _ = hydrometric.load_wsc_daily(args.james_csv, "level")
        jam = {d: v for d, v in zip(dj, lj) if v == v}
        pairs = [(jam[d], u) for d, u in zip(du, lu) if u == u and d in jam and d.month in (4, 5, 6)]
        rel, spread, (j_lo, j_hi) = hydrometric.binned_relation(*zip(*pairs), lo=args.james_min)
        james, _ = daily(args.james_csv, "level", start, end)
        level_up = {d: rel(james[d]) for d in days}
        up_est = dict(spread=spread, lo=j_lo, hi=j_hi, n=len(pairs),
                      missing=sum(v is None for v in level_up.values()),
                      beyond=sum(james[d] > j_hi for d in days))

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
    for r in reaches:
        r["crest"] = max((d for d in days if r["level"][d] is not None), key=r["level"].get)

    # Dots: each day's volume, gathered along that day's water's edges.
    rng = np.random.default_rng(args.seed)
    xs_all, ys_all, edges = [], [], {}
    channel_weight = channel.astype(float).ravel()

    # Riverbed from the chart soundings (sheet 01, 04): where the water is deep,
    # more of it passes, so part of each day's dots follow depth^(5/3).
    bed = np.full(a.shape, np.nan)
    if args.bed and os.path.exists(args.bed):
        bed = terrain.load_dem(args.bed)[0][r0:r0 + H, c0:c0 + W]
    above_dam = np.arange(H)[:, None] > dam

    def scatter(weight, n):
        if n <= 0 or weight.sum() == 0:
            return
        cells = rng.choice(weight.size, size=n, p=weight / weight.sum())
        rows, cols = np.divmod(cells, W)
        xs_all.append(cols + rng.random(n))
        ys_all.append(rows + rng.random(n))

    for d in days:
        wet = water_up | water_down
        lvl = np.full(a.shape, np.nan)
        for r in reaches:
            level = r["level"][d]
            r_wet = r["water"] if level is None else terrain.flood(floodable, level, r["water"])
            wet = wet | r_wet
            if level is not None:
                lvl = np.where(r_wet, level, lvl)
            if d in (start, r["crest"]) and level is not None:
                edges[(r["name"], d)] = r_wet
        dist = ndimage.distance_transform_edt(wet)         # metres to the nearest dry ground
        weight = np.where(wet & ~barrier, np.exp(-dist / args.front), 0.0).ravel()
        fw = fw_flow.get(d, 0.0)
        n_river = int(round((flow[d] - fw) * 86400 / args.volume))
        depth = np.where(np.isfinite(bed), lvl - bed, lvl - a)
        conv = np.where(wet & ~barrier & ~channel & np.isfinite(depth), np.clip(depth, 0, None) ** (5 / 3), 0.0).ravel()
        n_edge = int(round(n_river * args.edge_share)) if conv.sum() > 0 else n_river
        scatter(weight, n_edge)
        scatter(conv, n_river - n_edge)
        # The floodway's share, spread along its bed: its edge was not recorded.
        scatter(channel_weight, int(round(fw * 86400 / args.volume)))
    xs_all, ys_all = np.concatenate(xs_all), np.concatenate(ys_all)

    # Layout.
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale
    x0, y0 = m + 34.0, m + 50.0
    pw, ph = W * s, H * s

    svg = Svg(Wp, Hp)
    tw = args.graph                             # timeline tick weight, mm
    clip = svg.clip_rect(x0, y0, pw, ph)

    def mm(w):
        return f"{w:.3f}".rstrip("0").rstrip(".") + " mm"

    # Layers by line weight, declared bottom to top so they stack in this order.
    L = dict(
        frame=f"FRAME · {mm(0.15)}",
        contour=f"PLAN · CONTOURS 1 M · {mm(args.contour)}",
        contour5=f"PLAN · CONTOURS 5 M · {mm(args.contour * 1.4)}",
        riverbed=f"PLAN · RIVERBED CONTOURS 1 M (CHART), DASHED · {mm(0.035)}",
        bed=f"PLAN · FLOODWAY BED, DOTTED · {mm(0.1)}",
        dots=f"PLAN · DOTS · {mm(args.dot)}",
        first=f"PLAN · FIRST-DAY EDGES, DASHED · {mm(0.08)}",
        crest=f"PLAN · CREST EDGES · {mm(0.12)}",
        dam=f"PLAN · DAM AND LOCK · {mm(0.35)}",
        fit=f"TIMELINE · FIT LIMIT, DASHED · {mm(tw * 0.8)}",
        ticks=f"TIMELINE · DAILY TICKS · {mm(tw)}",
        peak=f"TIMELINE · PEAK DAY · {mm(tw * 2.5)}",
        share=f"TIMELINE · FLOODWAY SHARE · {mm(tw * 3.2)}",
        axis=f"TIMELINE · AXIS AND SCALE · {mm(tw * 1.6)}",
        scale="SCALE BAR AND NORTH ARROW",
        text="TEXT",
    )
    for name in L.values():
        svg.layer(name)

    def plan(key):
        """Switch to a plan layer, clipped to the plan."""
        svg.layer(L[key])
        svg.open_group(clip_path=clip)

    svg.layer(L["frame"])
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)

    # Ground: faint contours of the river valley, one per metre.
    for lv, lines in terrain.smooth_contours(a, np.arange(216, 232, 1.0)).items():
        d = " ".join(points_to_path(x0 + (ln[::2, 0] + 0.5) * s, y0 + (ln[::2, 1] + 0.5) * s)
                     for ln in lines if len(ln) > 8)
        index = lv % 5 == 0
        plan("contour5" if index else "contour")
        svg.path(d, fill="none", stroke="#a6a6a6", stroke_width=args.contour * (1.4 if index else 1.0),
                 opacity=0.6 if index else 0.4)
        svg.close_group()

    # The riverbed from the chart, as dashed contours every metre.
    if np.isfinite(bed).any():
        plan("riverbed")
        for side, ref in ((above_dam, args.bed_ref[0]), (~above_dam, args.bed_ref[1])):
            z = ndimage.gaussian_filter(np.where(np.isfinite(bed) & side, bed, ref), 2)
            z = np.where(side, z, np.nan)
            for lv in np.arange(np.floor(np.nanmin(bed)), ref - 0.5, 1.0):
                lns = contourpy.contour_generator(z=z, name="serial").lines(lv)
                svg.path(" ".join(points_to_path(x0 + (ln[::2, 0] + 0.5) * s, y0 + (ln[::2, 1] + 0.5) * s)
                                  for ln in lns if len(ln) > 8),
                         fill="none", stroke="#a6a6a6", stroke_width=0.035, stroke_dasharray="0.8 0.5", opacity=0.5)
        svg.close_group()

    # The floodway bed, dotted: where its dots go when its flow is known.
    plan("bed")
    path = " ".join(points_to_path(x0 + (ln[:, 0] + 0.5) * s, y0 + (ln[:, 1] + 0.5) * s)
                    for ln in terrain.mask_lines(channel))
    svg.path(path, fill="none", stroke=INK, stroke_width=0.1, stroke_dasharray="0.3 0.6")
    svg.close_group()

    # The dots, as zero-length round-capped strokes in one path.
    plan("dots")
    d = "".join(f"M{x0 + x * s:.2f},{y0 + y * s:.2f}h0" for x, y in zip(xs_all, ys_all))
    svg.path(d, fill="none", stroke=INK, stroke_width=args.dot, stroke_linecap="round", opacity=0.85)
    svg.close_group()

    # Hairlines: each reach's first and highest edges.
    near_dam = ndimage.binary_dilation(barrier, iterations=2)
    for (name, d), wet in edges.items():
        first = d == start
        style = dict(stroke_width=0.08, stroke_dasharray="0.6 0.6") if first else dict(stroke_width=0.12)
        lines = terrain.mask_lines(wet & ~barrier)
        # Leave the cut along the dam undrawn; it is not a water's edge.
        path = " ".join(points_to_path(
            x0 + (ln[:, 0] + 0.5) * s,
            np.where(near_dam[np.clip(ln[:, 1].round().astype(int), 0, H - 1),
                              np.clip(ln[:, 0].round().astype(int), 0, W - 1)],
                     np.nan, y0 + (ln[:, 1] + 0.5) * s))
            for ln in lines)
        plan("first" if first else "crest")
        svg.path(path, fill="none", stroke=INK, **style)
        svg.close_group()

    # The dam.
    if 0 <= dam < H:
        plan("dam")
        y = y0 + (dam + 0.5) * s
        svg.line(x0 + (args.dam_cols[0] - c0) * s, y, x0 + (args.dam_cols[1] - c0) * s, y,
                 stroke=INK, stroke_width=0.35)
        svg.close_group()

    svg.layer(L["text"])
    for text, row, col, angle in args.label:
        lr, lc = float(row) - r0, float(col) - c0
        if 0 <= lr < H and 0 <= lc < W:
            svg.text(x0 + lc * s, y0 + lr * s + 0.6, text, size=1.7, fill=INK, anchor="middle",
                     spacing=0.2, rotate=float(angle) or None, opacity=0.8)

    # Timeline: a tick per day down the margin, its length the day's flow.
    tx = x0 + pw + 30
    tick_max = min(Wp - m - tx - 40, 120.0)
    qmax = max(flow.values())
    svg.layer(L["text"])
    svg.text(tx, y0 - 7, "DAILY FLOW PASSING LOCKPORT", size=2.6, fill=INK, spacing=0.25)
    svg.text(tx, y0 - 3, f"STATION {hydrometric.station_id(args.flow_csv)} · ONE TICK PER DAY",
             size=2.0, fill=INK, opacity=0.75)
    if fw_flow:
        svg.text(tx, y0 + ph + 16, f"HEAVY: THE FLOODWAY'S SHARE ({hydrometric.station_id(args.floodway_flow)})",
                 size=2.0, fill=INK, opacity=0.85)
    svg.layer(L["axis"])
    svg.line(tx, y0, tx, y0 + ph, stroke=INK, stroke_width=tw * 1.6)
    crest_up = reaches[0]["crest"]
    fw_peak = max(fw_flow, key=fw_flow.get) if fw_flow else None
    for k, d in enumerate(days):
        y = y0 + k / (len(days) - 1) * ph
        svg.layer(L["peak" if d == peak_day else "ticks"])
        svg.line(tx, y, tx + flow[d] / qmax * tick_max, y, stroke=INK,
                 stroke_width=tw * 2.5 if d == peak_day else tw,
                 stroke_dasharray="0.6 0.6" if d in flow_filled else None)
        if fw_flow.get(d, 0) > 0:
            svg.layer(L["share"])
            svg.line(tx, y, tx + fw_flow[d] / qmax * tick_max, y, stroke=INK, stroke_width=tw * 3.2)
        if d.weekday() == 0 or d in (start, end):
            svg.layer(L["ticks"])
            svg.line(tx - 1.5, y, tx, y, stroke=INK, stroke_width=tw)
        svg.layer(L["text"])
        if d == fw_peak:
            svg.text(tx + flow[d] / qmax * tick_max + 2.5, y + 0.8,
                     f"FLOODWAY {fw_flow[d]:,.0f} m³/s", size=2.0, fill=INK, opacity=0.85)
        if d.weekday() == 0 or d in (start, end):
            svg.text(tx - 2.5, y + 0.8, d.strftime("%b %d").upper(), size=2.0, fill=INK, anchor="end")
        if d == peak_day:
            svg.text(tx + tick_max + 2.5, y + 0.8, f"{flow[d]:,.0f} m³/s · {d.strftime('%b %d').upper()}",
                     size=2.2, fill=INK)
        if d == crest_up and d not in (peak_day, fw_peak):
            svg.text(tx + flow[d] / qmax * tick_max + 2.5, y - 1.4,
                     f"CREST ABOVE DAM {'EST. ' if up_est else ''}{level_up[d]:.2f} m", size=2.0, fill=INK, opacity=0.85)
    if beyond:
        bx = tx + q_hi / qmax * tick_max
        svg.layer(L["fit"])
        svg.line(bx, y0, bx, y0 + ph, stroke=INK, stroke_width=tw * 0.8, stroke_dasharray="0.8 1.0")
        svg.layer(L["text"])
        svg.text(bx + 1, y0 + ph + 3.2, f"{q_hi:,.0f} m³/s: HIGHEST FLOW IN THE STAGE-FLOW FIT",
                 size=1.9, fill=INK, opacity=0.75)
    svg.layer(L["axis"])
    svg.line(tx, y0 + ph + 8, tx + 1000 / qmax * tick_max, y0 + ph + 8, stroke=INK, stroke_width=tw * 1.6)
    svg.layer(L["text"])
    svg.text(tx, y0 + ph + 11.5, "1,000 m³/s", size=2.0, fill=INK)

    # Title block.
    n = len(xs_all)
    svg.layer(L["text"])
    tx0, ty0 = m + 8, m + 12
    svg.text(tx0, ty0, "RED RIVER AT ST. ANDREWS LOCK AND DAM", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx0, ty0 + 7, f"THE {start.year} FLOOD — EACH DOT {args.volume:,.0f} M³ OF WATER, "
             f"{start.strftime('%b %d').upper()} – {end.strftime('%b %d').upper()}, 1:{args.scale:,}",
             size=2.4, fill=INK, spacing=0.3)
    side = round(args.volume ** (1 / 3))
    crest_dn = reaches[1]["crest"]
    notes = [
        f"Each day the water that passed Lockport is drawn as dots, one per {args.volume:,.0f} m³ (a cube about {side} m on a side): "
        f"{n:,} dots for {len(days)} days.",
        f"{args.edge_share:.0%} of each day's dots gather along that day's water's edges, thinning over about {args.front:g} m, so they "
        f"pile up where an edge lingered; the rest fall through the water by depth^5/3, where more of it passes. Dashed lines: "
        "the first day's edges; solid: each reach's crest.",
        "Riverbed (fine dashed contours, every metre): depth soundings digitized off a Navionics chart, taken below the summer "
        "pool (chart datum not stated; not a survey).",
        (f"Above the dam: daily water level recorded at {hydrometric.station_id(args.level_csv)} "
         f"(crest {level_up[crest_up]:.2f} m on {crest_up.strftime('%b %d')})." if not up_est else
         f"Above the dam: ESTIMATED from the daily level at James Avenue ({hydrometric.station_id(args.james_csv)}, Winnipeg) through their relation "
         f"in spring open water, {up_est['n']:,} days 1996–2025 (±{up_est['spread']:.1f} m; James Avenue {up_est['lo']:.1f}–{up_est['hi']:.1f} m); "
         f"crest {level_up[crest_up]:.2f} m on {crest_up.strftime('%b %d')}."
         + (f" {up_est['missing']} low-water days without an estimate show the lidar water only." if up_est["missing"] else "")),
        f"Below the dam: ESTIMATED from each day's flow through the stage-flow relation of {hydrometric.station_id(args.flow_csv)}, "
        f"fitted on its 2002–2008 days with both records (±{rmse:.1f} m, flows {q_lo:,.0f}–{q_hi:,.0f} m³/s); "
        f"crest {level_dn[crest_dn]:.2f} m on {crest_dn.strftime('%b %d')}.",
        (f"{len(beyond)} days exceeded the fitted range, up to {qmax:,.0f} m³/s: their levels below the dam are extrapolated."
         if beyond else "All days lie within the fitted range."),
        (f"Floodway: daily flow MEASURED at {hydrometric.station_id(args.floodway_flow)} (peak {fw_flow[fw_peak]:,.0f} m³/s on "
         f"{fw_peak.strftime('%b %d')}). Its share of each day's dots is spread along its bed (dotted outline), since its water's edge "
         "was not recorded; the river gets the rest."
         if fw_flow else
         f"Floodway (dotted outline: its channel bed): it carried water in {start.year}, but no floodway record for that year is on file "
         "(station 05OJ014 reported flow only, 1969–1976), so this sheet cannot place it."),
        "Faint lines: ground contours every metre. Heavy bar: the dam and lock, which no flood crosses here.",
        "SOURCES: Water Survey of Canada stations "
        + ", ".join(hydrometric.station_id(p) for p in
                    [args.level_csv, args.flow_csv] + ([args.james_csv] if up_est else [])
                    + ([args.floodway_flow] if fw_flow else []))
        + "; NRCan HRDEM lidar DTM. Levels applied without datum adjustment; the lidar cannot see under water.",
    ]
    for j, note in enumerate(notes):
        svg.text(tx0, ty0 + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 7 else 0.65)

    svg.layer(L["scale"])
    sb_y = Hp - m - 9
    scale_bar(svg, x0, sb_y, 500 * s, 500, "m", color=INK)
    north_arrow(svg, x0 + 500 * s + 14, sb_y + 0.5, r=3.0, color=INK,
                angle=true_north_angle(transform, crs, r0 + H / 2, c0 + W / 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  ({n:,} dots, {len(days)} days; crest above dam {crest_up} {level_up[crest_up]:.2f} m"
          f"{' (est. from James Ave, ' + str(up_est) + ')' if up_est else ''}, "
          f"below dam est. {crest_dn} {level_dn[crest_dn]:.2f} m; rating ±{rmse:.2f} m on {q_lo:.0f}-{q_hi:.0f} m3/s, "
          f"{len(beyond)} days beyond; floodway bed {fw_bed:.2f} m, {channel.sum()} cells"
          f"{', floodway peak ' + str(fw_peak) + ' ' + format(fw_flow[fw_peak], '.0f') if fw_flow else ''})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily.csv", help="daily level above the dam")
    ap.add_argument("--flow-csv", default="data/lockport/daily_05OJ010.csv",
                    help="daily flow, plus level for the stage-flow relation below the dam")
    ap.add_argument("--james-csv", default="data/lockport/daily_05OJ015.csv",
                    help="James Avenue level, to estimate the level above the dam before its gauge began")
    ap.add_argument("--james-min", type=float, default=224.5,
                    help="lowest James Avenue level (m) used; below it the dam, not the flow, sets the pool")
    ap.add_argument("--floodway-flow", help="daily floodway flow (e.g. data/lockport/daily_05OJ014.csv, 1969-1976)")
    ap.add_argument("--floodway-seed", type=int, nargs=2, default=[2586, 2356], metavar=("ROW", "COL"),
                    help="DEM cell inside the floodway channel bed")
    ap.add_argument("--floodway-bed", type=float, default=1.5,
                    help="m above the channel's lowest point counted as its bed")
    ap.add_argument("--start", default="1997-04-01")
    ap.add_argument("--end", default="1997-06-30")
    ap.add_argument("--volume", type=float, default=100_000, help="m³ of water per dot")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--bed-ref", type=float, nargs=2, default=[222.9, 218.2],
                    help="the bed's reference levels above / below the dam (as in tools/build_bed.py)")
    ap.add_argument("--edge-share", type=float, default=0.6, help="share of each day's dots placed along the water's edges")
    ap.add_argument("--front", type=float, default=12.0, help="m over which each day's dots thin from the edge")
    ap.add_argument("--dot", type=float, default=0.18, help="dot diameter in mm")
    ap.add_argument("--contour", type=float, default=0.035, help="contour line weight in mm (5 m contours 1.4x)")
    ap.add_argument("--graph", type=float, default=0.18, help="timeline tick weight in mm")
    ap.add_argument("--seed", type=int, default=1997)
    # Rows/cols of the DEM: from the floodway outlet north of the dam to
    # ~800 m above it, and east across the floodway channel.
    ap.add_argument("--window", type=int, nargs=4, default=[1150, 2950, 700, 2900],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595],
                    help="DEM columns the dam and lock span (the flood barrier)")
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"))
    ap.add_argument("--scale", type=int, default=4000)
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--out", help="default: outputs/sheets/05_flood_particles_<year>.svg")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"outputs/sheets/05_flood_particles_{args.start[:4]}.svg"
    if args.label is None:
        args.label = [("LOCK", 2200, 1272, 0), ("HENDERSON HWY", 2362, 1952, -90),
                      ("FLOODWAY", 2750, 2470, -90), ("FLOODWAY OUTLET", 1395, 2010, 0)]
    build(args)


if __name__ == "__main__":
    main()
