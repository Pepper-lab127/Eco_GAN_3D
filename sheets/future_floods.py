"""Sheet 06 - Future floods: the annual chance of the river reaching the ground.

Every metre of ground around the lock and dam is drawn by the chance, in any
year, that the Red River's spring flood reaches it - dots as dense as that
chance - with the edges of the 1-in-10, 1-in-100 and 1-in-700 year floods,
and the 1-in-100 edge in a drier and a wetter future.

  - flood frequency: annual peak daily flow at Lockport (05OJ010, 1963-2007,
    44 complete years), Gumbel (EV1) distribution fitted by the method of
    moments, with a bootstrap range. (GEV and log-Pearson III fits to these
    years are bounded above and make 1997 a 1-in-1,000+ year flood, which
    is not credible; Gumbel puts it near 1-in-170.)
  - flow to level: below the dam the stage-flow relation of 05OJ010
    (2002-2008); above it, 05OJ021's level against Lockport's flow on
    open-water days with flows over 1,500 m3/s. Beyond the largest flows
    measured both are extrapolated.
  - where water reaches: the lidar ground flooded from each reach at rising
    levels, through connected ground only (as sheet 05); the dam keeps the
    reaches apart and the floodway channel is left out.
  - the future: published climate projections for Red River spring floods
    show no clear trend in the ensemble median but a wide spread
    (Rasmussen and others 2015); peak flows 20% lower and 20% higher are
    drawn as a sensitivity range, not a forecast.

    python -m sheets.future_floods
"""

import argparse
import math
import os

import numpy as np
from scipy import ndimage

from drawing import hydrometric, terrain
from drawing.svg import PAPER, Svg, north_arrow, points_to_path, scale_bar
from sheets.calendar_cut import true_north_angle

INK = "#111111"
EULER = 0.5772156649


def gumbel_fit(x):
    a = np.std(x, ddof=1) * math.sqrt(6) / math.pi
    return float(np.mean(x) - EULER * a), float(a)


def gumbel_q(u, a, T):
    return u - a * np.log(-np.log(1 - 1 / np.asarray(T, float)))


def gumbel_exceed(u, a, q):
    return 1 - np.exp(-np.exp(-(np.asarray(q, float) - u) / a))


def build(args):
    # ------------------------------------------------------------ the record
    dq, q, sq = hydrometric.load_wsc_daily(args.flow_csv, "discharge")
    peaks, days_in = {}, {}
    peak_day = {}
    for d, v in zip(dq, q):
        if v == v:
            days_in[d.year] = days_in.get(d.year, 0) + 1
            if v > peaks.get(d.year, -1):
                peaks[d.year], peak_day[d.year] = v, d
    years = [y for y in sorted(peaks) if days_in[y] >= args.min_days]
    x = np.array([peaks[y] for y in years])
    u, a = gumbel_fit(x)
    rng = np.random.default_rng(args.seed)
    boot = np.array([gumbel_fit(rng.choice(x, len(x))) for _ in range(args.boot)])
    Ts = np.array([2, 5, 10, 25, 50, 100, 200, 500, 700, 1000])
    q_now = gumbel_q(u, a, Ts)
    q_band = np.percentile([gumbel_q(bu, ba, Ts) for bu, ba in boot], [5, 95], axis=0)

    # ------------------------------------------------------------ levels
    dh, h, _ = hydrometric.load_wsc_daily(args.flow_csv, "level")
    rating, rmse_dn, (q_lo, q_hi) = hydrometric.fit_rating(dq, q, sq, dh, h)
    du, lu, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
    qd = {d: (v, s) for d, v, s in zip(dq, q, sq)}
    pairs = np.array([(qd[d][0], z) for d, z in zip(du, lu)
                      if z == z and d in qd and qd[d][0] == qd[d][0] and qd[d][1] != "B" and qd[d][0] > args.up_min_q])
    c_up = np.polyfit(np.log(pairs[:, 0]), pairs[:, 1], 1)
    rmse_up = float(np.std(pairs[:, 1] - np.polyval(c_up, np.log(pairs[:, 0]))))
    q_up_max = float(pairs[:, 0].max())
    q_grid = np.linspace(100, 9000, 900)
    lvl_dn = np.array([rating(v) for v in q_grid])
    lvl_up = np.polyval(c_up, np.log(q_grid))

    # ------------------------------------------------------------ the ground
    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    g = dem[r0:r1, c0:c1]
    st = args.cell
    gs = g[::st, ::st]
    H, W = gs.shape
    dam = (args.dam_row - r0) // st
    _, w_up, w_dn = terrain.river_reaches(gs, dam, args.water_below, min_cells=150, keep_near=10)
    barrier = np.zeros(gs.shape, bool)
    barrier[max(dam - 1, 0):dam + 2, (args.dam_cols[0] - c0) // st:(args.dam_cols[1] - c0) // st] = True
    # The floodway channel, traced in the larger sheet-05 window and cropped.
    R0, R1, C0, C1 = args.channel_window
    big = dem[R0:R1:st, C0:C1:st]
    sr, sc = (args.floodway_seed[0] - R0) // st, (args.floodway_seed[1] - C0) // st
    box = big[max(sr - 15, 0):sr + 15, max(sc - 30, 0):sc + 30]
    k = np.unravel_index(np.nanargmin(box), box.shape)
    fw_seed = (max(sr - 15, 0) + k[0], max(sc - 30, 0) + k[1])
    ch_big = terrain.flood(big, float(big[fw_seed]) + args.floodway_bed, fw_seed)
    channel = np.zeros(gs.shape, bool)
    oy, ox = (r0 - R0) // st, (c0 - C0) // st
    sub = ch_big[oy:oy + H, ox:ox + W]
    channel[:sub.shape[0], :sub.shape[1]] = sub
    floodable = np.where(barrier | channel, np.inf, gs)

    def thresholds(water, lo):
        """Lowest water level (m) at which each cell is reached from `water`."""
        thr = np.full(gs.shape, np.inf)
        thr[water] = -np.inf
        for lv in np.arange(lo, args.z_max + 1e-6, args.step):
            wet = terrain.flood(floodable, lv, water)
            thr[wet & np.isinf(thr) & (thr > 0)] = lv
        return thr

    thr_up = thresholds(w_up, float(np.nanmedian(gs[w_up])))
    thr_dn = thresholds(w_dn, float(np.nanmedian(gs[w_dn])))

    def flow_needed(thr, lvl):
        out = np.interp(thr, lvl, q_grid, left=0.0, right=np.inf)
        return np.where(np.isneginf(thr), 0.0, np.where(np.isinf(thr), np.inf, out))

    qn_up, qn_dn = flow_needed(thr_up, lvl_up), flow_needed(thr_dn, lvl_dn)

    def chance(factor):
        p_up = gumbel_exceed(u * factor, a * factor, qn_up)
        p_dn = gumbel_exceed(u * factor, a * factor, qn_dn)
        p = np.maximum(np.where(np.isinf(qn_up), 0, p_up), np.where(np.isinf(qn_dn), 0, p_dn))
        return np.where(w_up | w_dn, 1.0, p)

    p_now, p_dry, p_wet = chance(1.0), chance(1 - args.change), chance(1 + args.change)

    # ------------------------------------------------------------ sheet
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale
    x0, y0 = m + 34.0, m + 50.0
    pw, ph = (c1 - c0) * s, (r1 - r0) * s
    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(frame="FRAME · 0.15 mm", contour="CONTOURS 1 M · 0.035 mm", contour5="CONTOURS 5 M · 0.05 mm",
             water="RIVER AT LIDAR LEVEL (ALWAYS WET) · 0.1 mm", dots=f"ANNUAL CHANCE ON LAND, DOTS · {args.dot} mm",
             e10="EDGE 1-IN-10 · 0.08 mm", e100="EDGE 1-IN-100 · 0.15 mm", e700="EDGE 1-IN-700, EXTRAPOLATED, DASHED · 0.12 mm",
             dry="EDGE 1-IN-100, PEAKS 20% LOWER, DOTTED · 0.12 mm", wet="EDGE 1-IN-100, PEAKS 20% HIGHER, DASHED · 0.12 mm",
             dam="DAM AND LOCK · 0.35 mm", ticks="FREQUENCY · ANNUAL PEAKS, ONE TICK PER YEAR · 0.18 mm",
             curves="FREQUENCY · FITTED CURVES · 0.1–0.2 mm", axis="FREQUENCY · AXIS · 0.25 mm",
             scale="SCALE BAR AND NORTH ARROW", text="TEXT")
    for name in L.values():
        svg.layer(name)
    clip = svg.clip_rect(x0, y0, pw, ph)

    def plan(key):
        svg.layer(L[key])
        svg.open_group(clip_path=clip)

    if args.annotate:
        svg.layer(L["frame"])
        svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)

    for lv, lines in terrain.smooth_contours(g[::2, ::2], np.arange(216, 233, 1.0)).items():
        index = lv % 5 == 0
        plan("contour5" if index else "contour")
        svg.path(" ".join(points_to_path(x0 + ln[::2, 0] * 2 * s, y0 + ln[::2, 1] * 2 * s) for ln in lines if len(ln) > 8),
                 fill="none", stroke="#a6a6a6", stroke_width=0.05 if index else 0.035, opacity=0.6 if index else 0.4)
        svg.close_group()

    # Dots: in each cell as many as its annual chance of flooding allows.
    land = ~(w_up | w_dn)
    lam = np.where(land, p_now, 0.0) * (st * st) / args.m2_per_dot
    n = rng.poisson(lam)
    rr, cc = np.nonzero(n)
    reps = n[rr, cc]
    ry = np.repeat(rr, reps) + rng.random(reps.sum())
    rx = np.repeat(cc, reps) + rng.random(reps.sum())
    plan("dots")
    svg.path("".join(f"M{x0 + xx * st * s:.2f},{y0 + yy * st * s:.2f}h0" for xx, yy in zip(rx, ry)), fill="none",
             stroke=INK, stroke_width=args.dot, stroke_linecap="round", opacity=0.85)
    svg.close_group()

    near_dam = ndimage.binary_dilation(barrier, iterations=2)

    def edge(mask, key, **style):
        lines = terrain.mask_lines(mask & ~barrier)
        path = " ".join(points_to_path(
            x0 + (ln[:, 0] + 0.5) * st * s,
            np.where(near_dam[np.clip(ln[:, 1].round().astype(int), 0, H - 1), np.clip(ln[:, 0].round().astype(int), 0, W - 1)],
                     np.nan, y0 + (ln[:, 1] + 0.5) * st * s)) for ln in lines if len(ln) > 6)
        plan(key)
        svg.path(path, fill="none", stroke=INK, **style)
        svg.close_group()

    smooth = lambda p: ndimage.gaussian_filter(p, 1.0)       # noqa: E731
    edge(w_up | w_dn, "water", stroke_width=0.1)
    edge(smooth(p_now) >= 1 / 10, "e10", stroke_width=0.08)
    edge(smooth(p_now) >= 1 / 100, "e100", stroke_width=0.15)
    edge(smooth(p_now) >= 1 / 700, "e700", stroke_width=0.12, stroke_dasharray="1.2 0.7")
    edge(smooth(p_dry) >= 1 / 100, "dry", stroke_width=0.12, stroke_dasharray="0.15 0.5", stroke_linecap="round")
    edge(smooth(p_wet) >= 1 / 100, "wet", stroke_width=0.12, stroke_dasharray="2 0.6 0.3 0.6")

    plan("dam")
    yd = y0 + (args.dam_row - r0) * s
    svg.line(x0 + (args.dam_cols[0] - c0) * s, yd, x0 + (args.dam_cols[1] - c0) * s, yd, stroke=INK, stroke_width=0.35)
    svg.close_group()

    svg.layer(L["text"])
    for text, row, col, ang in args.label:
        lr, lc = float(row) - r0, float(col) - c0
        if not (0 <= lr < r1 - r0 and 0 <= lc < c1 - c0):
            continue
        svg.text(x0 + lc * s, y0 + lr * s + 0.6, text, size=1.7, fill=INK, anchor="middle", spacing=0.2,
                 rotate=float(ang) or None, opacity=0.8)

    # ------------------------------------------------------------ frequency margin
    # Return period up the margin (log), flow across: one tick per year's peak
    # at its plotting position; the fitted curve, its bootstrap range and the
    # drier and wetter futures as lines; the flows of the drawn edges marked.
    tx = x0 + pw + 30
    tick_max = min(Wp - m - tx - 45, 110.0)
    qmax = 7000.0
    T_lo, T_hi = 1.0, 1000.0

    def ty(T):
        return y0 + ph - (math.log10(T) - math.log10(T_lo)) / (math.log10(T_hi) - math.log10(T_lo)) * ph

    def qx(v):
        return tx + v / qmax * tick_max

    svg.layer(L["axis"])
    svg.line(tx, y0, tx, y0 + ph, stroke=INK, stroke_width=0.25)
    order = np.argsort(-x)
    svg.layer(L["ticks"])
    for rank, i in enumerate(order, start=1):
        T = (len(x) + 1) / rank
        svg.line(tx, ty(T), qx(x[i]), ty(T), stroke=INK, stroke_width=0.18)
    TT = np.logspace(math.log10(1.05), math.log10(T_hi), 120)

    def curve(qs, **style):
        pts = [(qx(v), ty(T)) for v, T in zip(qs, TT) if v <= qmax]
        svg.path("M" + "L".join(f"{xx:.2f},{yy:.2f}" for xx, yy in pts), fill="none", stroke=INK, **style)

    svg.layer(L["curves"])
    curve(gumbel_q(u, a, TT), stroke_width=0.2)
    band = np.percentile([gumbel_q(bu, ba, TT) for bu, ba in boot], [5, 95], axis=0)
    curve(band[0], stroke_width=0.06, stroke_dasharray="0.15 0.4", stroke_linecap="round")
    curve(band[1], stroke_width=0.06, stroke_dasharray="0.15 0.4", stroke_linecap="round")
    curve(gumbel_q(u, a, TT) * (1 - args.change), stroke_width=0.12, stroke_dasharray="0.15 0.5", stroke_linecap="round")
    curve(gumbel_q(u, a, TT) * (1 + args.change), stroke_width=0.12, stroke_dasharray="2 0.6 0.3 0.6")
    svg.line(qx(q_hi), y0, qx(q_hi), y0 + ph, stroke=INK, stroke_width=0.06, stroke_dasharray="0.6 0.6", opacity=0.6)
    svg.line(qx(q_up_max), y0, qx(q_up_max), y0 + ph, stroke=INK, stroke_width=0.06, stroke_dasharray="0.6 0.6",
             opacity=0.6)

    svg.layer(L["text"])
    for T in (2, 10, 100, 700, 1000):
        svg.line(tx - 1.5, ty(T), tx, ty(T), stroke=INK, stroke_width=0.18)
        svg.text(tx - 2.5, ty(T) + 0.7, f"1 IN {T}", size=1.9, fill=INK, anchor="end")
    for i in order[:5]:
        T = (len(x) + 1) / (list(order).index(i) + 1)
        svg.text(qx(x[i]) + 1.5, ty(T) + 0.7, f"{years[i]} · {x[i]:,.0f} m³/s", size=1.7, fill=INK, opacity=0.85)
    for T in (10, 100, 700):
        v = gumbel_q(u, a, T)
        svg.text(qx(v) + 1.5, ty(T) - 1.2, f"{v:,.0f}", size=1.6, fill=INK, opacity=0.7)
    svg.text(qx(q_hi) + 0.8, y0 + ph + 3.0, "LARGEST FLOW IN THE STAGE–FLOW FIT, BELOW THE DAM", size=1.4, fill=INK,
             opacity=0.6)
    svg.text(qx(q_up_max) + 0.8, y0 + ph + 5.2, "LARGEST MEASURED, ABOVE THE DAM", size=1.4, fill=INK, opacity=0.6)
    svg.layer(L["scale"])
    svg.line(tx, y0 + ph + 9, qx(1000), y0 + ph + 9, stroke=INK, stroke_width=0.25)
    svg.text(tx, y0 + ph + 12, "1,000 m³/s", size=1.8, fill=INK)

    scale_bar(svg, x0, Hp - m - 9, 200 * s, 200, "m", color=INK)
    north_arrow(svg, x0 + 200 * s + 12, Hp - m - 8.5, r=2.5, color=INK,
                angle=true_north_angle(transform, crs, (r0 + r1) / 2, (c0 + c1) / 2))

    annot(m + 8, m + 12, "RED RIVER AT ST. ANDREWS LOCK AND DAM — FUTURE FLOODS", size=5.2, fill=INK, spacing=0.4)
    annot(m + 8, m + 19, f"ANNUAL CHANCE OF FLOODING · 1:{args.scale:,}", size=2.4, fill=INK, spacing=0.3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"Gumbel u {u:.0f} a {a:.0f}; " + ", ".join(f"T{T}: {v:.0f} ({lo:.0f}-{hi:.0f})" for T, v, lo, hi in
                                                    zip(Ts, q_now, q_band[0], q_band[1])))
    print(f"1997 T={1 / gumbel_exceed(u, a, 4320):.0f}; up fit rmse {rmse_up:.2f} on Q>{args.up_min_q}, "
          f"max {q_up_max:.0f}; dn rmse {rmse_dn:.2f} to {q_hi:.0f}")
    for T in (10, 100, 700):
        v = gumbel_q(u, a, T)
        print(f"  T{T}: Q {v:.0f}, above dam {np.polyval(c_up, math.log(v)):.2f} m, below {rating(v):.2f} m")
    print(f"wrote {args.out} ({reps.sum():,} dots)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--flow-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--level-csv", default="data/lockport/daily.csv")
    ap.add_argument("--min-days", type=int, default=300, help="days of record a year needs to count")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--up-min-q", type=float, default=1500.0)
    ap.add_argument("--change", type=float, default=0.2, help="future change in peak flows drawn either way")
    ap.add_argument("--window", type=int, nargs=4, default=[1650, 2700, 1000, 2150])
    ap.add_argument("--cell", type=int, default=1)
    ap.add_argument("--step", type=float, default=0.1, help="m between flood levels tested")
    ap.add_argument("--z-max", type=float, default=230.0)
    ap.add_argument("--channel-window", type=int, nargs=4, default=[1150, 2950, 700, 2900])
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--floodway-seed", type=int, nargs=2, default=[2586, 2356])
    ap.add_argument("--floodway-bed", type=float, default=1.5)
    ap.add_argument("--m2-per-dot", type=float, default=1.2, help="m² of ground per dot where flooding is certain")
    ap.add_argument("--dot", type=float, default=0.18)
    ap.add_argument("--scale", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=2100)
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"))
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--annotate", action="store_true", help="also print a title")
    ap.add_argument("--out", default="outputs/sheets/06_future_floods.svg")
    args = ap.parse_args()
    if args.label is None:
        args.label = [("LOCK", 2200, 1272, 0), ("HENDERSON HWY", 2362, 1952, -90),
                      ("FLOODWAY", 2750, 2470, -90), ("FLOODWAY OUTLET", 1395, 2010, 0)]
    build(args)


if __name__ == "__main__":
    main()
