"""Sheet 01 - Water motion: velocity field at St. Andrews Lock and Dam.

Two plans side by side on the same 1 m lidar terrain, black on white, after
the Bay of Fundy "water motion in a topographical vessel" drawing: a field of
short ticks, each one the water's direction and speed at that point. No banks
are drawn; they appear where the ticks stop.

  - a typical open-water day: the median June-September flow, with water
    levels measured above and below the dam that day; floodway closed
  - the 1997 crest: flow at Lockport, levels above the dam measured and below
    it estimated (sheet 05), the floodway's share from the published peak

The velocities are an ESTIMATE from conveyance (drawing/flowfield.py): each
day's flow spread over its wetted area by depth^(5/3), scaled to the measured
total. The riverbed is not surveyed: depth in the river channel grows from
the lidar water surface toward mid-channel up to an assumed maximum.

    python -m sheets.velocity_field
"""

import argparse
import datetime as dt
import os

import numpy as np
from scipy import ndimage

from drawing import flowfield, hydrometric, terrain
from drawing.svg import PAPER, Svg, north_arrow, points_to_path, scale_bar
from sheets.calendar_cut import true_north_angle


INK = "#111111"


def largest(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == np.argmax(sizes) + 1


def build(args):
    dem, transform, crs = terrain.load_dem(args.dem)
    r0, r1, c0, c1 = args.window
    st = args.cell
    a = dem[r0:r1:st, c0:c1:st]
    full = dem[r0:r1, c0:c1]
    H, W = a.shape
    dam = (args.dam_row - r0) // st
    _, wu, wd = terrain.river_reaches(a, dam, args.water_below, min_cells=150, keep_near=10)
    lidar = wu | wd
    bed = np.full(a.shape, np.nan)
    if args.bed and os.path.exists(args.bed):
        bed = terrain.load_dem(args.bed)[0][r0:r1:st, c0:c1:st]
    bar = np.zeros(a.shape, bool)
    bar[dam - 1:dam + 2, (args.dam_cols[0] - c0) // st:(args.dam_cols[1] - c0) // st] = True

    # Floodway channel bed, as on sheet 05.
    sr, sc = (args.floodway_seed[0] - r0) // st, (args.floodway_seed[1] - c0) // st
    box = a[max(sr - 10, 0):sr + 10, max(sc - 20, 0):sc + 20]
    k = np.unravel_index(np.nanargmin(box), box.shape)
    fw_seed = (max(sr - 10, 0) + k[0], max(sc - 20, 0) + k[1])
    fw_bed = float(a[fw_seed])
    channel = terrain.flood(a, fw_bed + args.floodway_bed, fw_seed)
    corridor = ndimage.binary_dilation(channel, iterations=args.floodway_reach // st)
    floodable = np.where(bar | channel, np.inf, a)

    # The two days.
    dq, q, sq = hydrometric.load_wsc_daily(args.flow_csv, "discharge")
    flow = {d: v for d, v in zip(dq, q) if v == v}
    dh, hh, _ = hydrometric.load_wsc_daily(args.flow_csv, "level")
    below = {d: v for d, v in zip(dh, hh) if v == v}
    du, lu, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
    above = {d: v for d, v in zip(du, lu) if v == v}
    rating, rmse, _ = hydrometric.fit_rating(dq, q, sq, dh, hh)

    typical = dt.date.fromisoformat(args.typical)
    peak = dt.date.fromisoformat(args.peak)
    days = [
        dict(date=typical, q=flow[typical], q_fw=0.0, up=above[typical], dn=below[typical],
             title=f"TYPICAL OPEN-WATER DAY · {typical.strftime('%B %d, %Y').upper()}",
             note=f"median June–September flow 2002–2008; levels measured above ({above[typical]:.2f} m) "
                  f"and below ({below[typical]:.2f} m) the dam; floodway closed"),
        dict(date=peak, q=flow[peak], q_fw=args.floodway_peak, up=above[peak], dn=rating(flow[peak]),
             title=f"THE 1997 CREST · {peak.strftime('%B %d, %Y').upper()}",
             note=f"level above the dam measured ({above[peak]:.2f} m), below it estimated from flow "
                  f"({rating(flow[peak]):.2f} m ±{rmse:.1f}); floodway share {args.floodway_peak:,.0f} m³/s (published peak)"),
    ]

    def wetted(day):
        up = terrain.flood(floodable, day["up"], wu)
        dn = terrain.flood(floodable, day["dn"], wd)
        cross = bar & ndimage.binary_dilation(up, iterations=3) & ndimage.binary_dilation(dn, iterations=3)
        return up, dn, cross

    def depth_of(dom, up, dn, day, extra=None):
        level = np.where(dn | (extra if extra is not None else False), day["dn"], day["up"])
        dist = ndimage.distance_transform_edt(dom) * st
        d = level - a
        proxy = d + np.minimum(args.max_depth, args.bed_slope * dist)
        # Chart-derived bed where it exists, else the assumed deepening.
        d = np.where(lidar, np.where(np.isfinite(bed), np.maximum(level - bed, d), proxy), d)
        d = np.clip(np.nan_to_num(d, nan=0.05), args.min_depth, None)
        # Through the dam the water keeps the depth it has on either side (the
        # lidar shows the structure, not the gate openings).
        return np.where(bar, ndimage.maximum_filter(np.where(bar, 0, d), size=7), d)

    for day in days:
        up, dn, cross = wetted(day)
        dom = largest(up | dn | cross)
        inflow = np.zeros_like(dom)
        inflow[-1] = dom[-1] & ~channel[-1]
        outflow = np.zeros_like(dom)
        outflow[0] = dom[0]
        depth = depth_of(dom, up, dn, day)
        u, v, fn = flowfield.solve(dom, depth, inflow, outflow, day["q"] - day["q_fw"], st)
        wet = dom.copy()
        if day["q_fw"] > 0:
            # The floodway: its channel joined to the river below the dam, through the outlet.
            # Floodway water: its channel filled to an assumed depth over the bed,
            # kept within its corridor; the lidar ground there is real (dry when flown).
            fw_level = fw_bed + args.floodway_depth
            fw_wet = terrain.flood(np.where(corridor & ~bar, a, np.inf), fw_level, fw_seed)
            ch_bridge = (ndimage.binary_dilation(fw_wet, iterations=6) & ndimage.binary_dilation(dn, iterations=6)
                         & (a < max(day["dn"], fw_level) + 1))
            dom_f = largest(fw_wet | dn | ch_bridge)
            in_f = np.zeros_like(dom_f)
            in_f[-1] = dom_f[-1] & fw_wet[-1]
            out_f = np.zeros_like(dom_f)
            out_f[0] = dom_f[0]
            depth_f = np.where(fw_wet | ch_bridge, np.clip(fw_level - a, 0.05, None), depth_of(dom_f, up, dn, day))
            uf, vf, fnf = flowfield.solve(dom_f, depth_f, in_f, out_f, day["q_fw"], st)
            u, v, fn = u + uf, v + vf, fn + fnf
            depth = np.where(dom, depth, depth_f)
            wet = wet | dom_f
            day["fw_cells"] = int(in_f.sum())
        day.update(u=u, v=v, wet=wet, speed=np.hypot(u, v), depth=depth, fn=fn)

    # Streamlines: contours of the stream function, each carrying an equal share
    # of the flow, so they crowd where more water passes and never merge.
    for day in days:
        lines = flowfield.stream_lines(day["u"], day["v"], day["fn"], day["wet"], args.per_line, st)
        day["lines"] = [(rr * st, cc * st, tt * st) for rr, cc, tt in lines]     # metres and seconds
        day["traced_q"] = len({round(float(np.median(cc)), -1) for _, cc, _ in lines})

    def draw_plan(day, x0, y0, rows, cols, scale, chevron_s, every, title=None, labels=True, bar=True):
        """One plan of a day: contours, streamlines, travel-time arrowheads, dam."""
        ra, rb = rows
        ca, cb = cols
        s_ = 1000.0 / scale
        pw_, ph_ = (cb - ca) * s_, (rb - ra) * s_
        clip = svg.clip_rect(x0, y0, pw_, ph_)

        def X(c):
            return x0 + (np.asarray(c) + c0 - ca) * s_

        def Y(r):
            return y0 + (np.asarray(r) + r0 - ra) * s_

        sub = full[ra - r0:rb - r0, ca - c0:cb - c0]
        stp = max(1, int(scale / 2500))
        for lv, lns in terrain.smooth_contours(sub[::stp, ::stp], np.arange(216, 232, 1.0), sigma=1.5).items():
            index = lv % 5 == 0
            svg.layer(L["contour5" if index else "contour"])
            svg.open_group(clip_path=clip)
            svg.path(" ".join(points_to_path(x0 + ln[::2, 0] * stp * s_, y0 + ln[::2, 1] * stp * s_)
                              for ln in lns if len(ln) > 8),
                     fill="none", stroke="#a6a6a6", stroke_width=0.05 if index else 0.035, opacity=0.6 if index else 0.4)
            svg.close_group()

        svg.layer(L["lines"])
        svg.open_group(clip_path=clip)
        svg.path(" ".join(points_to_path(X(cc_), Y(rr_)) for rr_, cc_, _ in day["lines"]),
                 fill="none", stroke=INK, stroke_width=args.line, stroke_linejoin="round", opacity=0.7)
        svg.close_group()

        # Arrowheads at equal travel time: close together where the water is slow.
        heads = []
        size = args.chevron
        for k, (rr_, cc_, tt_) in enumerate(day["lines"]):
            if k % every:
                continue
            for t_mark in np.arange(chevron_s * (0.5 + (k * 0.37) % 1), tt_[-1], chevron_s):
                i = int(np.searchsorted(tt_, t_mark))
                if i < 1 or i >= len(tt_):
                    continue
                px, py = float(X(cc_[i])), float(Y(rr_[i]))
                dx, dy = float(X(cc_[i]) - X(cc_[i - 1])), float(Y(rr_[i]) - Y(rr_[i - 1]))
                n = np.hypot(dx, dy)
                if n == 0:
                    continue
                dx, dy = dx / n, dy / n
                heads.append(f"M{px - size * (dx - 0.55 * dy):.2f},{py - size * (dy + 0.55 * dx):.2f}"
                             f"L{px:.2f},{py:.2f}"
                             f"L{px - size * (dx + 0.55 * dy):.2f},{py - size * (dy - 0.55 * dx):.2f}")
        svg.layer(L["heads"])
        svg.open_group(clip_path=clip)
        svg.path(" ".join(heads), fill="none", stroke=INK, stroke_width=args.line * 1.6,
                 stroke_linejoin="round", stroke_linecap="round")
        svg.close_group()

        svg.layer(L["dam"])
        svg.open_group(clip_path=clip)
        yd = y0 + (args.dam_row - ra) * s_
        svg.line(x0 + (args.dam_cols[0] - ca) * s_, yd, x0 + (args.dam_cols[1] - ca) * s_, yd, stroke=INK,
                 stroke_width=0.35)
        svg.close_group()
        svg.layer(L["frame_plans"])
        svg.rect(x0, y0, pw_, ph_, fill="none", stroke=INK, stroke_width=0.08, opacity=0.6)

        svg.layer(L["text"])
        if title:
            svg.text(x0, y0 - 2.5, title, size=1.9, fill=INK, spacing=0.2)
        if labels:
            for text, row, col, angle in args.label:
                if ra <= row < rb and ca <= col < cb:
                    svg.text(x0 + (col - ca) * s_, y0 + (row - ra) * s_ + 0.6, text, size=1.6, fill=INK,
                             anchor="middle", spacing=0.2, rotate=float(angle) or None, opacity=0.75)
        if bar:
            svg.layer(L["scale"])
            length = 500 if scale >= 4000 else 100
            scale_bar(svg, x0, y0 + ph_ + 3, length * s_, length, "m", color=INK)
            north_arrow(svg, x0 + length * s_ + 12, y0 + ph_ + 3.5, r=2.5, color=INK, angle=north)
        return pw_, ph_

    def flow_arrow(day, x0, y0, scale):
        """A large arrow along the mean flow at the upstream edge: the direction of flow."""
        s_ = 1000.0 / scale
        firsts = [(rr_[min(40, len(rr_) - 1)], cc_[min(40, len(cc_) - 1)], rr_[0], cc_[0])
                  for rr_, cc_, _ in day["lines"] if len(rr_) > 41]
        if not firsts:
            return
        r1_, c1_, r0_, c0_ = np.median(np.array(firsts), axis=0)
        dx, dy = c1_ - c0_, r1_ - r0_
        n = np.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        bx = x0 + (np.median([cc_[0] for _, cc_, _ in day["lines"]])) * s_
        by = y0 + (H * st - 60) * s_
        length = 24.0
        tipx, tipy = bx + dx * length, by + dy * length
        svg.layer(L["arrow"])
        svg.line(bx, by, tipx, tipy, stroke=INK, stroke_width=0.35)
        svg.path(f"M{tipx - 2.2 * (dx - 0.45 * dy):.2f},{tipy - 2.2 * (dy + 0.45 * dx):.2f}L{tipx:.2f},{tipy:.2f}"
                 f"L{tipx - 2.2 * (dx + 0.45 * dy):.2f},{tipy - 2.2 * (dy - 0.45 * dx):.2f}Z", fill=INK, stroke="none")
        svg.layer(L["text"])
        svg.text(bx + 3, by + 3.5, "FLOW", size=2.2, fill=INK, spacing=0.4)

    # Layout: two plans side by side, then the dam and outlet details below.
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale
    pw, ph = W * st * s, H * st * s
    gap = 24.0
    x_left = (Wp - (2 * pw + gap)) / 2                     # the pair of plans centred on the sheet
    x_plans = [x_left, x_left + pw + gap]
    y0 = m + 62
    svg = Svg(Wp, Hp)
    L = dict(frame="FRAME · 0.15 mm", frame_plans="PLAN FRAMES · 0.08 mm",
             contour="CONTOURS 1 M · 0.035 mm", contour5="CONTOURS 5 M · 0.05 mm",
             lines=f"STREAMLINES · {args.line:g} mm", heads=f"ARROWHEADS, EQUAL TRAVEL TIME · {args.line * 1.6:g} mm",
             dam="DAM AND LOCK · 0.35 mm", arrow="FLOW ARROWS · 0.35 mm",
             scale="SCALE BARS AND NORTH ARROWS", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["frame"])
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)
    north = true_north_angle(transform, crs, (r0 + r1) / 2, (c0 + c1) / 2)
    every_main = max(1, int(round(max(len(d["lines"]) for d in days) / args.head_lines)))

    for x0, day in zip(x_plans, days):
        draw_plan(day, x0, y0, (r0, r1), (c0, c1), args.scale, args.chevron_s, every_main)
        flow_arrow(day, x0, y0, args.scale)
        svg.layer(L["text"])
        svg.text(x0, y0 - 9, day["title"], size=2.6, fill=INK, spacing=0.25)
        q_txt = f"{day['q']:,.0f} m³/s at Lockport" + (f" ({day['q'] - day['q_fw']:,.0f} river, "
                                                      f"{day['q_fw']:,.0f} floodway)" if day["q_fw"] else "")
        svg.text(x0, y0 - 5, q_txt + f" · {len(day['lines'])} streamlines", size=2.0, fill=INK)
        svg.text(x0, y0 - 2, day["note"], size=1.6, fill=INK, opacity=0.75)
        sp = day["speed"][day["wet"]]
        svg.text(x0 + pw, y0 + ph + 4, f"SPEED: MEDIAN {np.median(sp):.2f} M/S · FASTEST 1% {np.percentile(sp, 99):.2f} M/S",
                 size=1.8, fill=INK, anchor="end")

    # Details: the dam and the floodway outlet, enlarged, each pair centred
    # under its own day's plan.
    dy0 = y0 + ph + 30
    det_w = (args.detail_dam[3] - args.detail_dam[2]) * 1000.0 / args.detail_scale
    svg.layer(L["text"])
    svg.text(x_plans[0], dy0 - 8, f"DETAILS · 1:{args.detail_scale:,} · ARROWHEADS EVERY {args.detail_chevron_s:g} S OF TRAVEL",
             size=2.2, fill=INK, spacing=0.25)
    for i, (xp, day) in enumerate(zip(x_plans, days)):
        dgap = pw - 2 * det_w
        for j, (place, (ra, rb, ca, cb)) in enumerate((("DAM AND LOCK", args.detail_dam),
                                                         ("FLOODWAY OUTLET", args.detail_outlet))):
            px0 = xp + j * (det_w + dgap)
            every_det = max(1, int(round(len(day["lines"]) / 60)))
            _, dph = draw_plan(day, px0, dy0, (ra, rb), (ca, cb), args.detail_scale, args.detail_chevron_s,
                               every_det, title=f"{place} · {day['date'].strftime('%b %d, %Y').upper()}",
                               bar=(i == 0 and j == 0))
            if place == "FLOODWAY OUTLET" and not day["q_fw"]:
                svg.layer(L["text"])
                svg.text(px0 + det_w / 2, dy0 + dph / 2, "FLOODWAY CLOSED · NO FLOW", size=2.0, fill=INK,
                         anchor="middle", spacing=0.3, opacity=0.7)

    # Legend, top right.
    lx, ly = Wp - m - 200, m + 12
    svg.layer(L["text"])
    svg.text(lx, ly, "READING THE PLANS", size=2.2, fill=INK, spacing=0.25)
    legend = [
        f"Each line carries {args.per_line:g} m³/s: lines crowd where more water passes.",
        f"Arrowheads point downstream, placed every {args.chevron_s / 60:g} minutes of travel on every "
        f"{every_main}{'th' if every_main > 3 else ('nd' if every_main == 2 else 'st')} line:",
        "far apart where the water is fast, close where it is slow.",
        "The river flows north, to Lake Winnipeg.",
    ]
    for j, t in enumerate(legend):
        svg.text(lx, ly + 4 + 2.8 * j, t, size=1.75, fill=INK, opacity=0.85)

    # Title block.
    svg.layer(L["text"])
    tx, ty = m + 8, m + 12
    svg.text(tx, ty, "RED RIVER AT ST. ANDREWS LOCK AND DAM — WATER MOTION", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx, ty + 7, f"ESTIMATED DEPTH-AVERAGED FLOW ON TWO DAYS · 1:{args.scale:,}", size=2.4, fill=INK,
             spacing=0.3)
    notes = [
        "Streamlines of equal discharge, traced through the flow; no banks are drawn — they appear where the lines run. "
        "Faint lines: ground contours every metre (lidar).",
        "ESTIMATE, not a hydraulic model: each day's flow spread over its wetted area in proportion to depth^5/3 (Manning), "
        "scaled to the measured total. Eddies, jets through the dam gates and backwater are not modelled.",
        "Riverbed: depth soundings digitized off a Navionics chart (feet, taken below the summer pool; chart datum not stated; "
        f"not a survey); beyond the chart, depth grows {args.bed_slope:g} m per metre from the bank to {args.max_depth:g} m more. "
        "Wetted areas as on sheet 05.",
        "Flow at Lockport (05OJ010) includes the floodway, which joins below the dam. On the typical day the floodway is "
        f"closed; in 1997 its share is the published peak (about 1,800 m³/s), carried by the floodway filled {args.floodway_depth:g} m "
        "over its bed (assumed) to its outlet.",
        "SOURCES: Water Survey of Canada stations 05OJ010 and 05OJ021; NRCan HRDEM lidar DTM; floodway peak: Government of "
        "Manitoba / published accounts of the 1997 flood.",
    ]
    for j, note in enumerate(notes):
        svg.text(tx, ty + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 4 else 0.65)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    for day in days:
        sp = day["speed"][day["wet"]]
        print(f"{day['date']}: Q {day['q']:.0f} (floodway {day['q_fw']:.0f}), wet {day['wet'].sum() * st * st / 1e4:.0f} ha, "
              f"speed median {np.median(sp):.2f} p99 {np.percentile(sp, 99):.2f}"
              + (f", floodway inflow cells {day.get('fw_cells')}" if day["q_fw"] else ""))
    print(f"wrote {args.out}  (streamline pieces: {[len(d['lines']) for d in days]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--flow-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--level-csv", default="data/lockport/daily.csv")
    ap.add_argument("--typical", default="2004-07-09", help="a day with median open-water flow")
    ap.add_argument("--peak", default="1997-05-04")
    ap.add_argument("--floodway-peak", type=float, default=1800.0, help="m³/s through the floodway on the peak day")
    ap.add_argument("--window", type=int, nargs=4, default=[1150, 2950, 700, 2700])
    ap.add_argument("--cell", type=int, default=3, help="m per solver cell")
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--water-below", type=float, default=223.2)
    ap.add_argument("--floodway-seed", type=int, nargs=2, default=[2586, 2356])
    ap.add_argument("--floodway-bed", type=float, default=1.5)
    ap.add_argument("--floodway-depth", type=float, default=7.0,
                    help="m of water assumed over the floodway bed at the peak")
    ap.add_argument("--floodway-reach", type=int, default=90, help="m either side of the bed the floodway may fill")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--min-depth", type=float, default=0.3, help="shallowest water depth used, m")
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--bed-slope", type=float, default=0.08)
    ap.add_argument("--per-line", type=float, default=16.0, help="m³/s carried by each streamline")
    ap.add_argument("--line", type=float, default=0.06, help="streamline weight, mm")
    ap.add_argument("--chevron", type=float, default=0.9, help="arrowhead size, mm")
    ap.add_argument("--chevron-s", type=float, default=180.0, help="seconds of travel between arrowheads (plans)")
    ap.add_argument("--head-lines", type=float, default=22, help="about how many lines per plan carry arrowheads")
    ap.add_argument("--detail-scale", type=int, default=2500)
    ap.add_argument("--detail-chevron-s", type=float, default=60.0)
    ap.add_argument("--detail-dam", type=int, nargs=4, default=[2020, 2250, 1180, 1640], metavar=("R0", "R1", "C0", "C1"))
    ap.add_argument("--detail-outlet", type=int, nargs=4, default=[1300, 1530, 1760, 2220], metavar=("R0", "R1", "C0", "C1"))
    ap.add_argument("--label", nargs=4, action="append", metavar=("TEXT", "ROW", "COL", "ANGLE"))
    ap.add_argument("--scale", type=int, default=5000)
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--out", default="outputs/sheets/01_velocity_field.svg")
    args = ap.parse_args()
    if args.label is None:
        args.label = [("LOCK", 2200, 1200, -90), ("FLOODWAY", 2750, 2470, -90), ("FLOODWAY OUTLET", 1330, 2080, 0)]
    build(args)


if __name__ == "__main__":
    main()
