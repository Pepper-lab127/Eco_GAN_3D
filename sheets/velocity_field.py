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
        d = np.where(lidar, d + np.minimum(args.max_depth, args.bed_slope * dist), d)
        return np.clip(np.nan_to_num(d, nan=0.05), 0.05, None)

    for day in days:
        up, dn, cross = wetted(day)
        dom = largest(up | dn | cross)
        inflow = np.zeros_like(dom)
        inflow[-1] = dom[-1] & ~channel[-1]
        outflow = np.zeros_like(dom)
        outflow[0] = dom[0]
        depth = depth_of(dom, up, dn, day)
        u, v = flowfield.solve(dom, depth, inflow, outflow, day["q"] - day["q_fw"], st)
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
            uf, vf = flowfield.solve(dom_f, depth_f, in_f, out_f, day["q_fw"], st)
            u, v = u + uf, v + vf
            wet = wet | dom_f
            day["fw_cells"] = int(in_f.sum())
        day.update(u=u, v=v, wet=wet, speed=np.hypot(u, v))

    # Layout: two plans side by side.
    Wp, Hp = PAPER[args.paper]
    m = 14.0
    s = 1000.0 / args.scale
    pw, ph = W * st * s, H * st * s
    gap = 24.0
    x_plans = [m + 30, m + 30 + pw + gap]
    y0 = m + 62
    svg = Svg(Wp, Hp)
    L = dict(frame="FRAME · 0.15 mm", contour="CONTOURS 1 M · 0.06 mm", contour5="CONTOURS 5 M · 0.1 mm",
             ticks=f"VELOCITY TICKS · {args.tick:g} mm", dam="DAM AND LOCK · 0.35 mm",
             legend="LEGEND · TICKS", scale="SCALE BARS AND NORTH ARROWS", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["frame"])
    svg.rect(m, m, Wp - 2 * m, Hp - 2 * m, fill="none", stroke=INK, stroke_width=0.15)

    contours = terrain.smooth_contours(full[::2, ::2], np.arange(216, 232, 1.0), sigma=1.5)
    north = true_north_angle(transform, crs, (r0 + r1) / 2, (c0 + c1) / 2)
    vmax_all = max(float(np.percentile(day["speed"][day["wet"]], 99)) for day in days)
    rng = np.random.default_rng(args.seed)

    for x0, day in zip(x_plans, days):
        clip = svg.clip_rect(x0, y0, pw, ph)
        for lv, lines in contours.items():
            index = lv % 5 == 0
            svg.layer(L["contour5" if index else "contour"])
            svg.open_group(clip_path=clip)
            svg.path(" ".join(points_to_path(x0 + ln[::2, 0] * 2 * s, y0 + ln[::2, 1] * 2 * s)
                              for ln in lines if len(ln) > 8),
                     fill="none", stroke="#8a8a8a", stroke_width=0.1 if index else 0.06, opacity=0.8 if index else 0.55)
            svg.close_group()

        # Ticks: on a jittered grid, each from its point downstream, length set by speed.
        sp = day["speed"]
        parts = []
        for rr in range(0, H, args.spacing):
            for cc in range(0, W, args.spacing):
                r_ = min(H - 1, rr + int(rng.integers(0, args.spacing)))
                c_ = min(W - 1, cc + int(rng.integers(0, args.spacing)))
                if not day["wet"][r_, c_] or sp[r_, c_] < 0.01:
                    continue
                vx, vy = day["u"][r_, c_], day["v"][r_, c_]
                length = min(args.tick_max, args.tick_k * np.sqrt(sp[r_, c_])) + 0.15
                ux, uy = vx / sp[r_, c_], vy / sp[r_, c_]
                px, py = x0 + (c_ + 0.5) * st * s, y0 + (r_ + 0.5) * st * s
                parts.append(f"M{px:.2f},{py:.2f}l{ux * length:.2f},{uy * length:.2f}")
        svg.layer(L["ticks"])
        svg.open_group(clip_path=clip)
        svg.path(" ".join(parts), fill="none", stroke=INK, stroke_width=args.tick, stroke_linecap="round")
        svg.close_group()

        svg.layer(L["dam"])
        yd = y0 + (args.dam_row - r0) * s
        svg.line(x0 + (args.dam_cols[0] - c0) * s, yd, x0 + (args.dam_cols[1] - c0) * s, yd, stroke=INK,
                 stroke_width=0.35)

        svg.layer(L["text"])
        svg.text(x0, y0 - 9, day["title"], size=2.6, fill=INK, spacing=0.25)
        q_txt = f"{day['q']:,.0f} m³/s at Lockport" + (f" ({day['q'] - day['q_fw']:,.0f} river, "
                                                      f"{day['q_fw']:,.0f} floodway)" if day["q_fw"] else "")
        svg.text(x0, y0 - 5, q_txt, size=2.0, fill=INK)
        svg.text(x0, y0 - 2, day["note"], size=1.6, fill=INK, opacity=0.75)
        wet_sp = sp[day["wet"]]
        svg.text(x0 + pw, y0 + ph + 4, f"SPEED: MEDIAN {np.median(wet_sp):.2f} M/S · FASTEST 1% {np.percentile(wet_sp, 99):.2f} M/S",
                 size=1.8, fill=INK, anchor="end")
        for text, row, col, angle in args.label:
            lr, lc = float(row) - r0, float(col) - c0
            if 0 <= lr < H * st and 0 <= lc < W * st:
                svg.text(x0 + lc * s, y0 + lr * s + 0.6, text, size=1.6, fill=INK, anchor="middle", spacing=0.2,
                         rotate=float(angle) or None, opacity=0.75)
        svg.layer(L["scale"])
        scale_bar(svg, x0, y0 + ph + 8, 500 * s, 500, "m", color=INK)
        north_arrow(svg, x0 + 500 * s + 12, y0 + ph + 8.5, r=2.5, color=INK, angle=north)

    # Legend: ticks at set speeds.
    lx, ly = x_plans[0], y0 + ph + 26
    svg.layer(L["text"])
    svg.text(lx, ly, "TICK LENGTH ∝ √SPEED", size=2.0, fill=INK, spacing=0.2)
    svg.layer(L["legend"])
    for i, val in enumerate((0.1, 0.25, 0.5, 1.0, 2.0)):
        x = lx + 4 + i * 22
        length = min(args.tick_max, args.tick_k * np.sqrt(val)) + 0.15
        svg.line(x, ly + 5, x + length, ly + 5, stroke=INK, stroke_width=args.tick, stroke_linecap="round")
        svg.layer(L["text"])
        svg.text(x, ly + 9, f"{val:g} m/s", size=1.7, fill=INK)
        svg.layer(L["legend"])

    # Title block.
    svg.layer(L["text"])
    tx, ty = m + 8, m + 12
    svg.text(tx, ty, "RED RIVER AT ST. ANDREWS LOCK AND DAM — WATER MOTION", size=5.2, fill=INK, spacing=0.4)
    svg.text(tx, ty + 7, f"ESTIMATED DEPTH-AVERAGED VELOCITY ON TWO DAYS · 1:{args.scale:,}", size=2.4, fill=INK,
             spacing=0.3)
    notes = [
        f"Each tick is the water's direction and speed at a point, every {args.spacing * st} m (jittered); no banks are drawn — "
        "they appear where the ticks stop. Faint lines: ground contours every metre (lidar).",
        "ESTIMATE, not a hydraulic model: each day's flow spread over its wetted area in proportion to depth^5/3 (Manning), "
        "scaled to the measured total. Eddies, jets through the dam gates and backwater are not modelled.",
        f"The riverbed is not surveyed: in the channel, depth grows from the lidar water surface by {args.bed_slope:g} m per metre "
        f"from the bank, to at most {args.max_depth:g} m more. Wetted areas as on sheet 05; water passes the dam where the river "
        "meets it on both sides.",
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
    print(f"wrote {args.out}  (vmax p99 {vmax_all:.2f})")


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
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--bed-slope", type=float, default=0.08)
    ap.add_argument("--spacing", type=int, default=4, help="solver cells between ticks")
    ap.add_argument("--tick", type=float, default=0.12, help="tick line weight, mm")
    ap.add_argument("--tick-k", type=float, default=4.5, help="tick length in mm = k * sqrt(speed in m/s)")
    ap.add_argument("--tick-max", type=float, default=9.0, help="longest tick, mm")
    ap.add_argument("--seed", type=int, default=11)
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
