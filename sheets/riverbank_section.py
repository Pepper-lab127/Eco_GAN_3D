"""Sheet 04b - Riverbank section.

A tall section through the wooded east bank of the Red River, on the same
line as the winter section (about 300 m below St. Andrews Lock and Dam), at
true scale (no vertical exaggeration):

  - sky: every hour's wind at Winnipeg airport in 2024, each hour a scatter of
    points along its west-east travel; height in the sky is time
  - the air at the bank, from the ground up to the sky band: the same hours,
    thinned where the trees and the bank upwind shelter it (drawing/wind.py)
  - trees by species, grown to each species' habit and fitted to the lidar's
    height and crown width; species inferred from how often the ground at
    each tree floods (drawing/trees.py); roots a diagram by species
  - water: every open-water day's level below the dam (05OJ010, 2002-2008)
  - lake clay, till, shale and carbonate aquifer from two well logs
  - below: a plan of the year's wind at 2 m around the cut, a wind rose of
    every hour, and the key plan

    python -m sheets.riverbank_section
"""

import argparse
import datetime as dt
import math
import os

import numpy as np
from scipy import ndimage

from drawing import climate, geology, hydrometric, terrain, trees, wind
from drawing.keyplan import key_plan
from drawing.section import Section, datum_grid, profile
from sheets.calendar_cut import true_north_angle
from drawing.svg import PAPER, Svg, north_arrow, points_to_path, scale_bar


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


def dots(svg, layer, xs, ys, sizes_mm, tier, op, clip=None, streak=None):
    """Round dots (or short streaks, dx/dy in mm) in one path, optionally blurred."""
    svg.layer(layer)
    if clip:
        svg.open_group(clip_path=clip)
    if tier:
        svg.open_group(filter=svg.blur(tier))
    if streak is None:
        d = "".join(f"M{x:.2f},{y:.2f}h0" for x, y in zip(xs, ys))
    else:
        d = "".join(f"M{x:.2f},{y:.2f}l{a:.2f},{b:.2f}" for x, y, a, b in zip(xs, ys, *streak))
    svg.path(d, fill="none", stroke=INK, stroke_width=sizes_mm, stroke_linecap="round", opacity=op)
    if tier:
        svg.close_group()
    if clip:
        svg.close_group()


def wind_plan(svg, L, dtm, dsm, hourly, rng, x0, y0, args, tiers, north=0.0):
    """Plan of the bank around the cut: the year's wind at 2 m above the
    ground. Each dot is an hour's air, kept in proportion to its local speed
    (so sheltered ground is sparse), with a short streak downwind as long as
    that local speed. Returns the plan's width in mm."""
    r0, r1, c0, c1 = args.wind_window
    st = args.wind_cell
    s_ = 1000.0 / args.wind_scale
    W, H = (c1 - c0) * s_, (r1 - r0) * s_
    rows, cols = np.mgrid[r0:r1:st, c0:c1:st].astype(float)
    g = dtm[r0:r1:st, c0:c1:st]
    can = np.clip(dsm[r0:r1:st, c0:c1:st] - g, 0, None)
    fields, hsec = wind.sector_fields(dsm, g, can, rows, cols, g + 2.0, hourly)
    f_ref = float(wind.profile_factor(2.0, 0.0))
    spd = np.array([h[2] for h in hourly])
    deg = np.array([h[1] for h in hourly])
    n = int(W * H * args.plan_density)
    py = rng.uniform(0, rows.shape[0], n)
    px = rng.uniform(0, rows.shape[1], n)
    hrs = rng.choice(len(hourly), n, p=spd / spd.sum())
    f = np.array([fields[hsec[h]][int(a), int(b)] for h, a, b in zip(hrs, py, px)])
    acc = rng.random(n) < np.clip(f / f_ref, 0, 1)
    loc = spd[hrs] * f / f_ref
    toward = np.radians(deg[hrs] + 180)
    ln = loc * args.plan_streak
    sx, sy = np.sin(toward) * ln, -np.cos(toward) * ln
    X = x0 + px * st * s_
    Y = y0 + py * st * s_
    clip = svg.clip_rect(x0, y0, W, H)
    for (lo, hi, blur, size, op), key in zip(tiers, ("plan_calm", "plan_mid", "plan_strong")):
        sel = acc & (loc >= lo) & (loc < hi)
        dots(svg, L[key], X[sel], Y[sel], args.wind_dot * 0.8 * size, blur, op, clip=clip,
             streak=(sx[sel], sy[sel]))
    # River, canopy edge, the cut.
    svg.layer(L["plan_lines"])
    svg.open_group(clip_path=clip)
    water = ndimage.binary_opening(g < args.plan_water, iterations=2)
    lab, nlab = ndimage.label(water)
    if nlab:
        water = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    for ln_ in terrain.mask_lines(water):
        svg.path(points_to_path(x0 + ln_[:, 0] * st * s_, y0 + ln_[:, 1] * st * s_), fill="none", stroke=INK,
                 stroke_width=0.12)
    for ln_ in terrain.smooth_contours(can, [3.0], sigma=1.0)[3.0]:
        if len(ln_) > 6:
            svg.path(points_to_path(x0 + ln_[:, 0] * st * s_, y0 + ln_[:, 1] * st * s_), fill="none", stroke=INK,
                     stroke_width=0.05, opacity=0.6)
    yc = y0 + (args.row - r0) * s_
    svg.line(x0 + (args.cols[0] - c0) * s_, yc, x0 + (args.cols[1] - c0) * s_, yc, stroke=INK, stroke_width=0.3,
             stroke_dasharray="2 0.6 0.4 0.6")
    svg.close_group()
    svg.rect(x0, y0, W, H, fill="none", stroke=INK, stroke_width=0.08, opacity=0.6)
    svg.layer(L["text"])
    svg.text(x0 + (args.cols[0] - c0) * s_ - 1.5, yc + 0.8, "B", size=2.4, fill=INK, anchor="end")
    svg.text(x0 + (args.cols[1] - c0) * s_ + 1.5, yc + 0.8, "B′", size=2.4, fill=INK)
    svg.text(x0, y0 - 2.5, f"WIND AT 2 M · 1:{args.wind_scale:,}", size=2.0, fill=INK, spacing=0.25)
    svg.layer(L["scale"])
    scale_bar(svg, x0, y0 + H + 3, 100 * s_, 100, "m", color=INK)
    north_arrow(svg, x0 + 100 * s_ + 12, y0 + H + 3.5, r=2.5, color=INK, angle=north)
    return W


def wind_rose(svg, L, hourly, cx, cy, R, rng, tiers, north=0.0):
    """Every hour as a dot in a petal pointing to where its wind came from.
    A petal's length is its share of the hours (R mm = the busiest sector);
    within a petal the hours run from the calmest at the centre to the
    strongest at the tip, crisp to blurred as in the sky."""
    deg = np.array([h[1] for h in hourly])
    spd = np.array([h[2] for h in hourly])
    sec = wind.sector_of(deg).astype(int)
    counts = np.bincount(sec, minlength=wind.SECTORS)
    xs, ys, ss = [], [], []
    for k in range(wind.SECTORS):
        idx = np.nonzero(sec == k)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(spd[idx])]
        length = counts[k] / counts.max() * R
        r = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx) * length
        half = math.radians(360 / wind.SECTORS / 2) * 0.8
        a = math.radians(k * 360 / wind.SECTORS + north) + rng.uniform(-half, half, len(idx)) * np.clip(r / 6, 0.2, 1)
        xs.append(cx + np.sin(a) * r)
        ys.append(cy - np.cos(a) * r)
        ss.append(spd[idx])
    X, Y, S = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)
    svg.layer(L["rose_lines"])
    for pct in (5, 10, 15):
        rr = pct / 100 / (counts.max() / counts.sum()) * R
        if rr <= R * 1.05:
            svg.circle(cx, cy, rr, fill="none", stroke=INK, stroke_width=0.04, opacity=0.5, stroke_dasharray="0.3 0.6")
    for ang in range(0, 360, 90):
        t = math.radians(ang + north)
        svg.line(cx, cy, cx + math.sin(t) * R * 1.05, cy - math.cos(t) * R * 1.05, stroke=INK, stroke_width=0.04,
                 opacity=0.5)
    for (lo, hi, blur, size, op), key in zip(tiers, ("rose_calm", "rose_mid", "rose_strong")):
        sel = (S >= lo) & (S < hi)
        dots(svg, L[key], X[sel], Y[sel], 0.12 * size, blur, op)
    svg.layer(L["text"])
    for pct in (5, 10, 15):
        rr = pct / 100 / (counts.max() / counts.sum()) * R
        if rr <= R * 1.05:
            svg.text(cx + 0.8, cy - rr - 0.6, f"{pct}%", size=1.4, fill=INK, opacity=0.6)
    for lab_, ang in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        t = math.radians(ang + north)
        svg.text(cx + math.sin(t) * R * 1.14, cy - math.cos(t) * R * 1.14 + 0.8, lab_, size=2.0, fill=INK, anchor="middle")
    svg.text(cx, cy + R + 8, f"WIND ROSE · {len(hourly):,} HOURS · % OF HOURS", size=1.6, fill=INK, anchor="middle",
             spacing=0.2, opacity=0.8)


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
    annot = svg.text if args.annotate else (lambda *a, **k: None)      # titles and notes: only with --annotate
    L = dict(frame="FRAME · 0.15 mm", grid="DATUM GRID, DOTTED · 0.03–0.05 mm",
             geo_clay="GEOLOGY · LAKE CLAY, VARVES · 0.08 mm", geo_till="GEOLOGY · TILL, CLASTS · 0.08 mm",
             geo_shale="GEOLOGY · SHALE, PARTINGS · 0.08 mm", geo_carbonate="GEOLOGY · CARBONATE, BEDS, JOINTS, VUGS · 0.08 mm",
             wind="SKY · CALMER HOURS, CRISP · 0.12 mm", wind_mid="SKY · WINDIER HOURS, SOFT BLUR",
             wind_strong="SKY · STRONGEST HOURS, BLURRED",
             plan_lines="WIND PLAN · RIVER, CANOPY EDGE, CUT · 0.05–0.3 mm", plan_calm="WIND PLAN · CALMER, CRISP · 0.1 mm",
             plan_mid="WIND PLAN · WINDIER, SOFT BLUR", plan_strong="WIND PLAN · STRONGEST, BLURRED",
             rose_lines="WIND ROSE · RINGS · 0.04 mm", rose_calm="WIND ROSE · CALMER HOURS, CRISP · 0.12 mm",
             rose_mid="WIND ROSE · WINDIER HOURS, SOFT BLUR", rose_strong="WIND ROSE · STRONGEST HOURS, BLURRED",
             water="WATER LEVELS · 0.05 mm", flood="1997 CREST (EST.), DASHED · 0.12 mm",
             canopy="CANOPY SURFACE (LIDAR), DOTTED · 0.06 mm", tree_wood="TREES · TRUNKS AND LIMBS, FILLED",
             tree_twigs="TREES · TWIGS · 0.04–0.06 mm", tree_leaves="TREES · LEAVES, STIPPLE · 0.15–0.22 mm",
             roots="ROOTS (DIAGRAM) · 0.05–0.2 mm", ground="GROUND (LIDAR) · 0.3 mm",
             key_lines="KEY PLAN · RIVER, VALLEY EDGE, DAM · 0.06–0.35 mm", key_cut="KEY PLAN · CUT LINE · 0.3 mm",
             scale="SCALE BAR", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["frame"])
    if args.annotate:
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
    all_speeds = np.array([g[2] for g in gusts])
    cuts = np.percentile(all_speeds, args.blur_tiers)

    # The air at the bank, from the ground up to the sky band: the same hours,
    # slowed by the trees and the bank upwind of each point along each hour's
    # direction (drawing/wind.py, a screening model). A point is kept in
    # proportion to the hour's local speed, so sheltered air thins out.
    if hourly:
        sky_density = keep.sum() / (width * (sky_hi - sky_lo))
        hz = np.arange(float(np.nanmin(ground)), sky_lo + 0.25, 0.5)
        X_, Z_ = np.meshgrid(np.arange(width) + 0.5, hz)
        g_at = np.interp(X_, ch + 0.5, ground)
        can = np.nan_to_num(height)
        c_at = np.interp(X_, ch + 0.5, can)
        fields, hsec = wind.sector_fields(dsm, g_at, c_at, np.full_like(X_, args.row, dtype=float),
                                          args.cols[0] + X_, Z_, gusts)
        f_ref = float(wind.profile_factor(sky_lo - float(np.nanmin(ground)), 0.0))
        spd_h = all_speeds
        n_cand = int(sky_density * width * (sky_lo - float(np.nanmin(ground))) * args.near_density)
        x_c = rng.uniform(0, width, n_cand)
        z_c = rng.uniform(float(np.nanmin(ground)), sky_lo, n_cand)
        above_ground = z_c > np.interp(x_c, ch + 0.5, ground)
        hrs = rng.choice(len(gusts), n_cand, p=spd_h / spd_h.sum())
        ix = np.clip(x_c.astype(int), 0, width - 1)
        iz = np.clip(np.round((z_c - hz[0]) / 0.5).astype(int), 0, len(hz) - 1)
        f_c = np.array([fields[hsec[h]][a, b] for h, a, b in zip(hrs, iz, ix)])
        acc = above_ground & (rng.random(n_cand) < np.clip(f_c / f_ref, 0, 1))
        xs_pts = np.concatenate([xs_pts, x_c[acc]])
        zs_pts = np.concatenate([zs_pts, z_c[acc]])
        sp_pts = np.concatenate([sp_pts, spd_h[hrs[acc]] * f_c[acc] / f_ref])
        keep = np.concatenate([keep, np.ones(acc.sum(), bool)])
    clip = svg.clip_rect(sec.X(0), sec.Y(args.z_top), width * sec.sh, (args.z_top - args.z_bottom) * sec.sv)
    # Tiers by wind speed: calm hours crisp, the strongest hours blurred into haze.
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

    # Trees by species, each grown to its habit and fitted to its lidar height
    # and crown width; species inferred from how often its ground floods.
    tops = crown_tops(height, args.min_tree, args.tree_spacing)
    levels = np.array([z for _, z in open_days])
    tree_rng = np.random.default_rng(args.seed + 7)
    placed = []
    for i in tops:
        lo_, hi_ = i, i
        while lo_ > 0 and np.nan_to_num(height[lo_ - 1]) >= 0.55 * height[i]:
            lo_ -= 1
        while hi_ < width - 1 and np.nan_to_num(height[hi_ + 1]) >= 0.55 * height[i]:
            hi_ += 1
        crown_m = max(4.0, min(hi_ - lo_ + 1.0, height[i] * 0.9))
        share = float((levels >= ground[i]).mean())
        sp = trees.zone_species(share, ground[i] > z97, tree_rng)
        placed.append((i, sp, share, crown_m))
        trees.draw(svg, dict(wood=L["tree_wood"], twigs=L["tree_twigs"], leaves=L["tree_leaves"]), sp,
                   sec.X(i + 0.5), sec.Y(ground[i]), height[i] * sec.sv, crown_m * sec.sh, tree_rng, ink=INK)
    # The lidar's canopy surface, as measured: a fine dotted line over the drawn trees.
    svg.layer(L["canopy"])
    svg.path(sec.path(ch, np.where(np.isfinite(height), top, np.nan)), fill="none", stroke=INK, stroke_width=0.06,
             stroke_dasharray="0.15 0.6", stroke_linecap="round", opacity=0.7)

    # Roots: diagram, depth and spread from each tree's height and species.
    svg.layer(L["roots"])
    for i, sp, _, _ in placed:
        dmul, smul = trees.SPECIES[sp]["root"]
        for x0_, z0_, x1_, z1_, lvl in roots(rng, i + 0.5, ground[i], min(3.5, height[i] * 0.15 * dmul),
                                             height[i] * 0.45 * smul):
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
    # Species names stand upright above each crown, clear of their neighbours.
    for i, sp, share, _ in placed:
        svg.text(sec.X(i + 0.5) + 0.5, sec.Y(top[i] + 1.5), f"{trees.SPECIES[sp]['name']} · {height[i]:.0f} M",
                 size=1.5, fill=INK, spacing=0.15, opacity=0.85, rotate=-90)
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
    annot(tx, ty, "RED RIVER AT LOCKPORT — RIVERBANK SECTION B–B′", size=5.2, fill=INK, spacing=0.4)
    annot(tx, ty + 7, f"EAST BANK, ABOUT 300 M BELOW ST. ANDREWS LOCK AND DAM · 1:{args.scale:,} (TRUE SCALE)",
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
        annot(tx, ty + 12 + 2.9 * j, note, size=1.75, fill=INK, opacity=0.85 if j < 5 else 0.65)

    svg.layer(L["scale"])
    scale_bar(svg, sec.x0, Hp - m - 9, 20 * sec.sh, 20, "m", color=INK)

    if hourly:
        tiers_ = [(0, cuts[0], 0.0, 1.0, 0.75), (cuts[0], cuts[1], args.blur[0], 2.4, 0.9),
                  (cuts[1], np.inf, args.blur[1], 4.5, 1.0)]
        px0, py0 = sec.x0, sec.Y(args.z_bottom) + 24
        north_ = true_north_angle(key_transform, key_crs, args.row, sum(args.cols) / 2)
        pw_ = wind_plan(svg, L, dtm, dsm, gusts, rng, px0, py0, args, tiers_, north=north_)
        wind_rose(svg, L, gusts, px0 + pw_ + 18 + args.rose_r, py0 + args.rose_r + 4, args.rose_r, rng, tiers_,
                  north=north_)

    # Key plan: where the section is cut. The other section's line is dashed.
    kw = args.key_window
    kx, ky = Wp - m - 122, Hp - m - 125
    key_plan(svg, dtm, kx, ky, kw, args.row, args.cols, args.dam_row, args.dam_cols, INK,
             scale=args.key_scale, angle=true_north_angle(key_transform, key_crs, (kw[0] + kw[1]) / 2, (kw[2] + kw[3]) / 2),
             other_cuts=[(1840, (1150, 1850))], layers=dict(lines=L["key_lines"], cut=L["key_cut"], text=L["text"]),
             label=("B", "B′"), names=[("DAM", 2105, 1700, 0), ("LOCK", 2240, 1200, -90), ("FLOODWAY", 2700, 2470, -90), ("RED RIVER", 2700, 1250, -60)])
    if hourly:
        ks = 1000.0 / args.key_scale
        wr0, wr1, wc0, wc1 = args.wind_window
        svg.layer(L["key_lines"])
        svg.rect(kx + (wc0 - kw[2]) * ks, ky + (wr0 - kw[0]) * ks, (wc1 - wc0) * ks, (wr1 - wr0) * ks, fill="none",
                 stroke=INK, stroke_width=0.08, stroke_dasharray="0.5 0.5")
        svg.layer(L["text"])
        svg.text(kx + (wc1 - kw[2]) * ks + 1, ky + (wr0 - kw[0]) * ks + 1.6, "WIND PLAN", size=1.3, fill=INK, opacity=0.8)
    # The cut's letters at both ends of the section.
    svg.layer(L["text"])
    for c, letter, anchor in ((0, ("B", "B′")[0], "end"), (len(ch), ("B", "B′")[1], "start")):
        svg.text(sec.X(c) + (-6 if anchor == "end" else 6), sec.Y(args.z_top) - 4, letter, size=4.5,
                 fill=INK, anchor=anchor)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print("trees:", [(i, sp, round(sh, 3)) for i, sp, sh, _ in placed])
    print(f"wrote {args.out}  ({len(gusts)} {'hours' if hourly else 'gust days'}, {len(tops)} trees, "
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
    ap.add_argument("--near-density", type=float, default=1.6,
                    help="density of the bank's air points relative to the sky band")
    ap.add_argument("--wind-window", type=int, nargs=4, default=[1640, 2040, 1260, 2060],
                    metavar=("ROW0", "ROW1", "COL0", "COL1"), help="DEM area of the wind plan")
    ap.add_argument("--wind-cell", type=int, default=2, help="m per cell of the wind plan model")
    ap.add_argument("--wind-scale", type=int, default=2500)
    ap.add_argument("--plan-density", type=float, default=0.9, help="candidate wind-plan dots per mm²")
    ap.add_argument("--plan-streak", type=float, default=0.045, help="mm of streak per km/h in the wind plan")
    ap.add_argument("--plan-water", type=float, default=218.4, help="lidar ground below this is river in the wind plan")
    ap.add_argument("--rose-r", type=float, default=34.0, help="wind rose radius (40 km/h), mm")
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
    ap.add_argument("--annotate", action="store_true",
                    help="also print the title, notes, legend, sources and page border; the reading guide covers them")
    ap.add_argument("--out", default="outputs/sheets/04b_riverbank_section.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
