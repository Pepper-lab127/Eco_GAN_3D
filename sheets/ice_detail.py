"""Sheet 04c - Under the ice: a close-up section at true scale.

The deepest water on section A-A', 2007-08, at 1:5 with no exaggeration,
rendered as a composite after the sea-ice section in Architecture of Nature /
Nature of Architecture. Organic rendering (domain-warped noise, filament webs
traced through a curl-noise field, light curtains), with layers set by data:

  - snow on the ice: settled depth from the winter's precipitation
  - ice: thickness grown day by day from Winnipeg's cold (Stefan's law); each
    day a growth band. Ice grown before the turn of the year is drawn as
    cloudy white ice (frazil and snow-ice, tangled), later ice as clear
    columnar ice (vertical crystal boundaries). Bubble trains per band scale
    with that day's growth rate (fast freezing traps more air).
  - water: depth from the chart soundings at the deepest point of the cut;
    the current's streaks set by winter flow / channel area (05OJ010)
  - bed: lake clay

    python -m sheets.ice_detail
"""

import argparse
import base64
import datetime as dt
import io
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from drawing import climate, hydrometric, terrain
from drawing.svg import PAPER, Svg, scale_bar

INK = "#1a1d1f"


def noise(shape, rng, sigma, aniso=(1.0, 1.0)):
    n = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                (sigma * aniso[0], sigma * aniso[1]), mode="wrap")
    return (n - n.mean()) / (n.std() + 1e-6)


def fbm(shape, rng, base, octaves=5, aniso=(1.0, 1.0), small=None):
    """Fractal noise ~N(0,1). Computed at a reduced size `small` then upsampled."""
    sh = small or shape
    f = sum((0.55 ** k) * noise(sh, rng, max(base / (2 ** k), 0.6), aniso) for k in range(octaves))
    f = (f - f.mean()) / (f.std() + 1e-6)
    if small:
        f = np.asarray(Image.fromarray(f.astype(np.float32), mode="F").resize(shape[::-1], Image.BICUBIC))
    return f


def warp(field, dx, dy):
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return ndimage.map_coordinates(field, [yy + dy, xx + dx], order=1, mode="reflect")


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def build(args):
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- data
    dtm, _, _ = terrain.load_dem(args.dem)
    bed_dem, _, _ = terrain.load_dem(args.bed)
    ground = dtm[args.row, args.cols[0]:args.cols[1]]
    bed_row = bed_dem[args.row, args.cols[0]:args.cols[1]]
    dl, lv, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
    dq, qv, _ = hydrometric.load_wsc_daily(args.level_csv, "discharge")
    level = {d: v for d, v in zip(dl, lv) if v == v}
    flow = {d: v for d, v in zip(dq, qv) if v == v}
    days, station = climate.load_eccc_daily(args.climate)
    y0 = args.winter
    afdd = climate.freezing_degree_days(days, dt.date(y0, 10, 1), dt.date(y0 + 1, 4, 30))
    wdays = sorted(d for d in level if dt.date(y0, 12, 1) <= d <= dt.date(y0 + 1, 3, 10))
    ice = {d: float(climate.stefan_ice(afdd[d], args.alpha)) for d in afdd}
    first_ice = next(d for d in sorted(afdd) if ice[d] > 0.01 and d.month in (10, 11, 12))
    last = wdays[-1]
    ice_t = ice[last]
    grow_days = [d for d in sorted(afdd) if first_ice <= d <= last]
    precip = sum(v["precip"] or 0 for d, v in days.items() if dt.date(y0, 11, 1) <= d <= last)
    snow_t = min(0.45, precip / 1000 / args.snow_density)        # settled snow depth, m
    wl = level[last]
    deepest = int(np.nanargmin(bed_row))
    water_d = wl - ice_t - float(bed_row[deepest])
    lidar_water = float(np.nanmedian(ground[np.abs(ground - np.nanmin(ground)) < 0.05]))
    wet = np.isfinite(bed_row) & (ground <= lidar_water + 0.15)
    area = float(np.nansum(np.clip(wl - ice_t - bed_row[wet], 0, None)))
    q_last = flow.get(last, np.nan)
    vel = q_last / area if area > 0 and q_last == q_last else 0.1
    clear_from = dt.date(y0 + 1, 1, 1)

    # ---------------------------------------------------------------- layout
    Wp, Hp = PAPER[args.paper]
    s = 1000.0 / args.scale                       # mm per m
    R = args.res
    img_x, img_y = 40.0, 70.0
    img_w = Wp - 2 * img_x
    air = 0.22
    bed_show = 0.38
    total_m = air + snow_t + ice_t + water_d + bed_show
    img_h = total_m * s
    W, H = int(img_w * R), int(img_h * R)
    ppm = s * R                                   # px per metre
    y_snow = air * ppm
    y_ice = (air + snow_t) * ppm
    y_water = (air + snow_t + ice_t) * ppm
    y_bed = (air + snow_t + ice_t + water_d) * ppm
    yy = np.arange(H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    xx = np.ones((H, 1), np.float32) * np.arange(W, dtype=np.float32)[None, :]
    small = (H // 4, W // 4)

    # Undulating interfaces (warped), so no boundary is a ruled line.
    def edge(y, amp_px, wl_px, k):
        n = ndimage.gaussian_filter1d(np.random.default_rng(args.seed + k).standard_normal(W).astype(np.float32),
                                      wl_px, mode="wrap")
        return y + amp_px * n / (n.std() + 1e-6)

    e_snow = edge(y_snow, ppm * 0.04, ppm * 0.25, 1)
    e_ice = edge(y_ice, ppm * 0.015, ppm * 0.4, 2)
    e_water = edge(y_water, ppm * 0.02, ppm * 0.35, 3)
    e_bed = edge(y_bed, ppm * 0.05, ppm * 0.5, 4)

    # Domain warp shared by all materials: everything flows a little.
    wx = fbm((H, W), rng, ppm * 0.25, 4, small=small) * ppm * 0.03
    wy = fbm((H, W), rng, ppm * 0.25, 4, small=small) * ppm * 0.02

    rgb = np.zeros((H, W, 3), np.float32)

    def paint(color, alpha):
        col = np.asarray(color, np.float32)
        a = np.clip(alpha, 0, 1)[..., None]
        rgb[:] = rgb + (col - rgb) * a

    # Paper-white air above.
    rgb[:] = np.array(args.paper_tone, np.float32)

    # ---------------------------------------------------------------- water
    m_water = (yy >= e_water[None, :]) & (yy < e_bed[None, :])
    depth01 = np.clip((yy - y_water) / max(y_bed - y_water, 1), 0, 1)
    top_c, mid_c, deep_c = map(np.array, (args.water_top, args.water_mid, args.water_deep))
    wcol = np.where(depth01[..., None] < 0.45,
                    top_c + (mid_c - top_c) * (depth01[..., None] / 0.45),
                    mid_c + (deep_c - mid_c) * ((depth01[..., None] - 0.45) / 0.55))
    mott = warp(fbm((H, W), rng, ppm * 0.18, 5, aniso=(0.6, 1.4), small=small), wx, wy)
    wcol = wcol * (1 + 0.08 * mott[..., None])
    paint(wcol, m_water.astype(np.float32))

    # Light curtains hanging from the ice: long vertical streaks, fading down,
    # stronger under clear ice (later winter = lower in the ice).
    cur = fbm((H, W), rng, ppm * 0.05, 4, aniso=(14.0, 1.0), small=small)
    cur = warp(cur, wx * 2, wy)
    hang = np.exp(-np.clip(yy - e_water[None, :], 0, None) / (ppm * args.curtain_len))
    paint(args.light, np.clip((cur - 0.2) * 0.55, 0, 1) * hang * m_water * 0.85)

    # The current: suspended grains drawn as short trails, their length the
    # distance carried in one 1/30 s exposure at the winter velocity.
    tr = Image.new("L", (W, H), 0)
    dt_ = ImageDraw.Draw(tr)
    trail = max(1.5, vel * ppm / 30 * 4)
    for _ in range(int(args.grains * W / 1000)):
        x = rng.uniform(0, W)
        y = rng.uniform(y_water, y_bed)
        dep_ = (y - y_water) / max(y_bed - y_water, 1)
        ln = trail * rng.uniform(0.4, 1.6) * (0.6 + dep_)
        dt_.line([(x, y), (x + ln, y + rng.normal(0, ln * 0.08))], fill=int(rng.integers(30, 150)), width=1)
    paint(args.light, np.asarray(tr.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0 * 0.6 * m_water)

    # Churned water and sediment toward the bed, after the reference's swell.
    near_bed = smoothstep(0.4, 1.0, depth01)
    swell = np.zeros((H, W), np.float32)
    for k, (lam_, amp, sh) in enumerate(((0.09, 4.0, 0.12), (0.035, 2.5, 0.05), (0.015, 1.5, 0.02))):
        n_ = fbm((H, W), rng, ppm * sh, 4, small=small)
        ph = yy / (ppm * lam_) + amp * warp(n_, wx * 3, wy * 3) + xx / (ppm * (0.7 + 0.6 * k))
        swell += (0.5 + 0.5 * np.sin(ph)) ** (8 - 2 * k) * (0.5 / (k + 1))
    paint(args.sediment, np.clip(swell, 0, 1) * near_bed * m_water * 0.7)
    spk = fbm((H, W), rng, 1.2, 2)
    paint(args.light, np.clip((spk - 2.3) * 0.8, 0, 1) * (0.3 + near_bed) * m_water)

    # ---------------------------------------------------------------- bed
    m_bed = yy >= e_bed[None, :]
    lam = np.sin((yy - e_bed[None, :]) / (ppm * 0.012) + 2.5 * warp(fbm((H, W), rng, ppm * 0.2, 3, small=small), wx, wy))
    bcol = np.array(args.clay) * (1 + 0.1 * lam[..., None] + 0.05 * mott[..., None])
    grain_b = fbm((H, W), rng, 1.5, 3)
    bcol = bcol * (1 + 0.07 * grain_b[..., None])
    bcol = bcol * (1 - 0.5 * smoothstep(0, ppm * bed_show, yy - e_bed[None, :]))[..., None]
    paint(bcol, m_bed.astype(np.float32))

    # ---------------------------------------------------------------- ice
    m_ice = (yy >= e_ice[None, :]) & (yy < e_water[None, :])
    # Depth of each day's underside below the ice top, from the growth record.
    y_of_day = {d: e_ice.mean() + ice[d] / ice_t * (y_water - y_ice) for d in grow_days}
    clear_y = y_of_day[min(grow_days, key=lambda d: abs((d - clear_from).days))]
    e_clear = edge(clear_y, ppm * 0.03, ppm * 0.18, 5)
    in_clear = smoothstep(e_clear[None, :] - ppm * 0.04, e_clear[None, :] + ppm * 0.04, yy)
    base = fbm((H, W), rng, ppm * 0.1, 5, small=small)
    clouds = warp(base, wx * 2.5, wy * 2.5)
    white_ice = np.array(args.ice_white) * (1 + 0.06 * clouds[..., None])
    blue_ice = np.array(args.ice_blue) + (np.array(args.ice_deep) - np.array(args.ice_blue)) * \
        smoothstep(-1.2, 1.5, clouds)[..., None]
    icol = white_ice + (blue_ice - white_ice) * in_clear[..., None]
    # Growth bands: one per day, warped, brighter where growth was fast.
    bands = np.zeros((H, W), np.float32)
    bw = np.zeros(H, np.float32)
    prev = None
    for d in grow_days:
        y = int(y_of_day[d])
        g = ice[d] - (ice[prev] if prev else 0)
        prev = d
        if 0 <= y < H:
            bw[y] += min(1.0, g / 0.01)
    bands = ndimage.gaussian_filter1d(bw, 1.2)[:, None] * np.ones((1, W), np.float32)
    bands = warp(bands, wx * 0.5, wy * 3)
    icol = icol * (1 + 0.12 * bands[..., None])
    paint(icol, m_ice.astype(np.float32))
    # Glow of the ice into the water just below.
    glow = np.exp(-np.clip(yy - e_water[None, :], 0, None) / (ppm * 0.08)) * m_water
    paint(args.ice_blue, glow * 0.35)

    # ---------------------------------------------------------------- snow
    m_snow = (yy >= e_snow[None, :]) & (yy < e_ice[None, :])
    sn = warp(fbm((H, W), rng, ppm * 0.03, 4, small=small), wx, wy)
    paint(np.array(args.snow) * (1 - 0.04 * sn[..., None]), m_snow.astype(np.float32))

    # ---------------------------------------------------------------- filaments
    # Traced through a curl-noise field: tangled in the snow and white ice,
    # vertical (crystal boundaries, bubble trains) in the clear ice, and
    # hanging into the water from the ice underside.
    pot = fbm(small, rng, small[0] * 0.08, 4) + 0.6 * fbm(small, rng, small[0] * 0.015, 3)
    gy_, gx_ = np.gradient(pot)
    cu, cv_ = gy_, -gx_                               # curl of the potential
    sc = H / small[0]

    def vel_at(x, y):
        i = min(max(int(y / sc), 0), small[0] - 1)
        j = min(max(int(x / sc), 0), small[1] - 1)
        return cu[i, j], cv_[i, j]

    web = Image.new("L", (W, H), 0)
    dw = ImageDraw.Draw(web)

    def trace(x, y, steps, step, vertical, amp, bounds, fork=0.0, depth=0):
        pts = [(x, y)]
        head = rng.uniform(0, 2 * math.pi)
        for _ in range(steps):
            u, v = vel_at(x, y)
            ang = math.atan2(v, u)
            # Follow the field loosely: the heading wanders.
            head += 0.35 * math.sin(ang - head) + rng.normal(0, 0.35)
            if vertical:
                dx_, dy_ = 0.45 * math.cos(head), 1.0
            else:
                dx_, dy_ = math.cos(head), 0.75 * math.sin(head)
            x, y = x + dx_ * step, y + dy_ * step
            if not (0 <= x < W):
                break
            lo, hi = bounds(x) if callable(bounds) else bounds
            if not (lo <= y <= hi):
                break
            pts.append((x, y))
            if fork and depth < 3 and rng.random() < fork:
                trace(x, y, steps // 2, step, vertical, int(amp * 0.7), bounds, fork, depth + 1)
        if len(pts) > 2:
            dw.line(pts, fill=amp, width=1 if rng.random() < 0.85 else 2)

    # Tangled web: snow and white ice.
    for _ in range(int(args.web * W / 1000)):
        x = rng.uniform(0, W)
        y = rng.uniform(e_snow.min() - ppm * 0.03, e_clear[int(x)])
        trace(x, y, int(rng.integers(6, 60)), R * rng.uniform(0.4, 1.2), False, int(rng.integers(30, 150)),
              lambda x: (e_snow[int(x)] - ppm * 0.05, e_clear[int(x)] + ppm * 0.04 * rng.random()), fork=0.04)
    # Columnar boundaries and bubble trains: clear ice, denser in bands of fast growth.
    gr = np.array([max(0.0, ice[d] - ice[grow_days[i - 1]]) if i else 0.0 for i, d in enumerate(grow_days)])
    for i, d in enumerate(grow_days):
        y = y_of_day[d]
        if y < clear_y:
            continue
        n = int(args.columns * gr[i] / max(gr.max(), 1e-6) * W / 1000)
        for _ in range(n):
            x0_ = rng.uniform(0, W)
            if y < e_clear[int(x0_)]:
                continue
            trace(x0_, y, int(rng.integers(8, 40)), R * 0.8, True, int(rng.integers(30, 110)),
                  lambda x: (e_clear[int(x)] - 2, e_water[int(x)] + ppm * 0.03 * rng.random()))
    # Hanging threads: from the underside into the water.
    centres = rng.uniform(0, W, int(args.clusters))
    for _ in range(int(args.hang * W / 1000)):
        x = float(np.clip(rng.choice(centres) + rng.normal(0, W * 0.035), 0, W - 1))
        y = e_water[int(x)] + rng.uniform(-2, 4)
        trace(x, y, int(rng.lognormal(3.6, 0.7)), R * rng.uniform(0.6, 1.4), True, int(rng.integers(20, 120)),
              (y - 2, y_water + ppm * args.curtain_len * 1.8), fork=0.03)
    a_web = np.asarray(web.filter(ImageFilter.GaussianBlur(0.45)), np.float32) / 255.0
    halo = np.asarray(web.filter(ImageFilter.GaussianBlur(R * 0.8)), np.float32) / 255.0
    fade = np.where(yy > e_water[None, :], np.exp(-(yy - e_water[None, :]) / (ppm * args.curtain_len)), 1.0)
    paint(args.light, np.clip(halo * 1.8, 0, 1) * 0.35 * fade)
    paint(args.web_tone, np.clip(a_web * 2.2, 0, 1) * fade)

    # Bubbles: small rings in the clear ice and drifting up under it.
    bub = Image.new("L", (W, H), 0)
    db = ImageDraw.Draw(bub)
    for _ in range(int(args.bubbles * W / 1000)):
        x = rng.uniform(0, W)
        y = rng.uniform(clear_y, y_water + ppm * 0.15)
        r = R * rng.uniform(0.2, 1.2)
        db.ellipse([x - r, y - r * 0.8, x + r, y + r * 0.8], outline=int(rng.integers(80, 200)))
    paint(args.light, np.asarray(bub.filter(ImageFilter.GaussianBlur(0.4)), np.float32) / 255.0 * 0.7)

    # ---------------------------------------------------------------- grade
    grain = rng.standard_normal((H, W, 1)).astype(np.float32)
    rgb[:] = rgb + grain * 0.018
    vx = (xx / W - 0.5) * 2
    vy = (yy / H - 0.5) * 2
    rgb *= np.clip(1 - 0.12 * (vx ** 2 + vy ** 2), 0.8, 1)[..., None]
    np.clip(rgb, 0, 1, out=rgb)
    raster = Image.fromarray((rgb * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    raster.save(os.path.splitext(args.out)[0] + "_render.jpg", quality=93)

    # ---------------------------------------------------------------- sheet
    svg = Svg(Wp, Hp, background=args.page)
    L = dict(render="RENDER (RASTER COMPOSITE)", ticks="DATA · DAY TICKS, FIRST OF EACH MONTH · 0.1 mm",
             scale="SCALE BAR", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["render"])
    buf = io.BytesIO()
    raster.save(buf, format="JPEG", quality=91)
    svg.image(img_x, img_y, img_w, img_h, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())

    # Month ticks on the ice, at the right edge: where each month's ice begins.
    svg.layer(L["ticks"])
    xr = img_x + img_w
    for d in grow_days:
        if d.day == 1:
            y = img_y + y_of_day[d] / R
            svg.line(xr, y, xr + 4, y, stroke=INK, stroke_width=0.1)
            svg.layer(L["text"])
            svg.text(xr + 5, y + 0.7, f"{d.strftime('%b 1').upper()} · {ice[d]:.2f} M", size=1.8, fill=INK)
            svg.layer(L["ticks"])
    svg.layer(L["text"])
    for y, label in ((y_snow, f"SNOW · {snow_t:.2f} M"), (y_ice, f"ICE · {ice_t:.2f} M BY {last.strftime('%b %d').upper()}"),
                     (y_water, f"WATER · {water_d:.1f} M"), (y_bed, "LAKE CLAY")):
        svg.text(img_x - 3, img_y + y / R + 0.7, label, size=1.8, fill=INK, anchor="end", spacing=0.15)

    svg.text(img_x, 30, "UNDER THE ICE", size=6, fill=INK, spacing=0.6)
    svg.text(img_x, 37, f"RED RIVER BELOW ST. ANDREWS LOCK AND DAM · DEEPEST WATER ON SECTION A–A′ · WINTER {y0}–{str(y0 + 1)[2:]} · "
             f"1:{args.scale} (TRUE SCALE)", size=2.2, fill=INK, spacing=0.3, opacity=0.85)
    cap = [
        f"Section through the river's winter cover at its deepest point, {last.strftime('%B %d, %Y')}. The ice grew "
        f"from {first_ice.strftime('%B %d')} to {ice_t:.2f} m,",
        "a band for each day (Stefan's law, Winnipeg daily temperature). Ice formed before the new year is drawn as cloudy",
        "white ice, tangled with frazil and snow; later ice grows downward in clear columns, its bubble trains denser on the",
        "days it grew fastest. Light hangs from the underside into water "
        f"{water_d:.1f} m deep (riverbed from chart soundings),",
        f"moving at about {vel:.2f} m/s ({q_last:.0f} m³/s under the ice, {hydrometric.station_id(args.level_csv)}). "
        f"Snow: {snow_t:.2f} m settled from {precip:.0f} mm of winter precipitation.",
        "Rendered composite; layer thicknesses, timing and current from data, textures interpretive.",
    ]
    for j, c in enumerate(cap):
        svg.text(img_x, img_y + img_h + 12 + 3.2 * j, c, size=2.1, fill=INK, opacity=0.85 if j < 5 else 0.6)
    svg.layer(L["scale"])
    scale_bar(svg, img_x, Hp - 22, 0.5 * s, 0.5, "m", color=INK)

    svg.save(args.out)
    print(f"wrote {args.out} ({W}x{H} px): snow {snow_t:.2f} m, ice {ice_t:.2f} m from {first_ice}, water {water_d:.2f} m, "
          f"v {vel:.2f} m/s, clear ice from {clear_from}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--climate", default="data/lockport/climate/en_climate_daily_MB_5023222_*.csv")
    ap.add_argument("--winter", type=int, default=2007)
    ap.add_argument("--row", type=int, default=1840)
    ap.add_argument("--cols", type=int, nargs=2, default=[1150, 1850])
    ap.add_argument("--alpha", type=float, default=0.015)
    ap.add_argument("--snow-density", type=float, default=0.25, help="settled snow density as a share of water")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--res", type=float, default=6.0, help="px per mm (6 = 152 dpi)")
    ap.add_argument("--web", type=float, default=2400, help="tangled filaments per 1000 px of width")
    ap.add_argument("--clusters", type=int, default=9, help="clusters of hanging threads")
    ap.add_argument("--grains", type=float, default=2500, help="suspended grains per 1000 px")
    ap.add_argument("--columns", type=float, default=120, help="column traces per day of fastest growth, per 1000 px")
    ap.add_argument("--hang", type=float, default=170, help="hanging threads per 1000 px")
    ap.add_argument("--bubbles", type=float, default=500)
    ap.add_argument("--curtain-len", type=float, default=0.55, help="m the light curtains hang")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD_P")
    ap.add_argument("--out", default="outputs/sheets/04c_under_the_ice.svg")
    ap.set_defaults(page="#f4f3f0", paper_tone=(0.95, 0.955, 0.955), snow=(0.93, 0.945, 0.95),
                    ice_white=(0.86, 0.91, 0.93), ice_blue=(0.55, 0.72, 0.82), ice_deep=(0.3, 0.47, 0.6),
                    water_top=(0.36, 0.52, 0.62), water_mid=(0.16, 0.27, 0.34), water_deep=(0.14, 0.22, 0.23),
                    sediment=(0.62, 0.7, 0.68), clay=(0.27, 0.27, 0.25), light=(0.9, 0.95, 0.97),
                    web_tone=(0.97, 0.98, 0.99))
    build(ap.parse_args())


if __name__ == "__main__":
    main()
