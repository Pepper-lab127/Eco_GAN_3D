"""Sheet 04a (scene) - The river in winter, as a composite section.

The winter section rendered as a scene rather than a line drawing, after the
tonal sections of Architecture of Nature / Nature of Architecture: a raster
composite (sky, receding valley, trees, snow, ice, water, fish, ground) with
the data drawn crisply over it. Every element is placed by data:

  - receding valley: lidar ground and canopy of rows north of the cut, each
    further row hazier (looking north, downstream)
  - trees on the cut and just behind it: lidar crown tops and heights
  - ice: 2007-08, the coldest winter on file, grown day by day from Winnipeg's
    temperature (Stefan's law); its growth bands are the days
  - water level: 05OJ010 below the dam; riverbed from chart soundings
  - falling snow: one flake per share of the winter's precipitation
  - fish: six species, by winter habit (diagram)
  - ground: two drillers' logs

    python -m sheets.winter_scene
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

from drawing import climate, fish, geology, hydrometric, terrain, texture
from drawing.section import Section, water_edges
from drawing.svg import PAPER, Svg, scale_bar
from drawing.keyplan import key_plan
from sheets.calendar_cut import true_north_angle
from sheets.riverbank_section import crown_tops

INK = "#eef1f2"


# ---------------------------------------------------------------- raster kit

class Canvas:
    """Float RGB canvas in sheet mm at `res` px per mm."""

    def __init__(self, w_mm, h_mm, res):
        self.res = res
        self.W, self.H = int(w_mm * res), int(h_mm * res)
        self.rgb = np.zeros((self.H, self.W, 3), np.float32)

    def px(self, mm):
        return np.asarray(mm) * self.res

    def over(self, color, alpha, x0=0, y0=0):
        """Composite colour (rgb tuple or HxWx3 array) through alpha (HxW, 0-1)."""
        h, w = alpha.shape
        x1, y1 = min(x0 + w, self.W), min(y0 + h, self.H)
        xa, ya = max(x0, 0), max(y0, 0)
        if x1 <= xa or y1 <= ya:
            return
        a = alpha[ya - y0:y1 - y0, xa - x0:x1 - x0, None]
        col = np.asarray(color, np.float32)
        if col.ndim == 3:
            col = col[ya - y0:y1 - y0, xa - x0:x1 - x0]
        region = self.rgb[ya:y1, xa:x1]
        region += (col - region) * a

    def mask(self):
        return Image.new("L", (self.W, self.H), 0)


def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(r)) if r > 0 else img


def arr(img):
    return np.asarray(img, np.float32) / 255.0


def fbm(shape, rng, scales=(64, 24, 8, 3), weights=(0.5, 0.25, 0.15, 0.1), aniso=(1, 1)):
    """Smooth fractal noise in 0..1, stretched by aniso (y, x)."""
    out = np.zeros(shape, np.float32)
    for s, w in zip(scales, weights):
        n = rng.standard_normal(shape).astype(np.float32)
        out += w * ndimage.gaussian_filter(n, (s * aniso[0], s * aniso[1]))
    out -= out.min()
    return out / max(out.max(), 1e-6)


def mix(a, b, t):
    return tuple(float(x + (y - x) * t) for x, y in zip(a, b))


# ---------------------------------------------------------------- the scene

def build(args):
    rng = np.random.default_rng(args.seed)
    dtm, transform, crs = terrain.load_dem(args.dem)
    dsm, _, _ = terrain.load_dem(args.dsm)
    c0, c1 = args.cols
    row = args.row
    ch = np.arange(c1 - c0, dtype=float)
    ground = dtm[row, c0:c1].astype(float)

    Wp, Hp = PAPER[args.paper]
    sec = Section(0.0, 0.0, args.h_scale, args.v_scale, args.z_top)   # full bleed
    cv = Canvas(Wp, Hp, args.res)
    R = cv.res

    def X(c):
        return np.asarray(sec.X(c)) * R

    def Y(z):
        return np.asarray(sec.Y(z)) * R

    xs_px = np.arange(cv.W)
    ch_at = xs_px / R / sec.sh                       # chainage of each pixel column
    col_idx = np.clip(np.round(ch_at).astype(int), 0, len(ch) - 1)

    def at_px(values):
        """A per-metre profile sampled smoothly at every pixel column."""
        v = np.asarray(values, float)
        ok = np.isfinite(v)
        out = np.interp(ch_at, ch[ok], v[ok]) if ok.any() else np.full(len(ch_at), np.nan)
        return np.where(np.isfinite(v[col_idx]), out, np.nan)

    # Water, ice and bed on the cut.
    lidar_water = float(np.nanmedian(ground[np.abs(ground - np.nanmin(ground)) < 0.05]))
    in_channel = ground <= lidar_water + 0.15
    mid = int(np.median(np.nonzero(in_channel)[0]))
    for_edges = np.where(in_channel, -np.inf, ground)
    bed = np.full(len(ch), np.nan)
    if os.path.exists(args.bed):
        bed = terrain.load_dem(args.bed)[0][row, c0:c1]
    bed = np.where(in_channel, bed, np.nan)
    bed = np.where(np.isfinite(bed), bed, np.where(in_channel, lidar_water - 2.0, np.nan))

    dl, lv, _ = hydrometric.load_wsc_daily(args.level_csv, "level")
    level = {d: v for d, v in zip(dl, lv) if v == v}
    days, station = climate.load_eccc_daily(args.climate)
    y0w = args.winter
    afdd = climate.freezing_degree_days(days, dt.date(y0w, 10, 1), dt.date(y0w + 1, 4, 30))
    wdays = sorted(d for d in level if dt.date(y0w, 12, 1) <= d <= dt.date(y0w + 1, 3, 10))
    ice = {d: float(climate.stefan_ice(afdd[d], args.alpha)) for d in wdays}
    last = wdays[-1]
    wl = level[last]                                  # water surface on the last day
    ice_bottom = wl - ice[last]
    precip = sum(v["precip"] or 0 for d, v in days.items() if dt.date(y0w, 11, 1) <= d <= dt.date(y0w + 1, 3, 31))
    gusts = [(v["gust_dir"] * 10, v["gust_spd"]) for d, v in days.items()
             if dt.date(y0w, 11, 1) <= d <= dt.date(y0w + 1, 3, 31) and v["gust_dir"] and v["gust_spd"]]
    wind_u = float(np.mean([math.sin(math.radians(a + 180)) * s for a, s in gusts])) if gusts else 0.0

    a_ch, b_ch = water_edges(for_edges, wl, mid)
    y_horizon = float(Y(args.horizon))

    # 1. Sky: deep at the top, a cold glow at the horizon, fine cloud texture.
    yy = np.arange(cv.H, dtype=np.float32)[:, None]
    t = np.clip(yy / y_horizon, 0, 1) ** 1.6
    top, glow = np.array(args.sky_top), np.array(args.sky_glow)
    cv.rgb[:] = top + (glow - top) * t[..., None]
    cloud = fbm((cv.H // 4, cv.W // 4), rng, scales=(40, 14, 5), weights=(0.6, 0.3, 0.1), aniso=(0.35, 1.0))
    cloud = np.asarray(Image.fromarray((cloud * 255).astype(np.uint8)).resize((cv.W, cv.H), Image.BILINEAR), np.float32) / 255
    band = np.exp(-((yy - y_horizon * 0.55) / (y_horizon * 0.35)) ** 2)
    cv.over(args.sky_glow, np.clip((cloud - 0.45) * 1.4, 0, 1) * band * 0.25)

    def draw_tree(draw, x, yb, h_px, w_px, width, depth=9):
        """A leafless tree grown naturally, then fitted to the lidar crown:
        height h_px, crown width w_px."""
        segs = []

        def seg(x, y, ang, length, wdt, d):
            x2, y2 = x + math.cos(ang) * length, y + math.sin(ang) * length
            segs.append((x, y, x2, y2, wdt))
            if d == 0:
                return
            n = 2 if rng.random() < 0.65 else 3
            for i in range(n):
                a2 = ang + (i - (n - 1) / 2) * rng.uniform(0.35, 0.75) + rng.normal(0, 0.15)
                a2 = a2 * 0.85 + (-math.pi / 2) * 0.15            # a slight lift toward the light
                seg(x2, y2, a2, length * rng.uniform(0.66, 0.8), wdt * 0.6, d - 1)

        trunk = rng.uniform(0.18, 0.3)
        segs.append((0, 0, rng.normal(0, 0.03), -trunk, 1.0))
        for _k in range(2 if rng.random() < 0.6 else 3):                  # forked low, like elm and cottonwood
            seg(segs[0][2], -trunk, -math.pi / 2 + rng.normal(0, 0.45), 0.2, 0.7, depth)
        pts = np.array([(a, b) for s_ in segs for a, b in ((s_[0], s_[1]), (s_[2], s_[3]))])
        top = pts[:, 1].min()
        sx = w_px / max(np.ptp(pts[:, 0]), 1e-3)
        sy = h_px / max(-top, 1e-3)
        for x1_, y1_, x2_, y2_, wd in segs:
            draw.line([(x + x1_ * sx, yb + y1_ * sy), (x + x2_ * sx, yb + y2_ * sy)],
                      fill=int(255 * min(1.0, 0.25 + 2.5 * wd)), width=max(1, int(round(width * wd))))

    drift_tex = texture.detail("snow", (cv.H, cv.W), rng, R * 60,
                               lambda: (fbm((cv.H // 2, cv.W // 2), rng, scales=(10, 3), weights=(0.6, 0.4), aniso=(0.2, 1))))
    if drift_tex.shape != (cv.H, cv.W):
        drift_tex = np.asarray(Image.fromarray(drift_tex.astype(np.float32), mode="F").resize((cv.W, cv.H)), np.float32)
    drift_tex = (drift_tex - drift_tex.min()) / (np.ptp(drift_tex) + 1e-6)

    # 2. The receding valley: lidar rows north of the cut, far (hazy) to near.
    trunks = fbm((cv.H // 2, cv.W // 2), rng, scales=(1.0, 0.5), weights=(0.6, 0.4), aniso=(40, 1))
    trunks = np.asarray(Image.fromarray(trunks.astype(np.float32), mode="F").resize((cv.W, cv.H)), np.float32)
    trunks = np.clip((trunks - 0.5) * 4, 0, 1)
    speck = fbm((cv.H // 2, cv.W // 2), rng, scales=(2.0, 0.8), weights=(0.5, 0.5), aniso=(0.6, 1.4))
    speck = np.asarray(Image.fromarray(speck.astype(np.float32), mode="F").resize((cv.W, cv.H)), np.float32)
    def haze_at(y):
        return tuple(float(v) for v in top + (glow - top) * np.clip(y / y_horizon, 0, 1) ** 1.6)

    for k, dist in enumerate(sorted(args.far_rows, reverse=True)):
        r = row - dist
        if r < 0:
            continue
        g = ndimage.uniform_filter1d(dtm[r, c0:c1].astype(float), 5)
        cnp = np.clip(dsm[r, c0:c1] - dtm[r, c0:c1], 0, 40)
        cnp = np.where(cnp > 2.5, cnp, 0)
        near = 1 - dist / max(args.far_rows)
        haze = 0.93 - 0.6 * near                        # how much the far row fades into the sky
        gy = Y(at_px(g))
        # Snow-covered ground mass, textured with drift and lit along its crest.
        m = (yy >= gy[None, :]).astype(np.float32)
        snow_col = np.array(mix(args.snow, haze_at(y_horizon), haze), np.float32)
        body_col = np.array(mix(args.tree, haze_at(y_horizon), 0.25 + 0.6 * haze), np.float32)
        fall = 1 - np.exp(-np.clip(yy - gy[None, :], 0, None) / (R * (1.5 + 3 * near)))
        # The bank faces are wooded where the lidar shows canopy: a stand of
        # thin dark trunks over the snow, fading into the valley fog below.
        wood = at_px(np.where(cnp > 2.5, 1.0, 0.0))
        wood = ndimage.gaussian_filter1d(np.nan_to_num(wood), R * 2)
        col = snow_col + (body_col - snow_col) * fall[..., None]
        col = col * (1 - 0.6 * (1 - haze) * (0.4 + 0.6 * wood[None, :, None]) * trunks[..., None])
        # Snow lying in patches on the slope between the trees.
        patch = np.clip((drift_tex - 0.55) * 4, 0, 1) * np.clip((speck - 0.45) * 3, 0, 1)
        col = col + (snow_col - col) * (patch * 0.5 * (1 - 0.6 * haze))[..., None]
        col = col * (1 + 0.12 * (1 - haze) * (drift_tex[..., None] - 0.5))
        fog_below = np.exp(-np.clip(yy - gy[None, :], 0, None) / (R * (40 + 60 * near)))
        cv.over(col, m * (0.55 + 0.4 * near) * (0.35 + 0.65 * fog_below))
        rim = np.exp(-np.clip(yy - gy[None, :], 0, None) / (R * 0.8)) * m
        cv.over(args.snow, rim * 0.2 * near * (0.5 + drift_tex))
        # Bare trees at each lidar crown on the row, softened by distance.
        hgt_r = np.where((cnp > 3) & (cnp < 40), cnp, np.nan)
        img = cv.mask()
        dr = ImageDraw.Draw(img)
        for i in crown_tops(hgt_r, 6.0, max(4, args.tree_spacing - 2)):
            h_px = float(hgt_r[i] * sec.sv * R)
            draw_tree(dr, float(X(i + 0.5)), float(Y(g[i])), h_px, h_px * rng.uniform(0.45, 0.7),
                      width=R * 0.35, depth=7)
        a_t = arr(blur(img, 0.8 + 3.5 * (1 - near)))
        cv.over(mix(args.tree, haze_at(y_horizon), haze), a_t * (0.35 + 0.5 * near))

    # 3. Low valley mist at the level of the ice.
    yi = float(Y(wl))
    mist = np.exp(-((yy - yi) / (R * 14)) ** 2) * (0.8 + 0.2 * cloud)
    cv.over(args.sky_glow, np.clip(mist * 0.35, 0, 1))

    # 4. Trees on and just behind the cut, from lidar crown tops.
    for k, back in enumerate(sorted(args.tree_rows, reverse=True)):
        r = row - back
        hgt = dsm[r, c0:c1] - dtm[r, c0:c1]
        hgt = np.where((hgt > 3) & (hgt < 40), hgt, np.nan)
        tops = crown_tops(hgt, 6.0, args.tree_spacing)
        img = cv.mask()
        dr = ImageDraw.Draw(img)
        for i in tops:
            x = float(X(i + 0.5))
            yb = float(Y(dtm[r, c0 + i]))
            h_px = float(hgt[i] * sec.sv * R)
            w_px = h_px * rng.uniform(0.4, 0.6)        # natural crown proportion (not to scale)
            draw_tree(dr, x, yb, h_px, w_px, width=R * 0.5 * (1 - 0.3 * k / max(1, len(args.tree_rows))))
        a_t = arr(blur(img, 0.6 + 0.6 * back / max(args.tree_rows)))
        haze = back / max(1, max(args.tree_rows)) * 0.5
        cv.over(mix(args.tree, haze_at(y_horizon), haze), a_t * 0.95)

    # 5. The cut: ground below the lidar surface, filled with the well-log strata.
    gy_cut = Y(at_px(ground))
    cut_top = np.where(in_channel[col_idx], Y(at_px(bed)), gy_cut)
    below = yy >= cut_top[None, :]
    tops = geology.interfaces(np.arange(c0, c1), lambda c: float(dtm[row, int(c)]))
    layer_y = {k: Y(at_px(v)) for k, v in tops.items()}
    strata = np.zeros((cv.H, cv.W), np.int8)
    for i, kname in enumerate(geology.LAYERS):
        strata[yy >= np.minimum(layer_y[kname], 1e9)[None, :]] = i
    tex_l = fbm((cv.H, cv.W), rng, scales=(1.5, 0.6), weights=(0.6, 0.4), aniso=(0.25, 6))   # laminae
    tex_g = fbm((cv.H, cv.W), rng, scales=(4, 1.2), weights=(0.5, 0.5))                       # grain
    base = np.array(args.strata, np.float32)                         # one tone per layer
    col = base[strata]
    lam = np.where(strata[..., None] == 0, (tex_l[..., None] - 0.5) * 0.22, 0)
    sh = np.where(strata[..., None] == 2, (tex_l[..., None] - 0.5) * 0.18, 0)
    clasts = np.where(strata[..., None] == 1, np.clip((tex_g[..., None] - 0.6) * 1.2, 0, 0.25)
                      - np.clip((0.35 - tex_g[..., None]) * 0.4, 0, 0.06), 0)
    vugs = np.where(strata[..., None] == 3, np.clip((tex_g[..., None] - 0.68) * 1.0, 0, 0.15)
                    - np.clip((0.3 - tex_g[..., None]) * 0.8, 0, 0.12), 0)
    beds = np.where(strata[..., None] == 3,
                    0.05 * np.sin((yy[..., None] / R + 6 * tex_l[..., None]) / 2.2 * 2 * np.pi), 0)
    col = col + lam + sh + clasts + vugs + beds
    # Large, soft mottling (weathering, moisture) and per-material grain:
    # photographs from data/textures if on file, otherwise procedural.
    mot = fbm((cv.H // 4, cv.W // 4), rng, scales=(60, 20, 6), weights=(0.5, 0.3, 0.2), aniso=(0.5, 1.0))
    mot = np.asarray(Image.fromarray(mot.astype(np.float32), mode="F").resize((cv.W, cv.H), Image.BICUBIC), np.float32)
    col = col * (1 + 0.5 * (mot[..., None] - 0.5))
    for i, (tname, tsize) in enumerate((("clay", 80), ("till", 60), ("shale", 60), ("limestone", 120))):
        sel = strata == i
        if not sel.any():
            continue
        tdet = texture.photo(tname, (cv.H, cv.W), R * tsize, rng)
        if tdet is not None:
            col = col * (1 + 0.12 * np.where(sel, tdet, 0)[..., None])
    depth_dark = np.clip((yy - cut_top[None, :]) / (R * 220), 0, 1)[..., None]
    col = col * (1 - 0.4 * depth_dark)
    cv.over(col, below.astype(np.float32))
    # Strata boundaries, faint.
    edge = np.zeros((cv.H, cv.W), np.float32)
    for kname in geology.LAYERS[1:]:
        ly = layer_y[kname].astype(int)
        ok = (ly > 0) & (ly < cv.H - 1) & (ly > cut_top)
        edge[ly[ok], xs_px[ok]] = 1
    cv.over((0.6, 0.6, 0.6), ndimage.gaussian_filter(edge, 1.2) * 0.8 * below)
    # Joints in the carbonate: near-vertical fractures and bedding partings,
    # the paths groundwater takes through the aquifer (interpretive).
    jm = cv.mask()
    jd = ImageDraw.Draw(jm)
    carb = layer_y["carbonate"]
    for _ in range(int(args.joints * cv.W / 1000)):
        x = rng.uniform(0, cv.W)
        y = float(carb[int(x)]) + rng.uniform(0, cv.H - float(carb[int(x)]))
        pts = [(x, y)]
        vertical = rng.random() < 0.6
        for _k in range(int(rng.integers(10, 80))):
            if vertical:
                x += rng.normal(0, 0.6)
                y += R * 0.6
            else:
                x += R * 0.8
                y += rng.normal(0, 0.35)
            if not (0 <= x < cv.W and carb[min(int(x), cv.W - 1)] < y < cv.H):
                break
            pts.append((x, y))
        if len(pts) > 2:
            jd.line(pts, fill=int(rng.integers(60, 200)), width=1)
    cv.over(args.joint, arr(blur(jm, 0.5)) * 0.6 * below)


    # Roots below the trees on the cut: branching, tapering, into the clay
    # to about a third of the tree's height, rarely past the till (interpretive).
    rm = cv.mask()
    rd = ImageDraw.Draw(rm)
    hgt0 = dsm[row, c0:c1] - dtm[row, c0:c1]
    hgt0 = np.where((hgt0 > 3) & (hgt0 < 40), hgt0, np.nan)

    def root(x, y, ang, length, wdt, d, y_max):
        for _k in range(int(length)):
            ang += rng.normal(0, 0.18)
            ang = ang * 0.92 + (math.pi / 2) * 0.08                       # gravity
            x2, y2 = x + math.cos(ang) * R * 0.5, y + math.sin(ang) * R * 0.5
            if y2 > y_max:
                return
            rd.line([(x, y), (x2, y2)], fill=int(90 + 120 * wdt), width=max(1, int(round(wdt * R * 0.5))))
            x, y = x2, y2
            wdt *= 0.985
            if d < 4 and rng.random() < 0.05:
                root(x, y, ang + rng.choice((-1, 1)) * rng.uniform(0.4, 1.0), length * 0.5, wdt * 0.6, d + 1, y_max)

    for i in crown_tops(hgt0, 6.0, args.tree_spacing):
        x0_, y0_ = float(X(i + 0.5)), float(Y(dtm[row, c0 + i]))
        depth_px = min(hgt0[i] * 0.33, 4.0) * sec.sv * R
        y_max = min(y0_ + depth_px, float(layer_y["shale"][min(int(x0_), cv.W - 1)]))
        for _k in range(int(rng.integers(5, 9))):
            root(x0_, y0_ + 1, rng.uniform(0.15, math.pi - 0.15), depth_px / (R * 0.5) * rng.uniform(0.5, 1.2), 1.0, 0, y_max)
    cv.over(args.root, arr(blur(rm, 0.5)) * 0.7 * below)

    # 6. Water under the ice: dark, lit from above through the ice in shafts.
    wtop = Y(ice_bottom)
    wbot = Y(at_px(bed))
    inwater = (yy >= wtop) & (yy < wbot[None, :]) & in_channel[col_idx][None, :] & \
              (xs_px[None, :] >= X(a_ch)) & (xs_px[None, :] <= X(b_ch + 1))
    dep = np.clip((yy - wtop) / (R * 2.6 * sec.sv), 0, 1)
    wcol = np.array(args.water_top) + (np.array(args.water_deep) - np.array(args.water_top)) * dep[..., None]
    cv.over(wcol, inwater.astype(np.float32))
    shafts = ndimage.gaussian_filter1d(rng.random(cv.W).astype(np.float32), R * 1.2)
    shafts = (shafts - shafts.min()) / (np.ptp(shafts) + 1e-6)
    light = (shafts[None, :] ** 3) * np.exp(-dep * 3.2) * inwater
    cv.over(args.ice_light, np.clip(light * 0.55, 0, 1))
    silt = fbm((cv.H // 2, cv.W // 2), rng, scales=(24, 8, 3), weights=(0.55, 0.3, 0.15), aniso=(0.4, 1.6))
    silt = np.asarray(Image.fromarray(silt.astype(np.float32), mode="F").resize((cv.W, cv.H), Image.BICUBIC), np.float32)
    cv.over(args.silt, np.clip((silt - 0.45) * 2.2, 0, 1) * (0.25 + 0.6 * dep) * inwater * 0.5)
    n_p = int(args.particles * (b_ch - a_ch))
    pm = cv.mask()
    pd = ImageDraw.Draw(pm)
    for _ in range(n_p):
        x = rng.uniform(X(a_ch), X(b_ch))
        y = rng.uniform(wtop, float(np.nanmax(wbot)))
        if inwater[min(int(y), cv.H - 1), min(int(x), cv.W - 1)]:
            r_ = rng.uniform(0.3, 1.2)
            pd.ellipse([x - r_, y - r_, x + r_, y + r_], fill=int(rng.uniform(60, 200)))
    cv.over(args.ice_light, arr(blur(pm, 0.5)) * 0.5)

    # 7. Fish: shaded bodies, sharp near the cut and softer behind it.
    placed = []
    fs = 1000.0 / args.fish_scale * R
    for sp in fish.SPECIES:
        for j in range(args.fish_each):
            for _ in range(300):
                c = int(rng.integers(a_ch + 3, b_ch - 3))
                d_here = ice_bottom - bed[c]
                if not d_here > 0.4:
                    continue
                z = bed[c] + d_here * rng.uniform(*sp["zone"])
                x, y = float(X(c)), float(Y(z))
                Lp = sp["length"] * fs * rng.uniform(0.85, 1.1)
                Dp = Lp * sp["depth"]
                facing = 1 if rng.random() < 0.5 else -1
                box = (min(x, x - facing * Lp * 1.05), max(x, x - facing * Lp * 1.05), y - Dp * 0.6, y + Dp * 0.6)
                if box[3] > Y(bed[c]) or box[2] < wtop:
                    continue
                if any(not (box[1] < q[0] - 4 or box[0] > q[1] + 4 or box[3] < q[2] - 2 or box[2] > q[3] + 2)
                       for q in placed):
                    continue
                placed.append(box)
                sp.setdefault("at", []).append((x, y, facing))
                far = rng.random()                                     # how far behind the cut
                fx0, fy0 = int(box[0] - 6), int(box[2] - 6)
                pw, ph = int(box[1] - box[0] + 12), int(box[3] - box[2] + 12)
                img = Image.new("L", (pw, ph), 0)
                d_ = ImageDraw.Draw(img)
                tt = np.linspace(0, 1, 40)
                half = Dp / 2 * np.sin(np.pi * tt ** 0.75) * (1 - 0.65 * tt ** 2)
                bx = x - facing * tt * Lp * 0.82 - fx0
                pts = [(a, y - fy0 - h) for a, h in zip(bx, half)] + \
                      [(a, y - fy0 + h * 0.85) for a, h in zip(bx[::-1], half[::-1])]
                d_.polygon(pts, fill=255)
                tx = x - facing * Lp * 0.82 - fx0
                d_.polygon([(tx, y - fy0 - Dp * 0.08), (tx - facing * Lp * 0.18, y - fy0 - Dp * 0.48),
                            (tx - facing * Lp * 0.1, y - fy0), (tx - facing * Lp * 0.18, y - fy0 + Dp * 0.42),
                            (tx, y - fy0 + Dp * 0.08)], fill=255)
                d0 = x - facing * Lp * 0.35 - fx0
                d_.polygon([(d0, y - fy0 - Dp * 0.45), (d0 - facing * Lp * 0.07, y - fy0 - Dp * 0.8),
                            (d0 - facing * Lp * 0.18, y - fy0 - Dp * 0.4)], fill=255)
                a_f = arr(blur(img, 0.4 + 1.6 * far))
                gyy = (np.arange(ph)[:, None] - (y - fy0)) / max(Dp / 2, 1)
                shade = np.clip(0.5 + 0.5 * gyy, 0, 1)[..., None]           # dark back, pale belly
                body = np.array(args.fish_back) + (np.array(args.fish_belly) - np.array(args.fish_back)) * shade
                body = np.broadcast_to(body, (ph, pw, 3))
                cv.over(body, a_f * (1 - 0.45 * far), fx0, fy0)
                break

    # 8. Ice: cloudy white ice formed before the new year over clear ice grown
    # down after it; one faint band per recorded day; threads hanging below.
    yt, yb_ = float(Y(wl)), float(Y(ice_bottom))
    xa, xb = float(X(a_ch)), float(X(b_ch + 1))
    ice_m = ((yy >= yt) & (yy < yb_) & (xs_px[None, :] >= xa) & (xs_px[None, :] < xb)).astype(np.float32)
    jan1 = min(wdays, key=lambda d: abs((d - dt.date(y0w + 1, 1, 1)).days))
    y_clear = yt + (yb_ - yt) * ice[jan1] / ice[last]
    y_clear = y_clear + ndimage.gaussian_filter1d(rng.standard_normal(cv.W), R * 4) * R * 3
    clear = np.clip((yy - y_clear[None, :]) / 2.0 + 0.5, 0, 1)
    tint = fbm((cv.H // 2, cv.W // 2), rng, scales=(8, 3, 1), aniso=(0.3, 1))
    tint = np.asarray(Image.fromarray(tint.astype(np.float32), mode="F").resize((cv.W, cv.H)), np.float32)
    white = np.array(args.snow) * (0.92 + 0.12 * tint[..., None])
    blue = np.array(args.ice) * (0.75 + 0.35 * tint[..., None])
    cv.over(white + (blue - white) * clear[..., None], ice_m * 0.95)
    bands = np.zeros((cv.H, cv.W), np.float32)
    for d in wdays:
        yb_d = int(Y(level[d] - ice[d]))
        if 0 < yb_d < cv.H:
            bands[yb_d, int(xa):int(xb)] += 0.25
    cv.over(args.ice_light, np.clip(ndimage.gaussian_filter(bands, 0.4), 0, 1) * ice_m * 0.6)
    th = cv.mask()
    td = ImageDraw.Draw(th)
    for _ in range(int(args.threads * (b_ch - a_ch))):
        x = rng.uniform(xa, xb)
        y = yb_ - rng.uniform(0, 2)
        pts = [(x, y)]
        for _k in range(int(rng.lognormal(2.2, 0.6))):
            x += rng.normal(0, 0.5)
            y += R * 0.3
            pts.append((x, y))
        td.line(pts, fill=int(rng.integers(50, 180)), width=1)
    cv.over(args.ice_light, arr(blur(th, 0.4)) * inwater * 0.6)
    glow = np.exp(-np.clip(yy - yb_, 0, None) / (R * 3)) * inwater
    cv.over(args.ice_light, glow * 0.25)

    # 9. Snow: lying on the ground and the ice, falling through the sky.
    snow_t = args.snow_depth * sec.sv * R
    top_surface = np.where(in_channel[col_idx] & (xs_px >= xa) & (xs_px <= xb), yt, gy_cut)
    drift_1d = ndimage.gaussian_filter1d(rng.standard_normal(cv.W), R * 6)
    thick = snow_t * np.clip(1 + 1.6 * drift_1d / (drift_1d.std() + 1e-6) * 0.4, 0.3, 2.2)
    s_top = top_surface - thick
    sl = ((yy >= s_top[None, :]) & (yy < top_surface[None, :] + 1)).astype(np.float32)
    sl = ndimage.gaussian_filter(sl, 0.6)
    in_s = np.clip((yy - s_top[None, :]) / np.maximum(thick, 1)[None, :], 0, 1)
    scol = np.array(args.snow) * (1.05 - 0.25 * in_s[..., None]) * (1 + 0.06 * (drift_tex[..., None] - 0.5))
    cv.over(scol, sl * 0.97)
    n_flakes = int(precip * args.flakes_per_mm)
    for layer, (n_share, size, bl, op) in enumerate(((0.65, 0.5, 0.2, 0.5), (0.27, 1.1, 0.8, 0.6), (0.08, 3.0, 3.0, 0.45))):
        img = cv.mask()
        d_ = ImageDraw.Draw(img)
        slant = wind_u * 0.12 * (1 + layer)
        for _ in range(int(n_flakes * n_share)):
            x, y = rng.uniform(0, cv.W), rng.uniform(0, cv.H * 0.95)
            r_ = size * R * 0.2 * rng.uniform(0.6, 1.4)
            d_.line([(x, y), (x + slant * r_, y + 2.2 * r_)], fill=255, width=max(1, int(r_)))
        a_s = arr(blur(img, bl))
        above = (yy < np.minimum(top_surface, gy_cut)[None, :]).astype(np.float32)
        cv.over(args.snow, a_s * op * above)

    # 10. Grade: vignette and grain.
    vx = (xs_px[None, :] / cv.W - 0.5) * 2
    vy = (yy / cv.H - 0.5) * 2
    vig = np.clip(1 - 0.35 * (vx ** 2 + vy ** 2), 0.4, 1)
    cv.rgb *= vig[..., None]
    cv.rgb += (rng.standard_normal((cv.H, cv.W, 1)).astype(np.float32) * 0.012)
    np.clip(cv.rgb, 0, 1, out=cv.rgb)

    raster = Image.fromarray((cv.rgb * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    raster_path = os.path.splitext(args.out)[0] + "_render.jpg"
    raster.save(raster_path, quality=92)

    # ------------------------------------------------------------ data overlay
    svg = Svg(Wp, Hp, background=None)
    L = dict(render="RENDER (RASTER COMPOSITE)", water="DATA · WATER SURFACES, EVERY DAY · 0.04 mm",
             ice="DATA · UNDERSIDE OF THE ICE, EVERY DAY · 0.04 mm", bed="DATA · RIVERBED (CHART), DASHED · 0.12 mm",
             cut="DATA · GROUND (LIDAR) · 0.12 mm", datum="DATUM TICKS · 0.1 mm", fishlab="FISH LABELS · 0.06 mm",
             key="KEY PLAN", scale="SCALE BARS", text="TEXT")
    for name in L.values():
        svg.layer(name)
    svg.layer(L["render"])
    buf = io.BytesIO()
    raster.save(buf, format="JPEG", quality=90)
    svg.image(0, 0, Wp, Hp, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())

    svg.layer(L["water"])
    for y in range(2002, 2008):
        for d in (d for d in level if dt.date(y, 12, 1) <= d <= dt.date(y + 1, 3, 10)):
            a, b = water_edges(for_edges, level[d], mid)
            svg.line(sec.X(a), sec.Y(level[d]), sec.X(b + 1), sec.Y(level[d]), stroke=INK, stroke_width=0.04, opacity=0.12)
    svg.layer(L["ice"])
    for d in wdays:
        a, b = water_edges(for_edges, level[d], mid)
        svg.line(sec.X(a), sec.Y(level[d] - ice[d]), sec.X(b + 1), sec.Y(level[d] - ice[d]), stroke=INK,
                 stroke_width=0.04, opacity=0.35)
    svg.layer(L["bed"])
    svg.path(sec.path(ch, np.where(in_channel, bed, np.nan)), fill="none", stroke=INK, stroke_width=0.12,
             stroke_dasharray="1 0.6", opacity=0.7)
    svg.layer(L["cut"])
    svg.path(sec.path(ch, np.where(in_channel, np.nan, ground)), fill="none", stroke=INK, stroke_width=0.12, opacity=0.8)

    # Datum: ticks and labels at the left edge only.
    svg.layer(L["datum"])
    for z in range(int(math.ceil(args.z_top - Hp / sec.sv)) // 5 * 5, int(args.z_top), 5):
        y = sec.Y(z)
        if 20 < y < Hp - 20:
            svg.line(6, y, 10, y, stroke=INK, stroke_width=0.1, opacity=0.6)
            svg.layer(L["text"])
            svg.text(11.5, y + 0.6, f"+{z}", size=1.7, fill=INK, opacity=0.6)
            svg.layer(L["datum"])

    # Fish: one label per species, leader down into the ground.
    for k, sp in enumerate(fish.SPECIES):
        if not sp.get("at"):
            continue
        x, y, _ = sp["at"][0]
        x, y = x / R, y / R
        ly = sec.Y(float(np.nanmin(bed))) + 10 + k * 4.2
        svg.layer(L["fishlab"])
        svg.path(f"M{x:.2f},{y:.2f} L{x:.2f},{ly:.2f} L{x + 6:.2f},{ly:.2f}", fill="none", stroke=INK,
                 stroke_width=0.06, opacity=0.7)
        svg.layer(L["text"])
        svg.text(x + 7, ly + 0.6, sp["name"] + (f" · {sp['status'].split(' (')[0]}" if sp["status"] else ""),
                 size=1.7, fill=INK, spacing=0.2)
        sp.pop("at", None)

    # Title and a short caption, top left; section letters.
    svg.layer(L["text"])
    svg.text(14, 18, "THE RIVER IN WINTER", size=6.0, fill=INK, spacing=0.6)
    svg.text(14, 25, f"SECTION A–A′ · RED RIVER BELOW ST. ANDREWS LOCK AND DAM · WINTER {y0w}–{str(y0w + 1)[2:]} · "
             f"H 1:{args.h_scale:,} V 1:{args.v_scale:,}", size=2.2, fill=INK, spacing=0.3, opacity=0.85)
    caption = [
        f"Looking north. Ice grown day by day from Winnipeg's cold (Stefan's law), {ice[last]:.2f} m by {last.strftime('%B %d')}; "
        f"each band one day. Snow: {precip:.0f} mm of winter precipitation.",
        "Valley behind the cut and every tree's place and height from 1 m lidar (crowns drawn in natural proportion); riverbed from chart soundings; ground from two drillers' logs; "
        "fish by winter habit (diagram, 1:{0}).".format(args.fish_scale),
    ]
    for j, c in enumerate(caption):
        svg.text(14, 30 + 2.8 * j, c, size=1.7, fill=INK, opacity=0.7)
    svg.text(4, sec.Y(args.horizon) - 4, "A", size=5, fill=INK)
    svg.text(Wp - 4, sec.Y(args.horizon) - 4, "A′", size=5, fill=INK, anchor="end")

    svg.layer(L["scale"])
    scale_bar(svg, 14, Hp - 12, 100 * sec.sh, 100, "m", color=INK)
    vx_ = 14 + 100 * sec.sh + 30
    svg.line(vx_, Hp - 11.2, vx_, Hp - 11.2 - 5 * sec.sv, stroke=INK, stroke_width=0.15)
    svg.text(vx_ + 2, Hp - 11.2, "5 m", size=1.8, fill=INK)

    svg.layer(L["key"])
    kw = args.key_window
    key_plan(svg, dtm, Wp - 14 - (kw[3] - kw[2]) * 1000 / args.key_scale, Hp - 14 - (kw[1] - kw[0]) * 1000 / args.key_scale - 8,
             kw, row, args.cols, args.dam_row, args.dam_cols, INK, scale=args.key_scale,
             angle=true_north_angle(transform, crs, (kw[0] + kw[1]) / 2, (kw[2] + kw[3]) / 2),
             layers=dict(lines=L["key"], cut=L["key"], text=L["text"]))

    svg.save(args.out)
    print(f"wrote {args.out} and {raster_path}  ({cv.W}x{cv.H} px; ice {ice[last]:.2f} m on {last}; precip {precip:.0f} mm; "
          f"wind east part {wind_u:.1f} km/h; fish {len(placed)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--level-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--climate", default="data/lockport/climate/en_climate_daily_MB_5023222_*.csv")
    ap.add_argument("--winter", type=int, default=2007, help="winter starting this year (2007 = the coldest on file)")
    ap.add_argument("--row", type=int, default=1840)
    ap.add_argument("--cols", type=int, nargs=2, default=[1150, 1700])
    ap.add_argument("--h-scale", type=int, default=600)
    ap.add_argument("--v-scale", type=int, default=120)
    ap.add_argument("--z-top", type=float, default=261.0, help="elevation at the top edge of the sheet")
    ap.add_argument("--horizon", type=float, default=231.0, help="elevation of the horizon glow")
    ap.add_argument("--far-rows", type=int, nargs="+", default=[60, 120, 200, 300, 420],
                    help="m north of the cut for each receding lidar profile")
    ap.add_argument("--tree-rows", type=int, nargs="+", default=[0, 12, 26], help="m behind the cut for drawn trees")
    ap.add_argument("--tree-spacing", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.015)
    ap.add_argument("--snow-depth", type=float, default=0.3, help="m of snow drawn lying on ground and ice")
    ap.add_argument("--flakes-per-mm", type=float, default=60)
    ap.add_argument("--particles", type=float, default=6, help="suspended particles per m of channel")
    ap.add_argument("--joints", type=float, default=140, help="carbonate joints per 1000 px")
    ap.add_argument("--threads", type=float, default=20, help="threads under the ice per m of channel")
    ap.add_argument("--fish-scale", type=int, default=40)
    ap.add_argument("--fish-each", type=int, default=3)
    ap.add_argument("--res", type=float, default=5.0, help="raster px per mm (5 = 127 dpi)")
    ap.add_argument("--seed", type=int, default=2008)
    ap.add_argument("--key-window", type=int, nargs=4, default=[1150, 2950, 700, 2900])
    ap.add_argument("--key-scale", type=int, default=25000)
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--dam-cols", type=int, nargs=2, default=[1255, 1595])
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--out", default="outputs/sheets/04a_winter_scene.svg")
    # Palette (rgb 0-1): cold, near monochrome, ice the only accent.
    ap.set_defaults(sky_top=(0.03, 0.04, 0.055), sky_glow=(0.36, 0.41, 0.47), snow=(0.8, 0.84, 0.88),
                    tree=(0.04, 0.045, 0.05), water_top=(0.09, 0.15, 0.18), water_deep=(0.015, 0.025, 0.03),
                    ice=(0.7, 0.84, 0.9), ice_light=(0.78, 0.92, 0.97), fish_back=(0.05, 0.06, 0.06),
                    fish_belly=(0.42, 0.46, 0.45), silt=(0.2, 0.26, 0.26), joint=(0.55, 0.57, 0.56),
                    root=(0.36, 0.31, 0.26),
                    strata=[(0.29, 0.265, 0.24), (0.23, 0.215, 0.195), (0.16, 0.17, 0.18), (0.27, 0.275, 0.28)])
    build(ap.parse_args())


if __name__ == "__main__":
    main()
