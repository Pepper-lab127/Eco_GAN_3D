"""Sheet 11 - The three site models at 1:500, as a print.

Each model's ground is drawn at the models' own scale (1:500), so a map is
the same size as its model (800 x 440 mm):

  - a stylised orthographic image made from the 1 m lidar (aerial imagery is
    not reachable from here): ground shading, tree crowns from the surface
    model, flat water, film grain
  - the elevation lines the model is cut from (its layer outlines, read from
    the model's own DXF), roads, buildings and structures (Overture Maps)
  - open-data overlays: the 1997 crest flood streamlines (as sheet 01, each
    line 8 m3/s) and a year of hourly wind at 2 m (as sheet 04b)
  - under each map, a section through both banks at 1:500, true scale, with
    the geology from the two well logs, the model's card steps and its base

A fine grid runs under everything; a locator, north arrow (a circle with a
line pointing to grid north) and a scale bar sit at bottom right.

    python -m sheets.model_print [--px 0.25] [--out ...]
"""

import argparse
import base64
import glob
import io
import json
import math
import os

import ezdxf
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio import features
from scipy import ndimage
from shapely.geometry import LineString, box, shape
from shapely.ops import transform as shp_transform, unary_union

from drawing import climate, geology, wind
from drawing.section import Section
from drawing.svg import Svg, points_to_path

INK = "#111111"
MODELS = [
    ("A", "THE BRIDGE ON ST. ANDREWS DAM", (2030, 2250, 1245, 1645), "outputs/site_models/A_bridge_dam"),
    ("B", "THE FLOODWAY OUTLET", (1300, 1520, 1490, 1890), "outputs/site_models/B_floodway_outlet"),
    ("C", "OLD HENDERSON HIGHWAY", (1705, 1925, 1560, 1960), "outputs/site_models/C_old_henderson"),
]
ROAD_W = {"primary": 12.0, "secondary": 9.0, "residential": 7.0, "unclassified": 6.0, "service": 4.5,
          "track": 3.0, "path": 2.0, "footway": 2.0, "steps": 2.0}


def hillshade(z, px, az=315.0, alt=45.0):
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = math.radians(az), math.radians(alt)
    return np.clip(np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def ortho(dtm, dsm, win, px, water_max, rng):
    """A stylised orthographic image (uint8 grey) of the window at px m per pixel: pale ground
    shaded by the lidar, tree crowns as a stipple whose density follows the crown shading,
    pale water with a sparse stipple, and a light grain."""
    r0, r1, c0, c1 = win
    p = 4
    g = dtm[r0 - p:r1 + p, c0 - p:c1 + p].astype(float)
    s = dsm[r0 - p:r1 + p, c0 - p:c1 + p].astype(float)
    g = np.where(np.isfinite(g), g, np.nanmedian(g))
    s = np.where(np.isfinite(s), s, g)
    f = 1.0 / px
    gz = ndimage.zoom(ndimage.gaussian_filter(g, 0.8), f, order=3)
    sz = ndimage.zoom(ndimage.gaussian_filter(s, 0.6), f, order=3)
    cut = int(p * f)
    gz, sz = gz[cut:-cut, cut:-cut], sz[cut:-cut, cut:-cut]
    hs = (hillshade(gz, px, 315) * 0.5 + hillshade(gz, px, 260) * 0.25 + hillshade(gz, px, 20) * 0.25)
    tone = 0.90 + 0.22 * (hs - 0.65)
    h = np.clip(sz - gz, 0, None)
    tree = ndimage.gaussian_filter((h > 2.5).astype(float), 0.8 * f) > 0.5
    crown = hillshade(sz, px, 315, 40)
    # Stipple: a pixel inside a crown is dark with a probability that grows where the crown is in shade.
    p_dark = np.clip(0.75 - 0.65 * crown + 0.01 * np.clip(h, 0, 25), 0.05, 0.9)
    dots = rng.random(tone.shape) < p_dark
    tone = np.where(tree, np.where(dots, 0.22, 0.86), tone)
    water = ndimage.binary_opening(gz < water_max, iterations=int(2 * f)) & (h < 1.0)
    wd = rng.random(tone.shape) < 0.015
    tone = np.where(water, np.where(wd, 0.55, 0.965), tone)
    tone = tone + rng.normal(0, 0.012, tone.shape)
    return (np.clip(tone, 0, 1) * 255).astype(np.uint8), water


def png_uri(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build(a):
    rng = np.random.default_rng(11)
    Wp, Hp = a.sheet
    svg = Svg(Wp, Hp)
    L = dict(grid="GRID · 0.05–0.1 mm", ortho="ORTHO · STYLISED FROM LIDAR (RASTER)",
             contours="ELEVATION LINES · MODEL LAYERS · 0.12 mm", water="WATER EDGE · 0.15 mm",
             roads="ROADS · WHITE, 0.08 mm EDGES", blds="BUILDINGS · FILLED", struct="STRUCTURES · 0.3 mm",
             flood="FLOOD · 1997 CREST STREAMLINES · 0.07 mm", wind="WIND · 2024 HOURS AT 2 M",
             cut="SECTION LINES · DASHED", geo_clay="GEOLOGY · CLAY", geo_till="GEOLOGY · TILL",
             geo_shale="GEOLOGY · SHALE", geo_carbonate="GEOLOGY · CARBONATE", sec="SECTION · GROUND, MODEL STEPS, BASE",
             frame="FRAMES · 0.15 mm", locator="LOCATOR", north="NORTH ARROW", scale="SCALE BAR", text="TEXT")
    for v in L.values():
        svg.layer(v)

    # Grid over the whole sheet.
    svg.layer(L["grid"])
    d_minor, d_major = [], []
    for x in np.arange(0, Wp + 0.1, a.grid):
        (d_major if round(x / a.grid) % 5 == 0 else d_minor).append(f"M{x:.1f},0V{Hp:.1f}")
    for y in np.arange(0, Hp + 0.1, a.grid):
        (d_major if round(y / a.grid) % 5 == 0 else d_minor).append(f"M0,{y:.1f}H{Wp:.1f}")
    svg.path("".join(d_minor), fill="none", stroke="#bdbdbd", stroke_width=0.05, opacity=0.6)
    svg.path("".join(d_major), fill="none", stroke="#9a9a9a", stroke_width=0.1, opacity=0.6)

    src = rasterio.open(a.dtm)
    dtm = src.read(1).astype(float)
    dtm[dtm < -1000] = np.nan
    dsm = rasterio.open(a.dsm).read(1).astype(float)
    dsm[dsm < -1000] = np.nan
    left, top = src.transform.c, src.transform.f
    fwd = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    bed = rasterio.open(a.bed).read(1).astype(float) if os.path.exists(a.bed) else None
    if bed is not None:
        bed[bed < -1000] = np.nan
    roads = json.load(open(os.path.join(a.overture, "roads.geojson")))["features"]
    blds = json.load(open(os.path.join(a.overture, "buildings.geojson")))["features"]
    infra = json.load(open(os.path.join(a.overture, "infrastructure.geojson")))["features"]
    lines97 = json.load(open(a.streamlines)).get("1997-05-04", [])
    hourly, _ = climate.load_eccc_hourly_wind(a.hourly)
    mm = 1000.0 / a.scale

    # Layout: A and B on the top row, C below A; the bottom right holds the locator and keys.
    mw, mh = 400 * mm, 220 * mm
    gap_x = (Wp - 2 * mw) / 3
    sec_h = (a.z_top - a.z_bottom) * mm
    block = mh + 10 + sec_h + 8
    gap_y = (Hp - 2 * block) / 3
    pos = {"A": (gap_x, gap_y), "B": (2 * gap_x + mw, gap_y), "C": (gap_x, 2 * gap_y + block)}

    for key, title, win, mdir in MODELS:
        r0, r1, c0, c1 = win
        ox, oy = pos[key]
        W, H = (c1 - c0) * mm, (r1 - r0) * mm

        def to_map(g):
            return shp_transform(lambda lon, lat, z=None: (
                ox + (np.asarray(fwd.transform(lon, lat)[0]) - left - c0) * mm,
                oy + (top - np.asarray(fwd.transform(lon, lat)[1]) - r0) * mm), g)

        frame = box(ox, oy, ox + W, oy + H)
        clip = svg.clip_rect(ox, oy, W, H)
        # 1. Stylised ortho.
        img, water = ortho(dtm, dsm, win, a.px, a.water_max, rng)
        svg.layer(L["ortho"])
        svg.image(ox, oy, W, H, png_uri(img), opacity=a.ortho_opacity)
        # Water edge.
        svg.layer(L["water"])
        svg.open_group(clip_path=clip)
        wl = ndimage.binary_opening(np.nan_to_num(dtm[r0:r1, c0:c1], nan=999) < a.water_max, iterations=2)
        cs = __import__("contourpy").contour_generator(z=ndimage.gaussian_filter(wl.astype(float), 1.5))
        for ln in cs.lines(0.5):
            if len(ln) > 10:
                svg.path(points_to_path(ox + (ln[:, 0] + 0.5) * mm, oy + (ln[:, 1] + 0.5) * mm), fill="none",
                         stroke=INK, stroke_width=0.15)
        svg.close_group()
        # 2. Wind: each dot an hour of 2024 at 2 m, kept in proportion to its local speed, streak downwind.
        st = 2
        rows, cols = np.mgrid[r0:r1:st, c0:c1:st].astype(float)
        gg = dtm[r0:r1:st, c0:c1:st]
        can = np.clip(dsm[r0:r1:st, c0:c1:st] - gg, 0, None)
        fields, hsec = wind.sector_fields(dsm, gg, can, rows, cols, gg + 2.0, hourly)
        f_ref = float(wind.profile_factor(2.0, 0.0))
        spd = np.array([h[2] for h in hourly])
        deg = np.array([h[1] for h in hourly])
        n = int(W * H * a.wind_density)
        py = rng.uniform(0, rows.shape[0], n)
        pxx = rng.uniform(0, rows.shape[1], n)
        hrs = rng.choice(len(hourly), n, p=spd / spd.sum())
        fv = np.array([fields[hsec[h]][int(p_), int(q_)] for h, p_, q_ in zip(hrs, py, pxx)])
        keep = rng.random(n) < np.clip(fv / f_ref, 0, 1)
        loc = spd[hrs] * fv / f_ref
        tw = np.radians(deg[hrs] + 180)
        ln_ = loc * a.wind_streak
        X = ox + pxx * st * mm
        Y = oy + py * st * mm
        svg.layer(L["wind"])
        svg.open_group(clip_path=clip)
        svg.path("".join(f"M{x:.2f},{y:.2f}h0" for x, y in zip(X[keep], Y[keep])), fill="none", stroke=INK,
                 stroke_width=0.3, stroke_linecap="round", opacity=0.6)
        svg.path("".join(f"M{x:.2f},{y:.2f}l{math.sin(t) * l:.2f},{-math.cos(t) * l:.2f}"
                         for x, y, t, l in zip(X[keep], Y[keep], tw[keep], ln_[keep])),
                 fill="none", stroke=INK, stroke_width=0.08, opacity=0.55)
        svg.close_group()
        # 3. Flood: the 1997 crest streamlines.
        svg.layer(L["flood"])
        svg.open_group(clip_path=clip)
        d = []
        for rr, cc in lines97:
            rr, cc = np.asarray(rr), np.asarray(cc)
            if cc.max() < c0 or cc.min() > c1 or rr.max() < r0 or rr.min() > r1:
                continue
            d.append(points_to_path(ox + (cc - c0) * mm, oy + (rr - r0) * mm))
        svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.07, opacity=0.6, stroke_linejoin="round")
        svg.close_group()
        # 4. Elevation lines: the model's own layer outlines.
        dxf = ezdxf.readfile(os.path.join(mdir, "lockport_contour_layers.dxf"))
        svg.layer(L["contours"])
        svg.open_group(clip_path=clip)
        for e in dxf.modelspace():
            if e.dxf.layer == "MODEL_EDGE":
                continue
            pts = np.array([(p[0], p[1]) for p in e.get_points()])
            k = int(e.dxf.layer[1:3])
            svg.path(points_to_path(ox + pts[:, 0], oy + H - pts[:, 1]) + "Z", fill="none", stroke=INK,
                     stroke_width=0.18 if k % 2 == 0 else 0.1)
        svg.close_group()
        # 5. Roads, buildings, structures.
        rp = []
        for f in roads:
            cls = f["properties"].get("class") or ""
            g = to_map(shape(f["geometry"]))
            if g.intersects(frame):
                rp.append(g.buffer(ROAD_W.get(cls, 4.0) * mm / 2, cap_style="flat"))
        rp = unary_union(rp).intersection(frame) if rp else None
        svg.layer(L["roads"])
        if rp is not None and not rp.is_empty:
            for q in getattr(rp, "geoms", [rp]):
                if q.geom_type == "Polygon":
                    dd = points_to_path(*np.asarray(q.exterior.coords).T) + "Z"
                    for hole in q.interiors:
                        dd += points_to_path(*np.asarray(hole.coords).T) + "Z"
                    svg.path(dd, fill="#ffffff", stroke=INK, stroke_width=0.08, fill_rule="evenodd", opacity=0.95)
        svg.layer(L["blds"])
        for f in blds:
            g = to_map(shape(f["geometry"]))
            if g.intersects(frame) and g.geom_type == "Polygon":
                g = g.intersection(frame)
                if g.geom_type == "Polygon":
                    svg.path(points_to_path(*np.asarray(g.exterior.coords).T) + "Z", fill="#3d3d3d", stroke=INK,
                             stroke_width=0.12)
        svg.layer(L["struct"])
        svg.open_group(clip_path=clip)
        bridges = [f for f in infra if f["properties"].get("subtype") == "bridge"]
        longest = max(bridges, key=lambda f: shape(f["geometry"]).length) if bridges else None
        for f in infra:
            sub = f["properties"].get("subtype")
            g = to_map(shape(f["geometry"]))
            if not g.intersects(frame) or (sub == "bridge" and f is not longest):
                continue
            if sub == "bridge":
                svg.path(points_to_path(*np.asarray(g.buffer(6 * mm, cap_style="flat").exterior.coords).T) + "Z",
                         fill="#ffffff", stroke=INK, stroke_width=0.3)
            elif f["properties"].get("class") == "weir" and g.geom_type == "LineString":
                svg.path(points_to_path(*np.asarray(g.coords).T), fill="none", stroke=INK, stroke_width=0.6)
        svg.close_group()
        # Frame and title.
        svg.layer(L["frame"])
        svg.rect(ox, oy, W, H, fill="none", stroke=INK, stroke_width=0.15)
        svg.layer(L["text"])
        svg.text(ox, oy - 3, f"{key} · {title} · 1:{a.scale}", size=3.2, fill=INK, spacing=0.4)

        # 6. Section through both banks along the map's middle row.
        row = (r0 + r1) // 2
        svg.layer(L["cut"])
        yc = oy + (row - r0) * mm
        svg.line(ox, yc, ox + W, yc, stroke=INK, stroke_width=0.25, stroke_dasharray="6 1.5 1 1.5")
        svg.layer(L["text"])
        svg.text(ox - 2, yc + 1, key, size=3.0, fill=INK, anchor="end")
        svg.text(ox + W + 2, yc + 1, key + "′", size=3.0, fill=INK)
        sec = Section(ox, oy + H + 10, a.scale, a.scale, a.z_top)
        ch = np.arange(c1 - c0, dtype=float)
        colsr = np.arange(c0, c1)
        ground = ndimage.gaussian_filter1d(np.nan_to_num(dtm[row, c0:c1], nan=np.nanmedian(dtm[row, c0:c1])), 1.0)
        river = ground < a.water_max
        bed_row = bed[row, c0:c1] if bed is not None else np.full_like(ground, np.nan)
        cap = np.where(river & np.isfinite(bed_row), bed_row, np.where(river, ground - 3.0, ground))
        geology.draw_layers(svg, sec, ch, colsr, ground, lambda c: float(dtm[row, int(c)]), a.z_bottom, INK,
                            weight=0.08, top_cap=cap, label=True,
                            layer_names={k_: L[f"geo_{k_}"] for k_ in geology.LAYERS})
        svg.layer(L["sec"])
        svg.path(sec.path(ch, ground), fill="none", stroke=INK, stroke_width=0.3)
        # Water: surface and a fine hatch down to the bed.
        if river.any():
            lv = np.where(river, ground, np.nan)
            svg.path(sec.path(ch, lv), fill="none", stroke=INK, stroke_width=0.2)
            hd = []
            for z in np.arange(math.floor(np.nanmin(cap[river])), np.nanmax(lv) + 0.01, 0.5):
                seg = np.where(river & (cap <= z) & (lv >= z), z, np.nan)
                hd.append(sec.path(ch, seg))
            svg.path("".join(hd), fill="none", stroke=INK, stroke_width=0.05, opacity=0.6)
        # The model: its card steps and base board.
        stepped = np.where(ground >= a.model_base, a.model_base + np.floor((ground - a.model_base) / a.layer_m) * a.layer_m,
                           np.nan)
        svg.path(sec.path(ch, stepped), fill="none", stroke=INK, stroke_width=0.12, stroke_dasharray="1.2 0.6")
        yb = float(sec.Y(a.model_base))
        svg.line(sec.X(0), yb, sec.X(ch[-1]), yb, stroke=INK, stroke_width=0.2, stroke_dasharray="3 1")
        svg.rect(ox, oy + H + 10, W, sec_h, fill="none", stroke=INK, stroke_width=0.1, opacity=0.6)
        svg.layer(L["text"])
        svg.text(ox + W - 1, yb - 1.2, f"MODEL BASE {a.model_base:g} M", size=1.8, fill=INK, anchor="end", spacing=0.2)
        svg.text(ox, oy + H + 10 + sec_h + 5, f"{key}–{key}′ · GEOLOGY OF BOTH BANKS · 1:{a.scale}, TRUE SCALE",
                 size=2.2, fill=INK, spacing=0.3)
        for z in range(int(a.z_bottom), int(a.z_top) + 1, 5):
            svg.text(ox - 1.5, float(sec.Y(z)) + 0.7, f"{z}", size=1.6, fill=INK, anchor="end")
        print(f"{key}: map {W:.0f} x {H:.0f} mm at ({ox:.0f}, {oy:.0f}); {int(keep.sum())} wind dots")

    # ---------------------------------------------------------------- bottom right: locator, north, scale
    bx0, by0 = pos["B"][0], pos["C"][1]
    lr0, lr1, lc0, lc1 = 1250, 2300, 1150, 2050
    ls = 1000.0 / a.locator_scale
    g = dtm[lr0:lr1, lc0:lc1]
    loc_img = (np.clip(0.75 + 0.35 * (hillshade(ndimage.gaussian_filter(np.nan_to_num(g, nan=np.nanmean(g)), 1.5), 1.0)
                                      - 0.6), 0, 1) * 255).astype(np.uint8)
    lw, lh = (lc1 - lc0) * ls, (lr1 - lr0) * ls
    svg.layer(L["locator"])
    svg.image(bx0, by0, lw, lh, png_uri(loc_img), opacity=0.9)
    svg.rect(bx0, by0, lw, lh, fill="none", stroke=INK, stroke_width=0.1)
    for key, title, (r0, r1, c0, c1), _ in MODELS:
        svg.rect(bx0 + (c0 - lc0) * ls, by0 + (r0 - lr0) * ls, (c1 - c0) * ls, (r1 - r0) * ls, fill="none",
                 stroke=INK, stroke_width=0.35)
        svg.layer(L["text"])
        svg.text(bx0 + (c0 - lc0) * ls + 1.5, by0 + (r0 - lr0) * ls + 4, key, size=3.0, fill=INK)
        svg.layer(L["locator"])
    svg.layer(L["text"])
    svg.text(bx0, by0 - 3, f"LOCATOR · 1:{a.locator_scale:,}", size=2.2, fill=INK, spacing=0.3)
    # The three models as dark blocks, in a row under the locator (renders of their OBJs).
    bw = (Wp - gap_x - bx0 - lw - 25) if False else None
    row_y = by0 + lh + 22
    slot = (Wp - gap_x - bx0) / 3
    for i, (key, title, _, mdir) in enumerate(MODELS):
        pth = os.path.join(mdir, "model_block_dark.png")
        if not os.path.exists(pth):
            continue
        im = Image.open(pth)
        w_mm = slot - 12
        h_mm = w_mm * im.height / im.width
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        svg.layer(L["locator"])
        svg.image(bx0 + i * slot, row_y, w_mm, h_mm, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        svg.layer(L["text"])
        svg.text(bx0 + i * slot, row_y + h_mm + 5, f"{key} · MODEL, 3 MM CARD ON A 218 M BASE", size=2.0, fill=INK,
                 spacing=0.25)
    # North arrow: a circle and a line pointing to grid north; scale bar 0-50 m at 1:500, bottom right.
    cx, cy, rr = Wp - gap_x - 70, Hp - gap_y - 22, 9.0
    svg.layer(L["north"])
    svg.circle(cx, cy, rr, fill="none", stroke=INK, stroke_width=0.3)
    svg.line(cx, cy + rr * 0.75, cx, cy - rr * 1.6, stroke=INK, stroke_width=0.45)
    svg.layer(L["text"])
    svg.text(cx, cy - rr * 1.6 - 2, "N", size=3.0, fill=INK, anchor="middle")
    svg.layer(L["scale"])
    x0s, y0s = Wp - gap_x - 50 * mm - 5, Hp - gap_y - 14
    for i in range(5):
        svg.rect(x0s + i * 10 * mm, y0s, 10 * mm, 2.0, fill=INK if i % 2 == 0 else "#ffffff", stroke=INK,
                 stroke_width=0.15)
    svg.layer(L["text"])
    for i in range(0, 6):
        svg.text(x0s + i * 10 * mm, y0s + 6, f"{i * 10}", size=2.2, fill=INK, anchor="middle")
    svg.text(x0s + 50 * mm + 3, y0s + 6, "M", size=2.2, fill=INK)
    svg.text(x0s, y0s - 3, f"1:{a.scale}", size=2.4, fill=INK, spacing=0.3)
    svg.text(Wp - gap_x, Hp - gap_y + 2, "LOCKPORT · THREE SITE MODELS · 1:500", size=3.0, fill=INK, anchor="end",
             spacing=0.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    svg.save(a.out)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--bed", default="data/lockport/bathymetry/bed_from_chart.tif")
    ap.add_argument("--overture", default="data/lockport/overture")
    ap.add_argument("--streamlines", default="data/lockport/derived/streamlines_8cms.json")
    ap.add_argument("--hourly", default="data/lockport/climate/en_climate_hourly_MB_*.csv")
    ap.add_argument("--sheet", type=float, nargs=2, default=[1828.8, 1117.6], help="mm (72 x 44 in)")
    ap.add_argument("--scale", type=int, default=500)
    ap.add_argument("--px", type=float, default=0.25, help="m per pixel of the stylised ortho")
    ap.add_argument("--ortho-opacity", type=float, default=0.9)
    ap.add_argument("--water-max", type=float, default=220.3, help="m: lidar ground below this is river")
    ap.add_argument("--grid", type=float, default=10.0, help="mm")
    ap.add_argument("--wind-density", type=float, default=0.06, help="candidate wind dots per mm2")
    ap.add_argument("--wind-streak", type=float, default=0.12, help="mm of streak per km/h")
    ap.add_argument("--z-top", type=float, default=236.0)
    ap.add_argument("--z-bottom", type=float, default=205.0)
    ap.add_argument("--model-base", type=float, default=218.0)
    ap.add_argument("--layer-m", type=float, default=1.5, help="m per card layer (3 mm at 1:500)")
    ap.add_argument("--locator-scale", type=int, default=5000)
    ap.add_argument("--out", default="outputs/sheets/11_model_print.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
