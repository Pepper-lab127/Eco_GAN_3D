"""Sheet 11 - Lockport at 1:500: one continuous map with the three site models
drawn in place, in their surrounding context.

The map is at the models' own scale (1:500), 740 m east-west x 1000 m
north-south of ground (1440 x 2000 mm) on a 60 x 84 in sheet:

  - a stylised orthographic image made from the 1 m lidar (pale ground relief,
    tree crowns as stipple, pale water); aerial imagery is not reachable here
  - elevation lines at the models' levels (every 1.5 m from 218 m): light
    across the context, and inside each model rectangle the model's own layer
    outlines (from its DXF), in black
  - roads, buildings and structures (Overture Maps), the 1997 crest flood
    streamlines (each 8 m3/s, as sheet 01) and a year of hourly wind at 2 m
    (as sheet 04b)
  - the three model rectangles, outlined and labelled; the context outside them
    is faded slightly
  - north arrow (a circle with a line to grid north) and a scale bar at bottom right

The geology strips for the model sides are on their own sheet
(tools/geology_strips.py).

    python -m sheets.site_print [--px 0.25]
"""

import argparse
import base64
import io
import json
import math
import os

import contourpy
import ezdxf
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

from drawing import climate, wind
from drawing.svg import Svg, points_to_path
from sheets.model_print import ROAD_W, hillshade, ortho, png_uri

INK = "#111111"
MODELS = [
    ("A", "THE BRIDGE ON ST. ANDREWS DAM", (2030, 2250, 1245, 1645), "outputs/site_models/A_bridge_dam"),
    ("B", "THE FLOODWAY OUTLET", (1300, 1520, 1490, 1890), "outputs/site_models/B_floodway_outlet"),
    ("C", "OLD HENDERSON HIGHWAY", (1705, 1925, 1560, 1960), "outputs/site_models/C_old_henderson"),
]


def build(a):
    rng = np.random.default_rng(12)
    Wp, Hp = a.sheet
    svg = Svg(Wp, Hp)
    L = dict(grid="GRID · 0.05–0.1 mm", ortho="ORTHO · STYLISED FROM LIDAR (RASTER)", fade="CONTEXT FADE · WHITE 50%",
             ctx="ELEVATION LINES · CONTEXT, EVERY 1.5 M · 0.06 mm", water="WATER EDGE · 0.15 mm",
             wind="WIND · 2024 HOURS AT 2 M", flood="FLOOD · 1997 CREST STREAMLINES · 0.07 mm",
             model="ELEVATION LINES · MODEL LAYERS · 0.1–0.18 mm", roads="ROADS · WHITE, 0.08 mm EDGES",
             blds="BUILDINGS · FILLED", struct="STRUCTURES · 0.3–0.6 mm", rects="MODEL RECTANGLES · 1.0 mm",
             frame="FRAME · 0.2 mm", north="NORTH ARROW", scale="SCALE BAR", text="TEXT")
    for v in L.values():
        svg.layer(v)
    mm = 1000.0 / a.scale
    r0, r1, c0, c1 = a.window
    W, H = (c1 - c0) * mm, (r1 - r0) * mm
    ox, oy = (Wp - W) / 2, a.top
    print(f"map {W:.0f} x {H:.0f} mm on a {Wp:.0f} x {Hp:.0f} mm sheet")

    # Grid over the whole sheet.
    svg.layer(L["grid"])
    dmi, dma = [], []
    for x in np.arange(0, Wp + 0.1, a.grid):
        (dma if round(x / a.grid) % 5 == 0 else dmi).append(f"M{x:.1f},0V{Hp:.1f}")
    for y in np.arange(0, Hp + 0.1, a.grid):
        (dma if round(y / a.grid) % 5 == 0 else dmi).append(f"M0,{y:.1f}H{Wp:.1f}")
    svg.path("".join(dmi), fill="none", stroke="#bdbdbd", stroke_width=0.05, opacity=0.6)
    svg.path("".join(dma), fill="none", stroke="#9a9a9a", stroke_width=0.1, opacity=0.6)

    src = rasterio.open(a.dtm)
    dtm = src.read(1).astype(float)
    dtm[dtm < -1000] = np.nan
    dsm = rasterio.open(a.dsm).read(1).astype(float)
    dsm[dsm < -1000] = np.nan
    left, top = src.transform.c, src.transform.f
    fwd = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    clip = svg.clip_rect(ox, oy, W, H)
    frame = box(ox, oy, ox + W, oy + H)

    def to_map(g):
        return shp_transform(lambda lon, lat, z=None: (
            ox + (np.asarray(fwd.transform(lon, lat)[0]) - left - c0) * mm,
            oy + (top - np.asarray(fwd.transform(lon, lat)[1]) - r0) * mm), g)

    def rect_of(win):
        a0, a1, b0, b1 = win
        return ox + (b0 - c0) * mm, oy + (a0 - r0) * mm, (b1 - b0) * mm, (a1 - a0) * mm

    # 1. Ortho, then a white veil over the context outside the model rectangles.
    img, water = ortho(dtm, dsm, (r0, r1, c0, c1), a.px, a.water_max, rng)
    svg.layer(L["ortho"])
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=a.jpeg_q)
    svg.image(ox, oy, W, H, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), opacity=0.92)
    svg.layer(L["fade"])
    d = f"M{ox:.2f},{oy:.2f}h{W:.2f}v{H:.2f}h{-W:.2f}Z"
    for _, _, win, _ in MODELS:
        x, y, w, h = rect_of(win)
        d += f"M{x:.2f},{y:.2f}v{h:.2f}h{w:.2f}v{-h:.2f}Z"
    svg.path(d, fill="#ffffff", stroke="none", fill_rule="evenodd", opacity=a.fade)

    # 2. Context elevation lines at the model levels; water edge.
    zs = ndimage.gaussian_filter(np.nan_to_num(dtm[r0:r1, c0:c1], nan=np.nanmedian(dtm[r0:r1, c0:c1])), 2.0)
    gen = contourpy.contour_generator(z=zs)
    svg.layer(L["ctx"])
    svg.open_group(clip_path=clip)
    for lv in np.arange(a.model_base, np.nanmax(zs), a.layer_m):
        dd = [points_to_path(ox + (ln[:, 0] + 0.5) * mm, oy + (ln[:, 1] + 0.5) * mm) for ln in gen.lines(lv)
              if len(ln) > 12]
        svg.path("".join(dd), fill="none", stroke="#555555", stroke_width=0.06, opacity=0.8)
    svg.close_group()
    svg.layer(L["water"])
    svg.open_group(clip_path=clip)
    wl = ndimage.binary_opening(zs < a.water_max, iterations=2)
    for ln in contourpy.contour_generator(z=ndimage.gaussian_filter(wl.astype(float), 1.5)).lines(0.5):
        if len(ln) > 20:
            svg.path(points_to_path(ox + (ln[:, 0] + 0.5) * mm, oy + (ln[:, 1] + 0.5) * mm), fill="none", stroke=INK,
                     stroke_width=0.15)
    svg.close_group()

    # 3. Wind over the whole map.
    hourly, _ = climate.load_eccc_hourly_wind(a.hourly)
    st = a.wind_cell
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
    X, Y, T, LN = ox + pxx * st * mm, oy + py * st * mm, tw, loc * a.wind_streak
    svg.layer(L["wind"])
    svg.open_group(clip_path=clip)
    svg.path("".join(f"M{x:.2f},{y:.2f}h0" for x, y in zip(X[keep], Y[keep])), fill="none", stroke=INK,
             stroke_width=0.3, stroke_linecap="round", opacity=0.55)
    svg.path("".join(f"M{x:.2f},{y:.2f}l{math.sin(t) * l:.2f},{-math.cos(t) * l:.2f}"
                     for x, y, t, l in zip(X[keep], Y[keep], T[keep], LN[keep])),
             fill="none", stroke=INK, stroke_width=0.08, opacity=0.5)
    svg.close_group()
    print(f"{int(keep.sum())} wind dots")

    # 4. Flood streamlines.
    lines97 = json.load(open(a.streamlines)).get("1997-05-04", [])
    svg.layer(L["flood"])
    svg.open_group(clip_path=clip)
    svg.path("".join(points_to_path(ox + (np.asarray(cc) - c0) * mm, oy + (np.asarray(rr) - r0) * mm)
                     for rr, cc in lines97), fill="none", stroke=INK, stroke_width=0.07, opacity=0.6,
             stroke_linejoin="round")
    svg.close_group()

    # 5. Model elevation lines inside each rectangle (the models' own DXFs).
    svg.layer(L["model"])
    for key, title, win, mdir in MODELS:
        x, y, w, h = rect_of(win)
        dxf = ezdxf.readfile(os.path.join(mdir, "lockport_contour_layers.dxf"))
        dd5, dd = [], []
        for e in dxf.modelspace():
            if e.dxf.layer == "MODEL_EDGE":
                continue
            pts = np.array([(p[0], p[1]) for p in e.get_points()])
            path = points_to_path(x + pts[:, 0], y + h - pts[:, 1]) + "Z"
            (dd5 if int(e.dxf.layer[1:3]) % 2 == 0 else dd).append(path)
        svg.path("".join(dd5), fill="none", stroke=INK, stroke_width=0.18)
        svg.path("".join(dd), fill="none", stroke=INK, stroke_width=0.1)

    # 6. Roads, buildings, structures.
    roads = json.load(open(os.path.join(a.overture, "roads.geojson")))["features"]
    rp = [to_map(shape(f["geometry"])).buffer(ROAD_W.get(f["properties"].get("class") or "", 4.0) * mm / 2,
                                               cap_style="flat") for f in roads]
    rp = unary_union(rp).intersection(frame)
    svg.layer(L["roads"])
    for q in getattr(rp, "geoms", [rp]):
        if q.geom_type == "Polygon":
            dd = points_to_path(*np.asarray(q.exterior.coords).T) + "Z"
            for hole in q.interiors:
                dd += points_to_path(*np.asarray(hole.coords).T) + "Z"
            svg.path(dd, fill="#ffffff", stroke=INK, stroke_width=0.08, fill_rule="evenodd", opacity=0.95)
    svg.layer(L["blds"])
    for f in json.load(open(os.path.join(a.overture, "buildings.geojson")))["features"]:
        g = to_map(shape(f["geometry"]))
        if g.geom_type == "Polygon" and g.intersects(frame):
            g = g.intersection(frame)
            if g.geom_type == "Polygon":
                svg.path(points_to_path(*np.asarray(g.exterior.coords).T) + "Z", fill="#3d3d3d", stroke=INK,
                         stroke_width=0.12)
    infra = json.load(open(os.path.join(a.overture, "infrastructure.geojson")))["features"]
    bridges = [f for f in infra if f["properties"].get("subtype") == "bridge"]
    longest = max(bridges, key=lambda f: shape(f["geometry"]).length) if bridges else None
    svg.layer(L["struct"])
    svg.open_group(clip_path=clip)
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

    # 7. Model rectangles and labels.
    for key, title, win, _ in MODELS:
        x, y, w, h = rect_of(win)
        svg.layer(L["rects"])
        svg.rect(x, y, w, h, fill="none", stroke=INK, stroke_width=1.0)
        tick = 8.0
        for cx, cy, sx, sy in ((x, y, -1, -1), (x + w, y, 1, -1), (x, y + h, -1, 1), (x + w, y + h, 1, 1)):
            svg.line(cx, cy, cx + sx * tick, cy, stroke=INK, stroke_width=0.25)
            svg.line(cx, cy, cx, cy + sy * tick, stroke=INK, stroke_width=0.25)
        svg.layer(L["text"])
        svg.rect(x, y - 9.5, 4.5 + len(title) * 2.0 + 60, 7.5, fill="#ffffff", stroke="none", opacity=0.85)
        svg.text(x + 2, y - 3.6, f"{key} · {title} · MODEL {w:.0f} × {h:.0f} MM", size=3.2, fill=INK, spacing=0.4)

    svg.layer(L["frame"])
    svg.rect(ox, oy, W, H, fill="none", stroke=INK, stroke_width=0.2)

    # 8. North arrow and scale bar, bottom right below the map; title bottom left.
    base_y = oy + H + (Hp - oy - H) / 2
    xr = ox + W
    svg.layer(L["scale"])
    x0s = xr - 100 * mm
    for i in range(10):
        svg.rect(x0s + i * 10 * mm, base_y, 10 * mm, 2.4, fill=INK if i % 2 == 0 else "#ffffff", stroke=INK,
                 stroke_width=0.15)
    svg.layer(L["text"])
    for i in range(0, 11, 2):
        svg.text(x0s + i * 10 * mm, base_y + 7, f"{i * 10}", size=2.6, fill=INK, anchor="middle")
    svg.text(xr + 3, base_y + 7, "M", size=2.6, fill=INK)
    svg.text(x0s, base_y - 3.5, f"1:{a.scale}", size=3.0, fill=INK, spacing=0.4)
    cx, cy, rr = x0s - 30, base_y + 1, 10.0
    svg.layer(L["north"])
    svg.circle(cx, cy, rr, fill="none", stroke=INK, stroke_width=0.35)
    svg.line(cx, cy + rr * 0.8, cx, cy - rr * 1.6, stroke=INK, stroke_width=0.5)
    svg.layer(L["text"])
    svg.text(cx, cy - rr * 1.6 - 2.5, "N", size=3.4, fill=INK, anchor="middle")
    svg.text(ox, base_y + 2, "LOCKPORT · THE THREE SITE MODELS IN PLACE · 1:500", size=4.4, fill=INK, spacing=0.5)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    svg.save(a.out)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--overture", default="data/lockport/overture")
    ap.add_argument("--streamlines", default="data/lockport/derived/streamlines_8cms.json")
    ap.add_argument("--hourly", default="data/lockport/climate/en_climate_hourly_MB_*.csv")
    ap.add_argument("--window", type=int, nargs=4, default=[1270, 2270, 1235, 1975],
                    help="DEM rows/cols: 1000 m north-south x 740 m east-west")
    ap.add_argument("--sheet", type=float, nargs=2, default=[1524.0, 2133.6], help="mm (60 x 84 in, portrait)")
    ap.add_argument("--top", type=float, default=42.0, help="mm above the map")
    ap.add_argument("--scale", type=int, default=500)
    ap.add_argument("--px", type=float, default=0.25, help="m per pixel of the stylised ortho")
    ap.add_argument("--jpeg-q", type=int, default=90)
    ap.add_argument("--fade", type=float, default=0.5, help="white veil over the context outside the models")
    ap.add_argument("--water-max", type=float, default=220.3)
    ap.add_argument("--grid", type=float, default=10.0)
    ap.add_argument("--wind-cell", type=int, default=4)
    ap.add_argument("--wind-density", type=float, default=0.05)
    ap.add_argument("--wind-streak", type=float, default=0.12)
    ap.add_argument("--model-base", type=float, default=218.0)
    ap.add_argument("--layer-m", type=float, default=1.5)
    ap.add_argument("--out", default="outputs/sheets/11_site_print.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
