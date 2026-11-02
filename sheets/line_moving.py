"""Sheet 10 - The line moving: Lockport on the prairie-boreal line, and the
winter-cold lines moving north past it.

A plan of Manitoba and its neighbours, 42-61 N, 1:3,000,000 (Arch D portrait) (NRCan Lambert,
EPSG:3979, as the lidar):

  - ECOREGIONS (RESOLVE Ecoregions 2017): boundaries, with a light texture by
    biome (grass dots for prairie, upright ticks for boreal forest, open
    rings for broadleaf and mixed forest). The line between the grassland and
    boreal ecoregions is drawn heavy: it passes within a kilometre of
    Lockport. North of it, a hatched band: where that line would be if it
    moved 150-200 km north, the shift projected for Manitoba in a 2003 essay
    from GCMs (no year or scenario given) - an ESTIMATE, drawn light.
  - HARDINESS (the USDA measure: the average of each year's coldest night):
    at GHCN-Daily stations, 1961-1990 and 1991-2020, interpolated (a smooth
    regional trend plus inverse-distance residuals). 1961-1990 is built as
    1991-2020 minus the change measured at stations with records in both
    periods, so changes in the station network do not show up as warming.
    Zone boundaries every half zone are drawn thin, 1961-1990 dashed and
    1991-2020 solid.
  - THE LINE: the value Lockport had in 1961-1990, followed through time -
    1961-1990 (through Lockport), 1991-2020 (measured), and 2041-2070 under
    SSP2-4.5 and SSP5-8.5 and 2071-2100 under SSP5-8.5 (NASA NEX-GDDP-CMIP6,
    the mean change of three models added to 1991-2020). Each is labelled
    with how far north of Lockport it crosses Lockport's meridian.
  - WHERE LOCKPORT'S FUTURE WINTERS ARE TODAY: the 1991-2020 line at each
    projected Lockport value, south of it, as small ticks on Lockport's
    meridian.

    python -m sheets.line_moving
"""

import argparse
import csv
import glob
import json
import math
import os

import contourpy
import numpy as np
from pyproj import Transformer
from scipy import interpolate, ndimage
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.ops import transform as shp_transform, unary_union

from drawing.svg import PAPER, Svg, points_to_path, scale_bar, north_arrow

INK = "#111111"
LOCKPORT = (-96.93, 50.08)
ZONES = [("2a", -45.6), ("2b", -42.8), ("3a", -40.0), ("3b", -37.2), ("4a", -34.4), ("4b", -31.7), ("5a", -28.9),
         ("5b", -26.1), ("6a", -23.3)]
GRASS = "Temperate Grasslands, Savannas & Shrublands"
BOREAL = "Boreal Forests/Taiga"


def zone_of(t):
    name = "1b"
    for z, lo in ZONES:
        if t >= lo:
            name = z
    return name


# --------------------------------------------------------------------------- fields
def station_field(csv_path, X, Y, fwd, exclude):
    rows = list(csv.DictReader(open(csv_path)))
    st = []
    for r in rows:
        if any(e in r["name"] for e in exclude):
            continue
        x, y = fwd.transform(float(r["lon"]), float(r["lat"]))
        a = float(r["tmin_ext_1961_1990"]) if r["tmin_ext_1961_1990"] else None
        b = float(r["tmin_ext_1991_2020"]) if r["tmin_ext_1991_2020"] else None
        st.append((x, y, a, b, r["name"]))
    return st


def fit(points, X, Y, sigma=60e3, trim=3.0, w0=0.4):
    """A smooth quadratic trend in x, y, plus the stations' departures from it
    averaged with a Gaussian kernel (sigma m), fading back to the trend where
    stations are sparse. Stations departing more than `trim` deg C from the
    trend (urban sites, frost hollows, errors) are left out. Returns the grid
    and how many were left out."""
    P = np.array([(p[0], p[1]) for p in points])
    v = np.array([p[2] for p in points])

    def design(x, y):
        x, y = x / 1e6, y / 1e6
        return np.c_[np.ones(len(x)), x, y, x * x, y * y, x * y]

    A = design(P[:, 0], P[:, 1])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    keep = np.abs(v - A @ coef) < trim
    coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
    P, res = P[keep], (v - A @ coef)[keep]
    trend = (design(X.ravel(), Y.ravel()) @ coef).reshape(X.shape)
    num = np.zeros(X.size)
    den = np.zeros(X.size)
    xs, ys = X.ravel(), Y.ravel()
    for (px, py), r in zip(P, res):
        w = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
        num += w * r
        den += w
    return trend + (num / (den + w0)).reshape(X.shape), int((~keep).sum())


def cmip_delta(cmip_dir, run, X, Y, inv):
    """Mean change in annual extreme minimum, run minus historical, over the
    models that have both; bilinear onto the map grid."""
    g = np.load(os.path.join(cmip_dir, "grid.npz"))
    lat, lon = g["lat"], g["lon"]
    deltas, models = [], []
    for m in sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(cmip_dir, "*_historical_*.npy"))}):
        load = lambda f: np.where(np.abs(np.load(f)) < 1e3, np.load(f), np.nan)   # ocean cells hold fill values
        h = [load(f) for f in glob.glob(os.path.join(cmip_dir, f"{m}_historical_[0-9][0-9][0-9][0-9].npy"))]
        fu = [load(f) for f in glob.glob(os.path.join(cmip_dir, f"{m}_{run}_[0-9][0-9][0-9][0-9].npy"))]
        if len(h) < 3 or len(fu) < 3:
            continue
        deltas.append(np.mean(fu, 0) - np.mean(h, 0))
        models.append(f"{m} ({len(h)}/{len(fu)})")
    if not deltas:
        return None, []
    D = np.mean(deltas, 0)
    # Fill ocean cells from the nearest land cell, so interpolation near coasts stays on land values.
    bad = ~np.isfinite(D)
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        D = D[tuple(idx)]
    lon_g, lat_g = inv.transform(X, Y)
    f = interpolate.RegularGridInterpolator((lat, lon), D, bounds_error=False, fill_value=None)
    return f(np.c_[lat_g.ravel(), lon_g.ravel()]).reshape(X.shape), models


# --------------------------------------------------------------------------- sheet
def build(args):
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3979", always_xy=True)
    inv = Transformer.from_crs("EPSG:3979", "EPSG:4326", always_xy=True)
    lon0, lat0, lon1, lat1 = args.bbox
    corners = [fwd.transform(lo, la) for lo in np.linspace(lon0, lon1, 30) for la in (lat0, lat1)] + \
              [fwd.transform(lo, la) for lo in (lon0, lon1) for la in np.linspace(lat0, lat1, 30)]
    cx = np.array([c[0] for c in corners]); cy = np.array([c[1] for c in corners])
    # The map frame: the central part of the projected box.
    xa, xb = np.percentile(cx, 8), np.percentile(cx, 92)
    ya, yb = cy.min(), cy.max()
    s = 1000.0 / args.scale                              # mm per m
    Wp, Hp = PAPER[args.paper]
    fw, fh = (xb - xa) * s, (yb - ya) * s
    ox, oy = (Wp - fw) / 2, (Hp - fh) / 2

    def MX(x):
        return ox + (np.asarray(x) - xa) * s

    def MY(y):
        return oy + (yb - np.asarray(y)) * s

    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(eco_tex="ECOREGIONS · TEXTURE BY BIOME", eco="ECOREGIONS · BOUNDARIES · 0.08 mm",
             eco_line="PRAIRIE–BOREAL LINE · 0.4 mm", eco_est="PRAIRIE–BOREAL LINE +150–200 KM · EST. HATCH",
             water="LAKES · 0.1 mm", rivers="RIVERS · 0.15 mm", borders="BORDERS · 0.2 mm DASHED",
             zones_then="HARDINESS ZONES 1961–1990 · 0.08 mm DASHED", zones_now="HARDINESS ZONES 1991–2020 · 0.1 mm",
             line="THE LINE · LOCKPORT 1961–1990 VALUE THROUGH TIME", analog="LOCKPORT'S FUTURE WINTERS TODAY",
             places="PLACES", stations="STATIONS · 0.1 mm", frame="FRAME · 0.15 mm", scale="SCALE BAR", text="TEXT")
    for v in L.values():
        svg.layer(v)
    clip = svg.clip_rect(ox, oy, fw, fh)
    frame_geo = box(xa, ya, xb, yb)

    def proj(geom):
        return shp_transform(lambda x, y, z=None: fwd.transform(x, y), geom)

    def draw_geom(g, **kw):
        if g.is_empty:
            return
        if g.geom_type in ("Polygon", "MultiPolygon"):
            polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
            d = ""
            for p in polys:
                for ring in [p.exterior] + list(p.interiors):
                    xy = np.array(ring.coords)
                    d += points_to_path(MX(xy[:, 0]), MY(xy[:, 1])) + "Z"
            svg.path(d, **kw)
        elif g.geom_type in ("LineString", "MultiLineString", "GeometryCollection"):
            parts = [g] if g.geom_type == "LineString" else [q for q in g.geoms if q.geom_type == "LineString"]
            d = ""
            for q in parts:
                xy = np.array(q.coords)
                d += points_to_path(MX(xy[:, 0]), MY(xy[:, 1]))
            if d:
                svg.path(d, **kw)

    rng = np.random.default_rng(3)
    # ------------------------------------------------------------ ecoregions
    eco = json.load(open(args.ecoregions))["features"]
    groups = {}
    for f in eco:
        g = proj(shape(f["geometry"])).intersection(frame_geo)
        if g.is_empty:
            continue
        f["g"] = g
        groups.setdefault(f["properties"]["BIOME_NAME"], []).append(g)
    lakes = [proj(shape(f["geometry"])).buffer(0) for f in json.load(open(args.lakes))["features"]]
    lakes_u = unary_union([l for l in lakes if not l.is_empty]).intersection(frame_geo)
    svg.layer(L["eco_tex"])
    svg.open_group(clip_path=clip)
    for biome, gs in groups.items():
        u = unary_union(gs).difference(lakes_u)
        minx, miny, maxx, maxy = u.bounds
        n = int((maxx - minx) * (maxy - miny) / (args.tex_spacing ** 2))
        px = rng.uniform(minx, maxx, n); py = rng.uniform(miny, maxy, n)
        from shapely import contains_xy
        inside = contains_xy(u, px, py)
        X, Y = MX(px[inside]), MY(py[inside])
        if biome == GRASS:
            d = "".join(f"M{a:.2f},{b:.2f}h0" for a, b in zip(X, Y))
            svg.path(d, fill="none", stroke=INK, stroke_width=0.22, stroke_linecap="round", opacity=0.55)
        elif biome == BOREAL:
            d = "".join(f"M{a:.2f},{b:.2f}v-1.1" for a, b in zip(X, Y))
            svg.path(d, fill="none", stroke=INK, stroke_width=0.1, opacity=0.6)
        else:
            for a, b in zip(X[::2], Y[::2]):
                svg.circle(a, b, 0.4, fill="none", stroke=INK, stroke_width=0.07, opacity=0.6)
    svg.close_group()
    svg.layer(L["eco"])
    svg.open_group(clip_path=clip)
    for f in eco:
        if "g" in f:
            draw_geom(f["g"], fill="none", stroke=INK, stroke_width=0.08, opacity=0.7)
    svg.close_group()
    # The prairie-boreal line: where grassland and boreal ecoregions meet (lakes excepted).
    grass_u = unary_union(groups.get(GRASS, []))
    boreal_u = unary_union(groups.get(BOREAL, []))
    pb = grass_u.buffer(500).intersection(boreal_u.buffer(500)).intersection(frame_geo)
    pb_line = boreal_u.boundary.intersection(pb).difference(lakes_u.buffer(1500))
    svg.layer(L["eco_line"])
    svg.open_group(clip_path=clip)
    draw_geom(pb_line, fill="none", stroke=INK, stroke_width=0.4, stroke_linejoin="round")
    svg.close_group()
    # Estimated shift: the same line moved 150-200 km north, as a hatched band.
    from shapely import affinity
    band = unary_union([affinity.translate(pb_line, 0, dy).buffer(1) for dy in np.arange(150e3, 200e3 + 1, 2.5e3)])
    svg.layer(L["eco_est"])
    svg.open_group(clip_path=clip)
    for dy in np.arange(150e3, 200e3 + 1, 5e3):
        draw_geom(affinity.translate(pb_line, 0, dy), fill="none", stroke=INK, stroke_width=0.05,
                  stroke_dasharray="0.6 0.9", opacity=0.55)
    for dy in (150e3, 200e3):
        draw_geom(affinity.translate(pb_line, 0, dy), fill="none", stroke=INK, stroke_width=0.12,
                  stroke_dasharray="1.6 1.0", opacity=0.7)
    svg.close_group()

    # ------------------------------------------------------------ water, borders
    svg.layer(L["water"])
    svg.open_group(clip_path=clip)
    draw_geom(lakes_u, fill="#ffffff", stroke=INK, stroke_width=0.1)
    svg.close_group()
    svg.layer(L["rivers"])
    svg.open_group(clip_path=clip)
    for f in json.load(open(args.rivers))["features"]:
        g = proj(shape(f["geometry"])).intersection(frame_geo).difference(lakes_u)
        sr = f["properties"].get("scalerank") or 9
        draw_geom(g, fill="none", stroke=INK, stroke_width=0.18 if sr <= 4 else 0.1, opacity=0.9)
    svg.close_group()
    svg.layer(L["borders"])
    svg.open_group(clip_path=clip)
    provs = [proj(shape(f["geometry"])).buffer(0) for f in json.load(open(args.provinces))["features"]]
    edges = []
    for i in range(len(provs)):
        for j in range(i + 1, len(provs)):
            e = provs[i].boundary.intersection(provs[j].buffer(800))
            if not e.is_empty:
                edges.append(e)
    draw_geom(unary_union(edges).intersection(frame_geo).difference(lakes_u), fill="none", stroke=INK,
              stroke_width=0.2, stroke_dasharray="2.0 0.8 0.4 0.8", opacity=0.8)
    svg.close_group()

    # ------------------------------------------------------------ hardiness fields
    cell = args.cell
    pad = 400e3
    gx = np.arange(xa - pad, xb + pad + cell, cell)
    gy = np.arange(ya - pad, yb + pad + cell, cell)
    X, Y = np.meshgrid(gx, gy)
    st = station_field(args.stations, X, Y, fwd, args.exclude)
    now_pts = [(x, y, b) for x, y, a, b, n in st if b is not None]
    diff_pts = [(x, y, b - a) for x, y, a, b, n in st if a is not None and b is not None]
    F_now, drop1 = fit(now_pts, X, Y)
    D_obs, drop2 = fit(diff_pts, X, Y, sigma=120e3, trim=1.5)
    F_then = F_now - D_obs
    lx, ly = fwd.transform(*LOCKPORT)
    fi = lambda F: float(interpolate.RegularGridInterpolator((gy, gx), F)((ly, lx)))
    v_then, v_now = fi(F_then), fi(F_now)
    print(f"stations: {len(now_pts)} for 1991-2020 ({drop1} dropped as outliers), {len(diff_pts)} with both periods "
          f"({drop2} dropped)")
    print(f"Lockport mean annual extreme minimum: 1961-1990 {v_then:.1f} C ({zone_of(v_then)}), "
          f"1991-2020 {v_now:.1f} C ({zone_of(v_now)})")
    futures = []
    for run, label, style in (("ssp245", "2041–2070 · SSP2-4.5", dict(stroke_width=0.35, stroke_dasharray="2.4 1.0")),
                              ("ssp585", "2041–2070 · SSP5-8.5", dict(stroke_width=0.35, stroke_dasharray="0.8 0.8")),
                              ("ssp585_late", "2071–2100 · SSP5-8.5",
                               dict(stroke_width=0.35, stroke_dasharray="0.3 0.9", stroke_linecap="round"))):
        D, models = cmip_delta(args.cmip, run, X, Y, inv)
        if D is None:
            print("no model data for", run)
            continue
        F = F_now + D
        futures.append((label, F, style, models))
        print(f"{label}: Lockport {fi(F):.1f} C ({zone_of(fi(F))}); models {', '.join(models)}")

    def iso(F, level):
        lines = contourpy.contour_generator(gx, gy, F).lines(level)
        return [ln for ln in lines if len(ln) > 3]

    def north_of_lockport(F, level):
        """km north (or south, negative) of Lockport where the line crosses its meridian."""
        lons = np.full(800, LOCKPORT[0]); lats = np.linspace(40.5, 62.5, 800)
        xs, ys = fwd.transform(lons, lats)
        f = interpolate.RegularGridInterpolator((gy, gx), F, bounds_error=False, fill_value=np.nan)
        v = f(np.c_[ys, xs])
        ok = np.isfinite(v)
        lats, v = lats[ok], v[ok]
        k = np.nonzero(np.diff(np.sign(v - level)))[0]
        if not len(k):
            return None, None
        j = k[np.argmin(np.abs(lats[k] - LOCKPORT[1]))]
        lat_c = np.interp(level, [v[j], v[j + 1]] if v[j] < v[j + 1] else [v[j + 1], v[j]],
                          [lats[j], lats[j + 1]] if v[j] < v[j + 1] else [lats[j + 1], lats[j]])
        return (lat_c - LOCKPORT[1]) * 111.2, lat_c

    # Zone boundaries: then dashed, now solid, every half zone.
    for layer, F, kw in ((L["zones_then"], F_then, dict(stroke_width=0.08, stroke_dasharray="1.0 0.8", opacity=0.75)),
                         (L["zones_now"], F_now, dict(stroke_width=0.1, opacity=0.85))):
        svg.layer(layer)
        svg.open_group(clip_path=clip)
        for z, lo in ZONES:
            for ln in iso(F, lo):
                svg.path(points_to_path(MX(ln[:, 0]), MY(ln[:, 1])), fill="none", stroke=INK, **kw)
        svg.close_group()
    # Zone names at the left edge of the frame, between 1991-2020 boundaries.
    svg.layer(L["text"])
    for z, lo in ZONES:
        hi = lo + 2.8
        col = F_now[:, int(np.searchsorted(gx, xa + 8e3))]
        rows = np.nonzero((col >= lo) & (col < hi))[0]
        if len(rows) > 3:
            r = rows[len(rows) // 2]
            if ya < gy[r] < yb:
                svg.text(ox + 2.0, float(MY(gy[r])) + 0.7, z.upper(), size=2.0, fill=INK, spacing=0.2)

    # The line.
    svg.layer(L["line"])
    svg.open_group(clip_path=clip)
    series = [("1961–1990", F_then, dict(stroke_width=0.5)), ("1991–2020", F_now, dict(stroke_width=0.5,
                                                                                      stroke_dasharray="4 0.8"))]
    series += [(lab, F, sty) for lab, F, sty, _ in futures]
    marks = []
    for lab, F, sty in series:
        for ln in iso(F, v_then):
            svg.path(points_to_path(MX(ln[:, 0]), MY(ln[:, 1])), fill="none", stroke=INK, stroke_linejoin="round", **sty)
        km, lat_c = north_of_lockport(F, v_then)
        marks.append((lab, km, lat_c))
    svg.close_group()
    # Labels where each line crosses Lockport's meridian.
    svg.layer(L["text"])
    for lab, km, lat_c in marks:
        if lat_c is None:
            continue
        x, y = fwd.transform(LOCKPORT[0], lat_c)
        if not (ya < y < yb):
            continue
        t = lab if abs(km) < 5 else f"{lab} · {km:+.0f} KM"
        svg.circle(float(MX(x)), float(MY(y)), 0.6, fill=INK, stroke="none")
        svg.text(float(MX(x)) + 2.0, float(MY(y)) - 1.0, t, size=1.9, fill=INK, spacing=0.15)
    # Where Lockport's future winters are today: the 1991-2020 line at each projected value.
    svg.layer(L["analog"])
    svg.open_group(clip_path=clip)
    for lab, F, sty, _ in futures:
        val = fi(F)
        km, lat_c = north_of_lockport(F_now, val)
        if lat_c is None:
            print(f"{lab}: today's {val:.1f} C line is off the map")
            continue
        x, y = fwd.transform(LOCKPORT[0], lat_c)
        print(f"{lab}: Lockport's winters are like today's {km:+.0f} km along its meridian ({lat_c:.2f} N)")
        if ya < y < yb:
            for ln in iso(F_now, val):
                svg.path(points_to_path(MX(ln[:, 0]), MY(ln[:, 1])), fill="none", stroke=INK, stroke_width=0.12,
                         opacity=0.6, **{k: v for k, v in sty.items() if k != "stroke_width"})
            svg.layer(L["text"])
            svg.text(float(MX(x)) - 2.0, float(MY(y)) + 2.6, f"LOCKPORT {lab} {val:.1f} °C, TODAY HERE · {km:.0f} KM",
                     size=1.6, fill=INK, anchor="end", spacing=0.12, opacity=0.85)
            svg.layer(L["analog"])
    svg.layer(L["analog"])
    svg.close_group()

    # ------------------------------------------------------------ places, stations, frame
    svg.layer(L["stations"])
    for x, y, a, b, n in st:
        if xa < x < xb and ya < y < yb and b is not None:
            svg.circle(float(MX(x)), float(MY(y)), 0.35, fill="none", stroke=INK, stroke_width=0.08, opacity=0.6)
    svg.layer(L["places"])
    places = json.load(open(args.places))["features"]
    for f in places:
        lo, la = f["geometry"]["coordinates"]
        x, y = fwd.transform(lo, la)
        big = (f["properties"]["pop_max"] or 0) >= args.min_pop or f["properties"]["name"] in args.towns
        if not (xa < x < xb and ya < y < yb) or not big:
            continue
        svg.circle(float(MX(x)), float(MY(y)), 0.5, fill=INK, stroke="none")
        svg.layer(L["text"])
        svg.text(float(MX(x)) + 1.4, float(MY(y)) + 0.7, f["properties"]["name"].upper(), size=1.7, fill=INK,
                 spacing=0.15)
        svg.layer(L["places"])
    svg.circle(float(MX(lx)), float(MY(ly)), 1.4, fill="#ffffff", stroke=INK, stroke_width=0.35)
    svg.circle(float(MX(lx)), float(MY(ly)), 0.5, fill=INK, stroke="none")
    svg.layer(L["text"])
    svg.text(float(MX(lx)) - 2.2, float(MY(ly)) + 0.8, "LOCKPORT", size=2.3, fill=INK, anchor="end", spacing=0.25,
             weight="bold")
    for name, lo, la in (("LAKE WINNIPEG", -97.3, 52.4), ("LAKE MANITOBA", -98.75, 51.0),
                         ("LAKE WINNIPEGOSIS", -100.1, 52.55), ("LAKE OF THE WOODS", -94.9, 49.35)):
        x, y = fwd.transform(lo, la)
        if xa < x < xb and ya < y < yb:
            svg.text(float(MX(x)), float(MY(y)), name, size=1.8, fill=INK, anchor="middle", spacing=0.4,
                     font_style="italic", opacity=0.8)
    for f in eco:
        if "g" not in f:
            continue
        g = f["g"]
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        if g.area < 1.5e10:
            continue
        pt = g.representative_point()
        words = f["properties"]["ECO_NAME"].upper().split()
        half = (len(words) + 1) // 2
        for j, t in enumerate((" ".join(words[:half]), " ".join(words[half:]))):
            if t:
                svg.text(float(MX(pt.x)), float(MY(pt.y)) + j * 2.6, t, size=1.8, fill=INK, anchor="middle",
                         spacing=0.35, opacity=0.9)
    svg.layer(L["frame"])
    svg.rect(ox, oy, fw, fh, fill="none", stroke=INK, stroke_width=0.15)
    svg.layer(L["scale"])
    scale_bar(svg, ox, oy + fh + 6, 200e3 * s, 200, "km", color=INK)
    north_arrow(svg, ox + fw - 6, oy + fh + 8, r=3.0, color=INK)
    svg.layer(L["text"])
    svg.text(ox, oy - 4, f"THE LINE MOVING · 1:{args.scale:,}", size=2.2, fill=INK, spacing=0.3)

    annot(20, 20, "THE LINE MOVING", size=5.2, fill=INK, spacing=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=[-107.0, 42.4, -85.5, 60.6], metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--scale", type=int, default=3_000_000)
    ap.add_argument("--cell", type=float, default=8000.0, help="m, interpolation grid")
    ap.add_argument("--tex-spacing", type=float, default=13000.0, help="m between biome texture marks")
    ap.add_argument("--min-pop", type=int, default=90000, help="places labelled above this population")
    ap.add_argument("--towns", nargs="*", default=["Selkirk", "Thompson", "The Pas", "Brandon", "Kenora", "Dauphin",
                                                  "Flin Flon", "Grand Forks", "Fargo"])
    ap.add_argument("--stations", default="data/region/ghcn/station_extreme_min.csv")
    ap.add_argument("--exclude", nargs="*", default=["THE FORKS"], help="station names left out (urban)")
    ap.add_argument("--cmip", default="data/region/cmip")
    ap.add_argument("--ecoregions", default="data/region/ecoregions_resolve2017_mb.geojson")
    ap.add_argument("--lakes", default="data/region/ne_lakes_na.geojson")
    ap.add_argument("--rivers", default="data/region/ne_rivers_na.geojson")
    ap.add_argument("--provinces", default="data/region/ne_provinces.geojson")
    ap.add_argument("--places", default="data/region/ne_places.geojson")
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD_P")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="outputs/sheets/10_line_moving.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
