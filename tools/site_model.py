"""A 1:500 physical site model of Lockport: cardboard terrain layers for the
laser cutter, wood parts (dam, bridge, buildings), and a 3D model.

  - TERRAIN: the 1 m lidar ground (DTM), lightly smoothed, cut into layers one
    card thick. At 1:500, 2 mm card = 1 m, so the layers are true-scale 1 m
    contours. Layer k is all ground at or above its level; slivers thinner
    than about 3 mm and specks under about 9 x 9 mm are removed so every piece
    can be cut and glued.
  - LASER FILES: each layer is split into tiles that fit the bed. Every piece
    is cut (red), and engraved (blue) with the outline of the layer that sits
    on it (for gluing), the roads on the part of it that stays exposed, building
    footprints (to place the wood blocks), and its label. Pieces are packed
    onto sheets; an assembly map shows every layer with its piece labels.
  - WOOD: building footprints with heights (Overture, else measured from the
    lidar surface model), the PTH 44 bridge deck over the dam, and the dam's
    curtain wall under it. The lidar was flown with the dam open and its bridge
    filtered out, so the deck height is taken from the road at each bank and
    the structure is drawn schematically.
  - 3D: an OBJ in model millimetres (terraced layers, wood parts, roads as
    lines), for Rhino / SketchUp / Blender.

Roads and buildings: Overture Maps (includes OpenStreetMap), fetched with
tools/fetch_overture.py.

    python -m tools.site_model
"""

import argparse
import json
import math
import os

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import features
from scipy import ndimage
from shapely import constrained_delaunay_triangles
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Polygon, box, shape
from shapely.ops import polylabel, transform as shp_transform, unary_union

CUT, ENGRAVE = "#FF0000", "#0000FF"
ROAD_W = {"primary": 12.0, "secondary": 9.0, "tertiary": 8.0, "residential": 7.0, "unclassified": 6.0,
          "service": 4.5, "track": 3.0, "path": 2.0, "footway": 2.0, "steps": 2.0, "unknown": 5.0}


# --------------------------------------------------------------------------- stroke font
_G = {  # single-stroke glyphs on a 4 x 6 grid (x right, y down), for engraving labels
    "0": [[(0, 0), (4, 0), (4, 6), (0, 6), (0, 0)], [(0, 6), (4, 0)]],
    "1": [[(1, 1), (2, 0), (2, 6)], [(1, 6), (3, 6)]],
    "2": [[(0, 0), (4, 0), (4, 3), (0, 3), (0, 6), (4, 6)]],
    "3": [[(0, 0), (4, 0), (4, 6), (0, 6)], [(1, 3), (4, 3)]],
    "4": [[(0, 0), (0, 3), (4, 3)], [(4, 0), (4, 6)]],
    "5": [[(4, 0), (0, 0), (0, 3), (4, 3), (4, 6), (0, 6)]],
    "6": [[(4, 0), (0, 0), (0, 6), (4, 6), (4, 3), (0, 3)]],
    "7": [[(0, 0), (4, 0), (2, 6)]],
    "8": [[(0, 0), (4, 0), (4, 6), (0, 6), (0, 0)], [(0, 3), (4, 3)]],
    "9": [[(4, 3), (0, 3), (0, 0), (4, 0), (4, 6), (0, 6)]],
    "L": [[(0, 0), (0, 6), (4, 6)]],
    "T": [[(0, 0), (4, 0)], [(2, 0), (2, 6)]],
    "P": [[(0, 6), (0, 0), (4, 0), (4, 3), (0, 3)]],
    "B": [[(0, 0), (3, 0), (4, 1.5), (3, 3), (0, 3)], [(3, 3), (4, 4.5), (3, 6), (0, 6), (0, 0)]],
    "W": [[(0, 0), (1, 6), (2, 3), (3, 6), (4, 0)]],
    "D": [[(0, 0), (3, 0), (4, 1.5), (4, 4.5), (3, 6), (0, 6), (0, 0)]],
    "S": [[(4, 0), (0, 0), (0, 3), (4, 3), (4, 6), (0, 6)]],
    "-": [[(1, 3), (3, 3)]],
    " ": [],
}


def stroke_text(s, x, y, h):
    """Engraving paths for text s, top-left at (x, y), cap height h (mm)."""
    k = h / 6.0
    d = []
    for i, ch in enumerate(s.upper()):
        for stroke in _G.get(ch, []):
            pts = [(x + (i * 5.5 + px) * k, y + py * k) for px, py in stroke]
            d.append("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in pts))
    return "".join(d), len(s) * 5.5 * k


# --------------------------------------------------------------------------- geometry helpers
def clean(g, open_r, min_area):
    """Remove slivers narrower than 2*open_r and parts / holes smaller than min_area."""
    g = g.buffer(-open_r, join_style="round").buffer(open_r, join_style="round")
    polys = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= min_area]
    out = []
    for p in polys:
        holes = [h for h in p.interiors if Polygon(h).area >= min_area]
        out.append(Polygon(p.exterior, holes))
    return unary_union(out) if out else Polygon()


def parts(g):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and not p.is_empty]


def path_d(g):
    d = []
    for p in parts(g):
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords)
            d.append("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in xy) + "Z")
    return "".join(d)


def line_d(g):
    d = []
    for q in getattr(g, "geoms", [g]):
        if q.geom_type == "LineString" and len(q.coords) > 1:
            xy = np.asarray(q.coords)
            d.append("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in xy))
        elif q.geom_type in ("Polygon", "MultiPolygon", "MultiLineString", "GeometryCollection"):
            d.append(line_d(q.boundary if q.geom_type.endswith("Polygon") else q))
    return "".join(d)


def write_dxf(path, body, w, h):
    """The same cut / engrave paths as a DXF (mm): layer CUT (red) and ENGRAVE (blue), y up."""
    import re
    import ezdxf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = ezdxf.new(units=4)
    doc.layers.add("CUT", color=1)
    doc.layers.add("ENGRAVE", color=5)
    msp = doc.modelspace()
    for m in re.finditer(r'<path d="([^"]*)"[^>]*stroke="(#[0-9A-F]{6})"', body):
        layer = "CUT" if m.group(2) == CUT else "ENGRAVE"
        for sub in re.findall(r"M[^M]+", m.group(1)):
            closed = sub.endswith("Z")
            nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", sub)]
            pts = [(nums[i], h - nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            if len(pts) > 1:
                msp.add_lwpolyline(pts, close=closed, dxfattribs={"layer": layer})
    doc.saveas(path)


def svg_doc(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}mm" height="{h:.2f}mm" '
            f'viewBox="0 0 {w:.2f} {h:.2f}">\n{body}\n</svg>\n')


# --------------------------------------------------------------------------- build
def build(a):
    out = a.out
    os.makedirs(os.path.join(out, "laser", "pieces"), exist_ok=True)
    os.makedirs(os.path.join(out, "laser", "sheets"), exist_ok=True)
    mm = 1000.0 / a.scale                                   # mm per m
    dz = a.card / mm                                        # m per layer
    r0, r1, c0, c1 = a.window
    src = rasterio.open(a.dtm)
    z = src.read(1).astype(float)[r0:r1, c0:c1]
    z[z < -1000] = np.nan
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    zs = ndimage.gaussian_filter(z, a.smooth)
    up = a.upsample
    zu = ndimage.zoom(zs, up, order=1)
    W, H = (c1 - c0) * mm, (r1 - r0) * mm                    # model size, mm
    left, top = src.transform.c, src.transform.f
    fwd = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    def lonlat_to_mm(g):
        return shp_transform(lambda x, y, zz=None: (
            (np.asarray(fwd.transform(x, y)[0]) - left - c0) * mm,
            (top - np.asarray(fwd.transform(x, y)[1]) - r0) * mm), g)

    frame = box(0, 0, W, H)
    import contourpy
    cgen = contourpy.contour_generator(z=np.pad(zu, ((0, 1), (0, 1)), mode="edge"),
                                       fill_type=contourpy.FillType.OuterOffset)
    zmin, zmax = float(np.nanmin(zs)), float(np.nanmax(zs))
    # Everything below --base-z is the base board (it would be full sheets of card); the first card layer
    # is the ground at or above base-z, so the lowest water sits on the board.
    z_base = a.base_z if a.base_z is not None else math.floor(zmin) - dz
    levels = [z_base + k * dz for k in range(int((zmax - z_base) / dz) + 1)]
    print(f"model {W:.0f} x {H:.0f} mm; ground {zmin:.1f}-{zmax:.1f} m; {len(levels)} layers of {a.card} mm "
          f"(1 layer = {dz:.2f} m)")

    # ------------------------------------------------------------ terrain layers
    layers = []
    for k, lev in enumerate(levels):
        if k == 0 and a.base_z is None:
            g = frame
        else:
            # Smooth outlines: filled contours (marching squares) of the ground at or above this level.
            pts_list, offs_list = cgen.filled(lev, np.inf)
            polys = []
            for pts, offs in zip(pts_list, offs_list):
                rings = [pts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
                rings = [r for r in rings if len(r) >= 4]
                if rings:
                    q = Polygon(rings[0], rings[1:]).buffer(0)
                    if not q.is_empty:
                        polys.append(q)
            g = unary_union(polys)
            # contour coordinates are pixel centres of the upsampled grid
            g = shp_transform(lambda x, y, zz=None: ((np.asarray(x) + 0.5) / up * mm, (np.asarray(y) + 0.5) / up * mm), g)
            g = g.simplify(a.simplify * mm).intersection(frame)
            g = clean(g, a.open_m * mm, a.min_area_m2 * mm * mm)
            if layers:
                g = g.intersection(layers[-1]["g"])             # each layer rests on the one below
        if g.is_empty:
            break
        layers.append(dict(k=k, z=lev, g=g))
    print(f"{len(layers)} layers with material; top at {layers[-1]['z']:.1f} m")
    # Hollow each layer under the next one, keeping a glue ring and a rib grid that lines up on every
    # layer (so ribs stack into columns); the top of each layer, where it shows, stays solid.
    nx_, ny_ = a.tiles
    xs_ = sorted(set(list(np.arange(0, W + 1e-6, a.rib_spacing)) + [W / nx_ * i for i in range(nx_ + 1)]))
    ys_ = sorted(set(list(np.arange(0, H + 1e-6, a.rib_spacing)) + [H / ny_ * i for i in range(ny_ + 1)]))
    ribs = unary_union([LineString([(x, 0), (x, H)]) for x in xs_] + [LineString([(0, y), (W, y)]) for y in ys_]
                       ).buffer(a.rib / 2, cap_style="flat")
    for i, L in enumerate(layers):
        if a.hollow and i + 1 < len(layers):
            void = layers[i + 1]["g"].buffer(-a.ring, join_style="round").difference(ribs)
            void = unary_union([q for q in parts(void) if q.area > 400.0])      # skip voids under 20 x 20 mm
            L["solid"] = L["g"].difference(void)
        else:
            L["solid"] = L["g"]
    full = sum(L["g"].area for L in layers)
    kept = sum(L["solid"].area for L in layers)
    print(f"cardboard area: {kept / 1e6:.2f} m2 hollow vs {full / 1e6:.2f} m2 solid")

    # ------------------------------------------------------------ roads, buildings, structures
    def load(name):
        p = os.path.join(a.overture, f"{name}.geojson")
        return json.load(open(p))["features"] if os.path.exists(p) else []

    roads = []
    for f in load("roads"):
        cls = f["properties"].get("class") or "unknown"
        g = lonlat_to_mm(shape(f["geometry"])).intersection(frame)
        if g.is_empty:
            continue
        w = ROAD_W.get(cls, 5.0) * mm
        roads.append(dict(cls=cls, line=g, poly=g.buffer(w / 2, cap_style="flat", join_style="round")))
    road_poly = unary_union([r["poly"] for r in roads if r["cls"] not in ("path", "footway", "steps", "track")])
    road_thin = unary_union([r["line"] for r in roads if r["cls"] in ("path", "footway", "steps", "track")])

    dsm = rasterio.open(a.dsm).read(1).astype(float)[r0:r1, c0:c1]
    chm = np.clip(dsm - z, 0, None)
    bld = []
    for f in load("buildings"):
        g = lonlat_to_mm(shape(f["geometry"]))
        if not g.intersects(frame) or g.area < 1.0:
            continue
        g = g.intersection(frame)
        h = f["properties"].get("height")
        src_h = "overture"
        if not h:
            m = features.rasterize([(shp_transform(lambda x, y, zz=None: (np.asarray(x) / mm, np.asarray(y) / mm), g), 1)],
                                   out_shape=chm.shape, dtype=np.uint8).astype(bool)
            vals = chm[m]
            h = float(np.percentile(vals, 50)) if vals.size else 0.0
            src_h = "lidar"
        cls = (f["properties"].get("class") or f["properties"].get("subtype") or "").lower()
        cap = 4.5 if cls in ("garage", "shed", "outbuilding") else 9.0 if cls in ("house", "residential", "static_caravan") else 15.0
        if src_h == "lidar" and h > cap:
            h, src_h = cap, "lidar, capped for type (trees over the roof)"
        h = float(min(max(h, 2.5), 30.0))
        cen = g.representative_point()
        zg = float(zs[int(min(max(cen.y / mm, 0), zs.shape[0] - 1)), int(min(max(cen.x / mm, 0), zs.shape[1] - 1))])
        bld.append(dict(g=g, h=h, src=src_h, z=zg, cls=f["properties"].get("class") or f["properties"].get("subtype")))
    bld = [b for b in bld if b["g"].area >= 4.0]           # under 2 x 2 mm: too small to make
    print(f"{len(roads)} road segments, {len(bld)} buildings "
          f"({sum(b['src'] == 'overture' for b in bld)} heights from Overture, {sum(b['src'] == 'lidar' for b in bld)} from lidar)")

    infra = load("infrastructure")
    bridge = [lonlat_to_mm(shape(f["geometry"])).intersection(frame) for f in infra
              if f["properties"].get("subtype") == "bridge"]
    bridge = [g for g in bridge if g.geom_type == "LineString" and g.length > 20 * mm]
    weirs = [(((f["properties"].get("names") or {}) or {}).get("primary"), lonlat_to_mm(shape(f["geometry"])))
             for f in infra if f["properties"].get("class") == "weir"]
    deck_line = max(bridge, key=lambda g: g.length) if bridge else None

    def z_at(x, y):
        r, c = int(np.clip(y / mm, 0, zs.shape[0] - 1)), int(np.clip(x / mm, 0, zs.shape[1] - 1))
        return float(zs[r, c])

    wood = []
    if deck_line is not None:
        # The deck spans the water: the part of the bridge line over ground below --water-max, plus
        # a short bearing onto each bank. Its level is the bank top where it lands.
        n = 400
        pts = [deck_line.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
        zz = np.array([z_at(q.x, q.y) for q in pts])
        wet = np.nonzero(zz < a.water_max)[0]
        i0, i1 = int(wet.min()), int(wet.max())
        step = deck_line.length / (n - 1)
        bear = int(a.bearing * mm / step) + 1
        j0, j1 = max(0, i0 - bear), min(n - 1, i1 + bear)
        span = LineString([(q.x, q.y) for q in pts[j0:j1 + 1]])
        z0 = float(np.max(zz[max(0, i0 - bear * 3):i0])) if i0 > 0 else float(zz[i0])
        z1 = float(np.max(zz[i1 + 1:min(n, i1 + 1 + bear * 3)])) if i1 + 1 < n else float(zz[i1])
        deck_z = max(z0, z1) + 0.5
        deck = span.buffer(a.deck_width / 2 * mm, cap_style="flat")
        wood.append(dict(name="S1 BRIDGE DECK (PTH 44)", g=deck, z_bottom=deck_z - 1.0, z_top=deck_z,
                         note=f"deck at about {deck_z:.1f} m (bank tops where it lands {z0:.1f} / {z1:.1f} m); "
                              f"{a.deck_width:.0f} m wide; span {span.length / mm:.0f} m (estimates)"))
    for name, wl in weirs:
        if not wl.intersects(frame):
            continue
        wl = wl.intersection(frame)
        if wl.geom_type != "LineString" or wl.length < 5 * mm:
            continue
        if (name or "").startswith("St. Andrews") and deck_line is not None:
            zw = float(np.median([z_at(*c) for c in wl.coords]))
            wood.append(dict(name="S2 DAM CURTAIN WALL", g=wl.buffer(0.6 * mm, cap_style="flat"), z_bottom=zw, elevation=True,
                             z_top=deck_z - 1.0, note=f"curtain frames from the water ({zw:.1f} m) up to the deck; "
                                                       "schematic: check photos for pier spacing"))
        else:
            mid = wl.interpolate(0.5, normalized=True)
            zw = z_at(mid.x, mid.y)
            ht = float(np.percentile([chm[int(np.clip(c[1] / mm, 0, chm.shape[0] - 1)),
                                          int(np.clip(c[0] / mm, 0, chm.shape[1] - 1))] for c in wl.coords], 90))
            wood.append(dict(name="S3 " + (name or "WEIR").upper(), g=wl.buffer(3.0 * mm, cap_style="flat"), z_bottom=zw,
                             z_top=zw + max(ht, 3.0), note="height from the lidar surface model; schematic"))
    print("wood structures:", ", ".join(w["name"] for w in wood))

    # ------------------------------------------------------------ pieces per layer and tile
    nx, ny = a.tiles
    tw, th = W / nx, H / ny
    print(f"tiles {nx} x {ny} of {tw:.0f} x {th:.0f} mm")
    pieces = []
    for i, L in enumerate(layers):
        above = layers[i + 1]["g"] if i + 1 < len(layers) else Polygon()
        exposed = L["g"].difference(above.buffer(0.5))
        eng_roads = unary_union([road_poly.boundary.intersection(exposed),
                                 road_thin.intersection(exposed) if not road_thin.is_empty else Polygon()])
        eng_bld = unary_union([b["g"].boundary for b in bld]).intersection(exposed) if bld else Polygon()
        for ty in range(ny):
            for tx in range(nx):
                tile = box(tx * tw, ty * th, (tx + 1) * tw, (ty + 1) * th)
                g = L["solid"].intersection(tile)
                for j, p in enumerate(sorted(parts(g), key=lambda q: -q.area)):
                    if p.area < a.min_area_m2 * mm * mm:
                        continue
                    lab = f"L{L['k']:02d}-T{ty * nx + tx + 1}-P{j + 1}"
                    pieces.append(dict(label=lab, layer=L["k"], z=L["z"], tile=ty * nx + tx + 1, g=p,
                                       above=above.intersection(p).boundary,
                                       roads=eng_roads.intersection(p), blds=eng_bld.intersection(p)))
    print(f"{len(pieces)} cardboard pieces")

    def piece_svg_body(pc, dx, dy):
        g = translate(pc["g"], dx, dy)
        body = [f'<path d="{path_d(g)}" fill="none" stroke="{CUT}" stroke-width="0.025"/>']
        for key in ("above", "roads", "blds"):
            gg = pc[key]
            if gg is not None and not gg.is_empty:
                body.append(f'<path d="{line_d(translate(gg, dx, dy))}" fill="none" stroke="{ENGRAVE}" '
                            f'stroke-width="0.025"/>')
        # Label where there is room.
        try:
            pt = polylabel(pc["g"], tolerance=0.5)
        except Exception:
            pt = pc["g"].representative_point()
        room = pc["g"].exterior.distance(pt) if pc["g"].geom_type == "Polygon" else 0
        hgt = min(a.label_h, room * 0.9)
        if hgt >= 1.6:
            d, wtxt = stroke_text(pc["label"], 0, 0, hgt)
            if wtxt < 2 * room * 0.95 or True:
                d, wtxt = stroke_text(pc["label"], pt.x + dx - wtxt / 2, pt.y + dy - hgt / 2, hgt)
                body.append(f'<path d="{d}" fill="none" stroke="{ENGRAVE}" stroke-width="0.025"/>')
        return "\n".join(body)

    for pc in pieces:
        x0, y0, x1, y1 = pc["g"].bounds
        m = 3.0
        open(os.path.join(out, "laser", "pieces", f"{pc['label']}.svg"), "w").write(
            svg_doc(x1 - x0 + 2 * m, y1 - y0 + 2 * m, piece_svg_body(pc, m - x0, m - y0)))

    # Nest pieces by shape onto bed-sized sheets: rasterise at --nest-res mm, place each piece (largest
    # first, four rotations) at the first free spot scanning top-left, so small pieces fill the hollows.
    from scipy.signal import fftconvolve
    from shapely.affinity import rotate
    SW, SH = a.sheet
    res = a.nest_res
    gx, gy = int(SW / res), int(SH / res)
    pad = max(1, int(round(a.nest_gap / res)))

    def raster(g):
        x0, y0, x1, y1 = g.bounds
        w, h = int(math.ceil((x1 - x0) / res)) + 1, int(math.ceil((y1 - y0) / res)) + 1
        t = rasterio.transform.from_origin(x0, y0, res, -res)
        m = features.rasterize([(g, 1)], out_shape=(h, w), transform=t, all_touched=True, dtype=np.uint8)
        return m.astype(bool)

    sheets = []                                  # each: dict(occ=bool grid, items=[(piece, rot, x_mm, y_mm)])
    order = sorted(pieces, key=lambda p: -p["g"].area)
    for pc in order:
        placed = False
        cands = []
        for rot in (0, 90, 180, 270):
            g = rotate(pc["g"], rot, origin=(0, 0))
            m = raster(g)
            md = ndimage.binary_dilation(m, iterations=pad)
            cands.append((rot, g, m, md))
        for sh in sheets + [None]:
            if sh is None:
                sh = dict(occ=np.zeros((gy, gx), bool), items=[])
                sheets.append(sh)
            best = None
            for rot, g, m, md in cands:
                hh, ww = md.shape
                if hh > gy or ww > gx:
                    continue
                conv = fftconvolve(sh["occ"].astype(np.float32), md[::-1, ::-1].astype(np.float32), mode="valid")
                free = np.argwhere(conv < 0.5)
                if not len(free):
                    continue
                score = free[:, 0] * gx + free[:, 1]          # top-most, then left-most
                j = int(np.argmin(score))
                if best is None or score[j] < best[0]:
                    best = (score[j], rot, g, md, free[j])
            if best is not None:
                _, rot, g, md, (iy, ix) = best
                sh["occ"][iy:iy + md.shape[0], ix:ix + md.shape[1]] |= md
                x0, y0, _, _ = g.bounds
                sh["items"].append((pc, rot, ix * res + pad * res - x0, iy * res + pad * res - y0))
                placed = True
                break
        if not placed:
            print("could not place", pc["label"])
    for si, sh in enumerate(sheets, 1):
        body = []
        for pc, rot, dx, dy in sh["items"]:
            p2 = dict(pc)
            for key in ("g", "above", "roads", "blds"):
                if p2[key] is not None and not p2[key].is_empty:
                    p2[key] = rotate(p2[key], rot, origin=(0, 0))
            body.append(piece_svg_body(p2, dx, dy))
        open(os.path.join(out, "laser", "sheets", f"sheet_{si:02d}.svg"), "w").write(svg_doc(SW, SH, "\n".join(body)))
        write_dxf(os.path.join(out, "laser", "sheets_dxf", f"sheet_{si:02d}.dxf"), "\n".join(body), SW, SH)
    print(f"nested onto {len(sheets)} sheets of {SW:.0f} x {SH:.0f} mm")

    # ------------------------------------------------------------ assembly map (all layers, small multiples)
    cols = 6
    sc = 110.0 / max(W, H)
    cw, ch = W * sc + 12, H * sc + 16
    rows_n = math.ceil(len(layers) / cols)
    body = []
    for i, L in enumerate(layers):
        ox, oy = 8 + (i % cols) * cw, 10 + (i // cols) * ch
        g = shp_transform(lambda x, y, zz=None: (ox + np.asarray(x) * sc, oy + np.asarray(y) * sc), L["g"])
        body.append(f'<rect x="{ox:.2f}" y="{oy:.2f}" width="{W * sc:.2f}" height="{H * sc:.2f}" fill="none" '
                    f'stroke="#999" stroke-width="0.1"/>')
        body.append(f'<path d="{path_d(g)}" fill="#e9e4da" stroke="#111" stroke-width="0.15"/>')
        for tx in range(1, nx):
            body.append(f'<line x1="{ox + tx * tw * sc:.2f}" y1="{oy:.2f}" x2="{ox + tx * tw * sc:.2f}" '
                        f'y2="{oy + H * sc:.2f}" stroke="#c00" stroke-width="0.1" stroke-dasharray="1 1"/>')
        for t2 in range(1, ny):
            body.append(f'<line x1="{ox:.2f}" y1="{oy + t2 * th * sc:.2f}" x2="{ox + W * sc:.2f}" '
                        f'y2="{oy + t2 * th * sc:.2f}" stroke="#c00" stroke-width="0.1" stroke-dasharray="1 1"/>')
        body.append(f'<text x="{ox:.2f}" y="{oy - 2:.2f}" font-family="Helvetica,Arial" font-size="3">LAYER {L["k"]:02d} '
                    f'· +{L["z"]:.0f} M · {sum(p["layer"] == L["k"] for p in pieces)} PIECES</text>')
        for pc in pieces:
            if pc["layer"] == L["k"]:
                pt = pc["g"].representative_point()
                body.append(f'<text x="{ox + pt.x * sc:.2f}" y="{oy + pt.y * sc:.2f}" font-family="Helvetica,Arial" '
                            f'font-size="1.6" text-anchor="middle">{pc["label"].split("-", 1)[1]}</text>')
    open(os.path.join(out, "assembly_map.svg"), "w").write(svg_doc(8 + cols * cw, 10 + rows_n * ch, "\n".join(body)))

    # ------------------------------------------------------------ wood parts sheet + schedule
    def layer_top_z(zg):
        k = max(L["k"] for L in layers if L["z"] <= zg) if any(L["z"] <= zg for L in layers) else 0
        return (k + 1) * a.card                             # model height of the cardboard top there, mm
    wood_items = [dict(name=w["name"], g=w["g"], h=(w["z_top"] - w["z_bottom"]) * mm,
                       base=max(0.0, (w["z_bottom"] - z_base) * mm), note=w["note"],
                       elevation=w.get("elevation", False)) for w in wood]
    for i, b in enumerate(sorted(bld, key=lambda b: -b["g"].area), 1):
        wood_items.append(dict(name=f"B{i:03d}", g=b["g"], h=b["h"] * mm, base=layer_top_z(b["z"]),
                               note=f"{b['cls'] or 'building'}; height {b['h']:.1f} m ({b['src']})"))
    # Pack footprints for cutting from wood sheet.
    gap = 3.0
    body, x, y, shelf = [], gap, gap, 0.0
    for it in wood_items:
        g_cut = it["g"]
        if it.get("elevation"):
            # an upright piece: cut its elevation, length x height
            ln = it["g"].length / 2
            g_cut = box(0, 0, ln, max(it["h"], 1.0))
        bx0, by0, bx1, by1 = g_cut.bounds
        w, h = bx1 - bx0, by1 - by0
        if x + w > SW - gap:
            x, y, shelf = gap, y + shelf + gap + 3, 0.0
        g = translate(g_cut, x - bx0, y - by0)
        body.append(f'<path d="{path_d(g) if g.geom_type != "LineString" else line_d(g)}" fill="none" '
                    f'stroke="{CUT}" stroke-width="0.025"/>')
        d, _ = stroke_text(it["name"].split()[0], x, y + h + 0.6, 1.6)
        if d:
            body.append(f'<path d="{d}" fill="none" stroke="{ENGRAVE}" stroke-width="0.025"/>')
        x += max(w, 7.0) + gap
        shelf = max(shelf, h)
    open(os.path.join(out, "laser", "wood_parts.svg"), "w").write(svg_doc(SW, max(SH, y + shelf + 10), "\n".join(body)))
    with open(os.path.join(out, "wood_schedule.csv"), "w") as f:
        f.write("part,length_mm,width_mm,height_mm,underside_above_base_board_mm,note\n")
        for it in wood_items:
            rr = it["g"].minimum_rotated_rectangle
            cs = np.asarray(rr.exterior.coords)
            e = sorted([float(np.hypot(*(cs[i + 1] - cs[i]))) for i in range(2)])
            f.write(f"{it['name']},{e[1]:.1f},{e[0]:.1f},{it['h']:.1f},{it['base']:.0f},\"{it['note']}\"\n")

    # ------------------------------------------------------------ 3D model (OBJ, model mm, z up)
    verts, faces, objs = [], [], []
    vindex = {}

    def V(x, y, zz):
        key = (round(x, 2), round(-y, 2), round(zz, 2))
        if key not in vindex:
            verts.append(key)
            vindex[key] = len(verts)
        return vindex[key]

    def add_prism(name, g, zb, zt, simp=0.0):
        start_f = len(faces)
        if simp:
            g = g.simplify(simp)
        for p in parts(g) if g.geom_type != "LineString" else []:
            tris = constrained_delaunay_triangles(p)
            for t in getattr(tris, "geoms", []):
                cs = list(t.exterior.coords)[:3]
                faces.append([V(cx, cy, zt) for cx, cy in cs])
                faces.append([V(cx, cy, zb) for cx, cy in cs][::-1])
            for ring in [p.exterior] + list(p.interiors):
                cs = list(ring.coords)
                for (xa, ya), (xb, yb) in zip(cs[:-1], cs[1:]):
                    faces.append([V(xa, ya, zb), V(xb, yb, zb), V(xb, yb, zt), V(xa, ya, zt)])
        objs.append((name, start_f, len(faces)))

    add_prism("base_board", frame, -a.base_mm, 0.0)
    for L in layers:
        add_prism(f"card_layer_{L['k']:02d}", L["g"], L["k"] * a.card, (L["k"] + 1) * a.card, simp=0.8)
    for it in wood_items:
        g = it["g"] if it["g"].geom_type != "LineString" else it["g"].buffer(0.5)
        add_prism("wood_" + it["name"].replace(" ", "_").replace("(", "").replace(")", ""), g, it["base"],
                  it["base"] + max(it["h"], 0.5))
    lines = []
    for r in roads:
        for q in getattr(r["line"], "geoms", [r["line"]]):
            if q.geom_type != "LineString":
                continue
            idx = []
            for cx, cy in q.coords:
                idx.append(V(cx, cy, layer_top_z(z_at(cx, cy)) + 0.05))
            lines.append(idx)
    with open(os.path.join(out, "lockport_1to500.obj"), "w") as f:
        f.write("# Lockport site model, 1:500, units mm (model), z up; x east, y north\n")
        for v in verts:
            f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
        for name, s, e in objs:
            f.write(f"o {name}\n")
            for fc in faces[s:e]:
                f.write("f " + " ".join(map(str, fc)) + "\n")
        f.write("o roads\n")
        for idx in lines:
            f.write("l " + " ".join(map(str, idx)) + "\n")
    print(f"OBJ: {len(verts)} vertices, {len(faces)} faces")

    # ------------------------------------------------------------ summary
    with open(os.path.join(out, "layers.csv"), "w") as f:
        f.write("layer,level_m,area_m2,pieces\n")
        for L in layers:
            f.write(f"{L['k']},{L['z']:.1f},{L['g'].area / (mm * mm):.0f},{sum(p['layer'] == L['k'] for p in pieces)}\n")
    json.dump(dict(window_rows_cols=a.window, scale=a.scale, card_mm=a.card, model_mm=[W, H], tiles=a.tiles,
                   layers=len(layers), pieces=len(pieces), sheets=len(sheets), z_base=z_base),
              open(os.path.join(out, "model.json"), "w"), indent=1)
    write_layers_dxf(os.path.join(out, "lockport_contour_layers.dxf"), layers, W, H, a.card)
    return layers, pieces, bld, wood, W, H


def write_layers_dxf(path, layers, W, H, card):
    """Every terrain layer's outline in one DXF, model mm, y up (north), all lines black.
    Each layer is its own DXF layer (L00_218m ...), and its polylines sit at that layer's
    height (k x card), so the file reads as a plan and stacks in 3D."""
    import ezdxf
    doc = ezdxf.new(units=4)
    black = ezdxf.colors.rgb2int((0, 0, 0))
    msp = doc.modelspace()
    doc.layers.add("MODEL_EDGE", true_color=black)
    msp.add_lwpolyline([(0, 0), (W, 0), (W, H), (0, H)], close=True,
                       dxfattribs={"layer": "MODEL_EDGE", "true_color": black})
    for L in layers:
        name = f"L{L['k']:02d}_{L['z']:.0f}m"
        doc.layers.add(name, true_color=black)
        for p in parts(L["g"]):
            for ring in [p.exterior] + list(p.interiors):
                pts = [(x, H - y) for x, y in ring.coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": name, "true_color": black,
                                                                "elevation": L["k"] * card})
    doc.saveas(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--overture", default="data/lockport/overture")
    ap.add_argument("--window", type=int, nargs=4, default=[1415, 2265, 1200, 1960], metavar=("R0", "R1", "C0", "C1"),
                    help="DEM rows and columns (1 m): 850 m north-south x 760 m east-west")
    ap.add_argument("--scale", type=int, default=500)
    ap.add_argument("--card", type=float, default=2.0, help="mm per cardboard layer")
    ap.add_argument("--tiles", type=int, nargs=2, default=[3, 2], help="tiles across, down")
    ap.add_argument("--sheet", type=float, nargs=2, default=[590.0, 890.0], help="mm, usable laser bed")
    ap.add_argument("--smooth", type=float, default=2.0, help="m, Gaussian smoothing of the lidar")
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--simplify", type=float, default=0.25, help="m")
    ap.add_argument("--open-m", type=float, default=0.75, help="m: slivers thinner than twice this are removed")
    ap.add_argument("--min-area-m2", type=float, default=20.0)
    ap.add_argument("--deck-width", type=float, default=12.0, help="m, bridge deck (estimate)")
    ap.add_argument("--water-max", type=float, default=220.5, help="m: lidar ground below this along the bridge is water")
    ap.add_argument("--bearing", type=float, default=8.0, help="m the deck rests on each bank")
    ap.add_argument("--label-h", type=float, default=4.0, help="mm, engraved label height")
    ap.add_argument("--base-z", type=float, default=218.0,
                    help="m: ground below this is the base board, not card (the river below the dam is 217.7 m)")
    ap.add_argument("--base-mm", type=float, default=12.0, help="mm, base board thickness (3D only)")
    ap.add_argument("--hollow", type=int, default=0, help="1: hollow layers under the next one (less card, more cutting)")
    ap.add_argument("--ring", type=float, default=12.0, help="mm of glue ring kept under the next layer's edge")
    ap.add_argument("--rib", type=float, default=8.0, help="mm, support rib width")
    ap.add_argument("--rib-spacing", type=float, default=80.0, help="mm between ribs")
    ap.add_argument("--nest-res", type=float, default=2.0, help="mm, nesting raster")
    ap.add_argument("--nest-gap", type=float, default=3.0, help="mm between nested pieces")
    ap.add_argument("--out", default="outputs/site_model")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
