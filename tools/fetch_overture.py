"""Roads, buildings and infrastructure for a small box from Overture Maps
(open data that includes OpenStreetMap), read straight from its public
GeoParquet release on AWS.

Each file's footer gives every row group's bounding box (bbox.xmin...ymax
statistics), so only row groups over the box are downloaded. Geometry is
WKB; output is GeoJSON in lon/lat.

    python -m tools.fetch_overture --bbox -96.944 50.0826 -96.9346 50.0909
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import urllib.request

import fsspec
import pyarrow.parquet as pq
from shapely import wkb
from shapely.geometry import box, mapping

B = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com"
TYPES = {
    "roads": "theme=transportation/type=segment",
    "buildings": "theme=buildings/type=building",
    "infrastructure": "theme=base/type=infrastructure",
    "water": "theme=base/type=water",
}
KEEP = {
    "roads": ["id", "subtype", "class", "names"],
    "buildings": ["id", "subtype", "class", "height", "num_floors", "names"],
    "infrastructure": ["id", "subtype", "class", "names"],
    "water": ["id", "subtype", "class", "names"],
}


def keys(prefix):
    out, token = [], None
    while True:
        u = f"{B}/?list-type=2&prefix={prefix}/"
        if token:
            u += "&continuation-token=" + urllib.request.quote(token)
        s = urllib.request.urlopen(u, timeout=120).read().decode()
        out += re.findall(r"<Key>([^<]+\.parquet)</Key>", s)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", s)
        if not m:
            return out
        token = m.group(1)


def groups_in_box(meta, bb):
    """Row groups whose bbox statistics overlap bb = (x0, y0, x1, y1)."""
    names = [meta.schema.column(i).path for i in range(meta.num_columns)]
    idx = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax") if n in names}
    out = []
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        st = {k: rg.column(i).statistics for k, i in idx.items()}
        if any(s is None or not s.has_min_max for s in st.values()):
            out.append(g)
            continue
        if (st["bbox.xmin"].min <= bb[2] and st["bbox.xmax"].max >= bb[0] and
                st["bbox.ymin"].min <= bb[3] and st["bbox.ymax"].max >= bb[1]):
            out.append(g)
    return out


def read_file(key, bb, cols):
    f = fsspec.open(f"{B}/{key}", "rb", block_size=2 ** 20, cache_type="none").open()
    pf = pq.ParquetFile(f)
    gs = groups_in_box(pf.metadata, bb)
    if not gs:
        return []
    avail = [c for c in cols if c in pf.schema_arrow.names]
    t = pf.read_row_groups(gs, columns=avail + ["geometry", "bbox"])
    B_ = box(*bb)
    rows = []
    for rec in t.to_pylist():
        b = rec["bbox"]
        if b["xmax"] < bb[0] or b["xmin"] > bb[2] or b["ymax"] < bb[1] or b["ymin"] > bb[3]:
            continue
        g = wkb.loads(rec["geometry"])
        if not g.intersects(B_):
            continue
        props = {c: rec.get(c) for c in avail}
        rows.append(dict(type="Feature", properties=json.loads(json.dumps(props, default=str)), geometry=mapping(g)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--release", default="2026-11-18.0")
    ap.add_argument("--types", nargs="*", default=list(TYPES))
    ap.add_argument("--out", default="data/lockport/overture")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for t in a.types:
        ks = keys(f"release/{a.release}/{TYPES[t]}")
        feats = []
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for rows in ex.map(lambda k: read_file(k, a.bbox, KEEP[t]), ks):
                feats += rows
        dst = os.path.join(a.out, f"{t}.geojson")
        json.dump(dict(type="FeatureCollection", features=feats), open(dst, "w"))
        print(f"{t}: {len(feats)} features from {len(ks)} files -> {dst}", flush=True)


if __name__ == "__main__":
    main()
