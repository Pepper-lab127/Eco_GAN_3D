"""Mean annual extreme minimum temperature (the USDA hardiness measure) at
GHCN-Daily stations around southern Manitoba, for 1961-1990 and 1991-2020.

A station-year counts only if its daily minimum record is near complete
through the cold months (at least 25 valid days in each of January, February
and December); a period needs at least 20 such years. Quality-flagged values
are dropped. Canadian stations change identifier when an instrument moves
(Winnipeg airport is four), so stations within 5 km and 50 m of elevation
are joined into one record, each year taken from whichever has it complete.

    python -m tools.ghcn_extreme_min [--raw DIR]
"""

import argparse
import concurrent.futures as cf
import csv
import gzip
import io
import os
import urllib.request

import numpy as np

B = "https://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/by_station"
PERIODS = {"1961_1990": (1961, 1990), "1991_2020": (1991, 2020)}


def station_minima(sid, raw):
    path = os.path.join(raw, f"{sid}.csv.gz")
    if not os.path.exists(path):
        with urllib.request.urlopen(f"{B}/{sid}.csv.gz", timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    ymin, count = {}, {}
    with gzip.open(path, "rt") as f:
        for row in csv.reader(f):
            if len(row) < 6 or row[2] != "TMIN" or row[5]:
                continue
            d = row[1]
            y, m = int(d[:4]), int(d[4:6])
            if m not in (1, 2, 12):
                continue
            v = int(row[3]) / 10.0
            ymin[y] = min(ymin.get(y, 99.0), v)
            count[(y, m)] = count.get((y, m), 0) + 1
    return {y: v for y, v in ymin.items() if all(count.get((y, m), 0) >= 25 for m in (1, 2, 12))}


def select(stations_txt, inventory_txt, box=((41, 62), (-110, -82))):
    """Every station in the box with at least 5 years of daily minima between 1961 and 2020."""
    inv = {}
    for l in open(inventory_txt):
        if l[31:35] == "TMIN":
            a, b = int(l[36:40]), int(l[41:45])
            if min(b, 2020) - max(a, 1961) >= 4:
                inv[l[0:11]] = (a, b)
    out = []
    for l in open(stations_txt):
        sid, lat, lon, el, name = l[0:11], float(l[12:20]), float(l[21:30]), float(l[31:37]), l[41:71].strip()
        if sid in inv and box[0][0] <= lat <= box[0][1] and box[1][0] <= lon <= box[1][1]:
            out.append((sid, lat, lon, el, name))
    return out


def clusters(st, km=5.0, dz=50.0):
    """Group stations within km and dz of each other (single linkage)."""
    n = len(st)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lat = np.array([s[1] for s in st]); lon = np.array([s[2] for s in st]); el = np.array([s[3] for s in st])
    for i in range(n):
        d = np.hypot((lat - lat[i]) * 111.2, (lon - lon[i]) * 111.2 * np.cos(np.radians(lat[i])))
        for j in np.nonzero((d < km) & (np.abs(el - el[i]) < dz))[0]:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(st[i])
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations-txt", default="data/region/ghcn/ghcnd-stations.txt")
    ap.add_argument("--inventory-txt", default="data/region/ghcn/ghcnd-inventory.txt")
    ap.add_argument("--raw", required=True, help="directory for the downloaded station files")
    ap.add_argument("--out", default="data/region/ghcn/station_extreme_min.csv")
    a = ap.parse_args()
    os.makedirs(a.raw, exist_ok=True)
    sel = select(a.stations_txt, a.inventory_txt)
    print(len(sel), "stations with daily minima in the box", flush=True)
    years = {}
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(station_minima, s[0], a.raw): s for s in sel}
        for fu in cf.as_completed(futs):
            s = futs[fu]
            try:
                years[s[0]] = fu.result()
            except Exception as e:
                print("failed", s[0], e)
    rows = []
    for g in clusters([s for s in sel if s[0] in years]):
        g = sorted(g, key=lambda s: -len(years[s[0]]))          # longest record first
        merged = {}
        for s in g:
            for y, v in years[s[0]].items():
                merged.setdefault(y, v)
        head = g[0]
        r = []
        for name, (y0, y1) in PERIODS.items():
            vals = [v for y, v in merged.items() if y0 <= y <= y1]
            r += [f"{np.mean(vals):.2f}" if len(vals) >= 20 else "", len(vals)]
        if any(r[0::2]):
            rows.append(["+".join(s[0] for s in g), head[4], head[1], head[2], head[3]] + r)
    rows.sort()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "lat", "lon", "elev_m"] + [f"{k}_{p}" for p in PERIODS for k in ("tmin_ext", "years")])
        w.writerows(rows)
    n = {p: sum(1 for r in rows if r[5 + 2 * i]) for i, p in enumerate(PERIODS)}
    print(f"wrote {a.out}: {len(rows)} stations; usable {n}")


if __name__ == "__main__":
    main()
