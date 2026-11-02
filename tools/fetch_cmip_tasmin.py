"""Annual extreme minimum temperature over southern Manitoba from NASA
NEX-GDDP-CMIP6 (daily, 0.25 degree, bias-corrected and downscaled), for the
hardiness lines on sheet 10.

For each model, run and sampled year, read only the winter days (January 1 -
March 15 and December) over the region and keep that calendar year's lowest
daily minimum per grid cell. Files are read over HTTP, a day at a time
(each day is one compressed chunk of the globe), one winter in every --step
years for each 30-year period: CanESM5 (warms most), MPI-ESM1-2-HR, MRI-ESM2-0
(warms least).

    python -m tools.fetch_cmip_tasmin [--workers 8]

Writes data/region/cmip/<model>_<run>_<year>.npy and lat/lon grids.
"""

import argparse
import concurrent.futures as cf
import os
import re
import urllib.request

import fsspec
import h5py
import numpy as np

B = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com"
MODELS = {"CanESM5": "r1i1p1f1", "MPI-ESM1-2-HR": "r1i1p1f1", "MRI-ESM2-0": "r1i1p1f1"}
RUNS = {"historical": (1985, 2015), "ssp245": (2041, 2071), "ssp585": (2041, 2071), "ssp585_late": (2071, 2101)}
BOX = dict(lat=(40.0, 63.0), lon=(-112.0, -80.0))


def keys(model, scen, member):
    out, token = [], None
    while True:
        u = f"{B}/?list-type=2&prefix=NEX-GDDP-CMIP6/{model}/{scen}/{member}/tasmin/"
        if token:
            u += "&continuation-token=" + urllib.request.quote(token)
        s = urllib.request.urlopen(u, timeout=60).read().decode()
        out += re.findall(r"<Key>([^<]+)</Key>", s)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", s)
        if not m:
            return out
        token = m.group(1)


def pick(all_keys, year):
    """The file for a year, preferring the v2.0 reissue."""
    c = [k for k in all_keys if re.search(rf"_{year}(_v\d\.\d)?\.nc$", k)]
    v2 = [k for k in c if "_v2" in k]
    return (v2 or c or [None])[0]


def one(job, out_dir):
    model, run, year, key = job
    dst = os.path.join(out_dir, f"{model}_{run}_{year}.npy")
    if os.path.exists(dst):
        return dst
    f = fsspec.open(f"{B}/{key}", "rb", block_size=4 * 2**20, cache_type="none").open()
    h = h5py.File(f, "r")
    lat, lon = h["lat"][:], h["lon"][:]
    lon = np.where(lon > 180, lon - 360, lon)
    ri = np.nonzero((lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1]))[0]
    ci = np.nonzero((lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1]))[0]
    v = h["tasmin"]
    n = v.shape[0]
    a = v[0:74, ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1].min(0)
    b = v[n - 31:n, ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1].min(0)
    m = np.minimum(a, b).astype(np.float32) - 273.15
    np.save(dst, m)
    grid = os.path.join(out_dir, "grid.npz")
    if not os.path.exists(grid):
        np.savez(grid, lat=lat[ri[0]:ri[-1] + 1], lon=lon[ci[0]:ci[-1] + 1])
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/region/cmip")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step", type=int, default=6, help="years between sampled winters")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for model, member in MODELS.items():
        for run, years in RUNS.items():
            scen = run.split("_")[0]
            ks = keys(model, scen, member)
            for y in range(years[0], years[1], a.step):
                k = pick(ks, y)
                if k:
                    jobs.append((model, run, y, k))
                else:
                    print("missing", model, run, y, flush=True)
    print(len(jobs), "files", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, j, a.out): j for j in jobs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            j = futs[fu]
            try:
                fu.result()
                print(f"{i}/{len(jobs)} {j[0]} {j[1]} {j[2]}", flush=True)
            except Exception as e:
                print(f"FAILED {j[:3]}: {e}", flush=True)


if __name__ == "__main__":
    main()
