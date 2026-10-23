"""Cut a site window out of NRCan's 1 m HRDEM lidar and save it locally.

The national lidar projects are published as cloud-optimized GeoTIFFs, so only
the requested window is downloaded (a few MB), not the whole project.
Projects are merged in the order given: later projects fill gaps in earlier
ones, so list the preferred (usually newest) survey first.

    python tools/fetch_hrdem.py                  # Lockport defaults
    python tools/fetch_hrdem.py --keep-projects  # also save each survey separately

Requires rasterio (pip install rasterio).
"""

import argparse
import os

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


BUCKET = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/hrdem-lidar"

# Lockport, MB: the lock and dam, both banks and the Red River Floodway outlet.
LOCKPORT_BBOX = (-96.9588, 50.0679, -96.9107, 50.1032)  # lon/lat W, S, E, N
LOCKPORT_PROJECTS = ["MB-Red_River_2020", "MB-Cooks_Creek"]


def read_window(project, surface, bounds_lonlat, snap_bounds=None):
    url = f"/vsicurl/{BUCKET}/{project}-1m-{surface}.tif"
    with rasterio.open(url) as src:
        if snap_bounds is None:
            b = transform_bounds("EPSG:4326", src.crs, *bounds_lonlat)
            # Snap to whole metres so every project lands on the same grid.
            snap_bounds = (np.floor(b[0]), np.floor(b[1]), np.ceil(b[2]), np.ceil(b[3]))
        win = from_bounds(*snap_bounds, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win, boundless=True, fill_value=src.nodata)
        data = np.where(data == src.nodata, np.nan, data).astype(np.float32)
        profile = src.profile.copy()
        profile.update(
            width=data.shape[1], height=data.shape[0],
            transform=src.window_transform(win), driver="GTiff",
        )
        date = src.tags().get("TIFFTAG_DATETIME", "")[:10].replace(":", "-")
    return data, profile, snap_bounds, date


def write(path, data, profile, description):
    nodata = -32767.0
    out = np.where(np.isnan(data), nodata, np.round(data, 2)).astype(np.float32)
    profile.update(
        dtype="float32", nodata=nodata, count=1, compress="deflate", predictor=3,
        zlevel=9, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="NO",
    )
    for k in ("photometric", "interleave"):
        profile.pop(k, None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(DESCRIPTION=description)
    print(f"wrote {path}  {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=LOCKPORT_BBOX,
                    metavar=("W", "S", "E", "N"), help="lon/lat bounds")
    ap.add_argument("--projects", nargs="+", default=LOCKPORT_PROJECTS)
    ap.add_argument("--surfaces", nargs="+", default=["dtm", "dsm"], choices=["dtm", "dsm"])
    ap.add_argument("--name", default="lockport")
    ap.add_argument("--out", default="data/lockport/dem")
    ap.add_argument("--keep-projects", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for surface in args.surfaces:
        merged, profile, snap, sources = None, None, None, []
        for project in args.projects:
            data, prof, snap, date = read_window(project, surface, args.bbox, snap)
            coverage = np.mean(~np.isnan(data))
            print(f"{project} {surface} ({date}): {coverage:.0%} of window")
            if args.keep_projects:
                write(os.path.join(args.out, f"{args.name}_{project}_{surface}_1m.tif"),
                      data, prof.copy(), f"{project} {surface} {date}")
            if merged is None:
                merged, profile = data, prof
            else:
                merged = np.where(np.isnan(merged), data, merged)
            sources.append(f"{project} ({date})")
        print(f"merged {surface}: {np.mean(~np.isnan(merged)):.0%} of window")
        write(os.path.join(args.out, f"{args.name}_{surface}_1m.tif"), merged, profile,
              f"HRDEM 1 m {surface.upper()} merged, priority order: " + ", ".join(sources))


if __name__ == "__main__":
    main()
