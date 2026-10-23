"""Build the riverbed GeoTIFF from the digitized chart soundings.

    python tools/build_bed.py CHART_IMAGE
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from drawing import bathymetry, terrain  # noqa: E402
from register_chart import chart_water_edge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="the chart screenshot (for its water edge)")
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--transform", default="data/lockport/bathymetry/chart_transform.json")
    ap.add_argument("--soundings", default="data/lockport/bathymetry/chart_soundings.csv")
    ap.add_argument("--dam-row", type=int, default=2135)
    ap.add_argument("--ref", type=float, nargs=2, default=[222.9, 218.2],
                    help="assumed water level the depths are measured from: above / below the dam")
    ap.add_argument("--out", default="data/lockport/bathymetry/bed_from_chart.tif")
    args = ap.parse_args()
    dem, _, _ = terrain.load_dem(args.dem)
    xs, ys, _ = chart_water_edge(args.image, 1340)
    bed, depth = bathymetry.bed_surface(dem.shape, args.transform, args.soundings, (xs, ys), args.dam_row, *args.ref)
    bathymetry.save_tif(args.out, bed, args.dem)
    ok = np.isfinite(bed)
    print(f"wrote {args.out}: {ok.sum() / 1e4:.1f} ha, depth max {np.nanmax(depth):.2f} m, bed {np.nanmin(bed):.2f}-{np.nanmax(bed):.2f} m")
    # Section line check.
    row = 1840
    line = bed[row, 1300:1650]
    print("row 1840 bed (every 25 m):", [round(float(v), 1) if v == v else None for v in line[::25]])


if __name__ == "__main__":
    main()
