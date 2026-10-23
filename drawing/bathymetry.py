"""Riverbed from chart soundings.

Soundings digitized from a chart (pixel position, depth in feet) are placed
on the lidar grid with the transform found by tools/register_chart.py,
joined with zero depth along the chart's water edge, and interpolated into
a bed surface. Depths are taken below an assumed reference water level on
each side of the dam (the summer pool), since the chart's datum is not given.
"""

import csv
import json
import os

import numpy as np
from scipy import interpolate, ndimage

FT = 0.3048


def chart_to_dem(t, x, y):
    th = np.radians(t["rotation_deg"])
    s = t["scale_m_per_px"]
    col = t["col0"] + s * (np.cos(th) * (x - t["x0"]) - np.sin(th) * (y - t["y0"]))
    row = t["row0"] + s * (np.sin(th) * (x - t["x0"]) + np.cos(th) * (y - t["y0"]))
    return row, col


def load_soundings(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    return np.array([[float(r["x"]), float(r["y"]), float(r["depth_ft"])] for r in rows])


def bed_surface(shape, transform_json, soundings_csv, water_edge_xy, dam_row, ref_above, ref_below, smooth=6.0):
    """Bed elevation (m) on a DEM-sized grid (NaN outside the chart's water)
    and the depth grid below the reference level (m).

    water_edge_xy: (xs, ys) chart pixels on the water edge (depth 0)."""
    t = json.load(open(transform_json))
    pts = load_soundings(soundings_csv)
    r_s, c_s = chart_to_dem(t, pts[:, 0], pts[:, 1])
    d_s = pts[:, 2] * FT
    ex, ey = water_edge_xy
    k = slice(None, None, max(1, len(ex) // 1500))
    r_e, c_e = chart_to_dem(t, np.asarray(ex)[k], np.asarray(ey)[k])
    rows = np.concatenate([r_s, r_e])
    cols = np.concatenate([c_s, c_e])
    depth = np.concatenate([d_s, np.zeros(len(r_e))])

    r0, r1 = int(max(rows.min(), 0)), int(min(rows.max() + 1, shape[0]))
    c0, c1 = int(max(cols.min(), 0)), int(min(cols.max() + 1, shape[1]))
    gr, gc = np.mgrid[r0:r1, c0:c1]
    d = interpolate.griddata((rows, cols), depth, (gr, gc), method="linear")
    filled = np.where(np.isnan(d), 0.0, d)
    d = np.where(np.isnan(d), np.nan, ndimage.gaussian_filter(filled, smooth))

    out_depth = np.full(shape, np.nan)
    out_depth[r0:r1, c0:c1] = np.clip(d, 0, None)
    ref = np.where(np.arange(shape[0])[:, None] > dam_row, ref_above, ref_below)
    return ref - out_depth, out_depth


def save_tif(path, arr, like_path):
    import rasterio
    with rasterio.open(like_path) as src:
        prof = src.profile.copy()
    prof.update(dtype="float32", nodata=-32767.0, compress="deflate", predictor=3, tiled=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.where(np.isnan(arr), -32767.0, arr).astype("float32"), 1)
