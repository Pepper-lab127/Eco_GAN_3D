"""Elevation helpers: loading, hillshade, connected flooding, contour lines."""

import base64
import io

import contourpy
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage


def load_dem(path):
    """Return (heights with NaN for no data, rasterio transform, crs)."""
    with rasterio.open(path) as src:
        a = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        return a, src.transform, src.crs


def hillshade(a, cell=1.0, azimuth=315.0, altitude=40.0, z=3.0):
    """Lambertian hillshade in 0..1, heights exaggerated by z."""
    filled = np.where(np.isnan(a), np.nanmin(a), a)
    gy, gx = np.gradient(filled * z, cell, cell)
    az, alt = np.radians(azimuth), np.radians(altitude)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def flood(a, level, sources):
    """Cells at or below level that connect to any source cell.

    A bathtub fill: water spreads only through ground lower than the level,
    so low ground cut off by higher ground stays dry. sources is a boolean
    mask (e.g. the river) or a single (row, col) cell; sources are always wet.
    """
    if isinstance(sources, tuple):
        cell, sources = sources, np.zeros(a.shape, bool)
        sources[cell] = True
    below = (a <= level) | sources
    labels, _ = ndimage.label(below)
    hit = np.unique(labels[sources])
    return np.isin(labels, hit[hit > 0])


def flat_water(a, below, max_slope=0.02, min_cells=500, window=7, bridge=4,
               tolerance=0.2, min_share=0.1, largest_only=False):
    """Lidar water surfaces: broad areas at a flight-day water level.

    Lidar returns little from water, so each survey flattens a river to its
    level on the flight day, and merged surveys can carry several such levels.
    Those levels are found from the clearly flat ground lower than `below`;
    then every broad area within `tolerance` of one of them counts as water,
    which also catches the noisier parts of a flattened surface. Thin rings
    where a level crosses a bank are removed, and the result is grown by
    `bridge` cells to close seams between surveys. largest_only keeps just
    the biggest body (the river), dropping ditches and ponds at similar heights.
    """
    gy, gx = np.gradient(np.nan_to_num(a, nan=np.nanmax(a)))
    steep = ndimage.maximum_filter(np.hypot(gx, gy), window)
    flat = (steep < max_slope) & (a < below)
    if not flat.any():
        return flat

    # Flight-day levels: 5 cm bins holding a good share of the flat cells.
    bins = np.round(a[flat] / 0.05) * 0.05
    values, counts = np.unique(bins, return_counts=True)
    levels = values[counts >= min_share * counts.max()]

    near = np.zeros(a.shape, bool)
    for level in levels:
        near |= np.abs(a - level) <= tolerance
    near = ndimage.binary_opening(near, iterations=2)
    labels, n = ndimage.label(near)
    if n == 0:
        return near
    sizes = ndimage.sum(near, labels, range(1, n + 1))
    water = np.isin(labels, np.flatnonzero(sizes >= min_cells) + 1)
    if bridge:
        water |= ndimage.binary_dilation(water, iterations=bridge) & (a < below)
    if largest_only and water.any():
        labels, n = ndimage.label(water)
        sizes = ndimage.sum(water, labels, range(1, n + 1))
        water = labels == np.argmax(sizes) + 1
    return water


def mask_lines(mask):
    """Boundary lines of a boolean mask as arrays of (col, row) points."""
    if not mask.any() or mask.all():
        return []
    gen = contourpy.contour_generator(z=mask.astype(np.float32), name="serial")
    return gen.lines(0.5)


def png_data_uri(gray):
    """uint8 greyscale array -> data URI for embedding in SVG."""
    buf = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def river_reaches(a, dam, below, min_cells=1000):
    """Split a window at a dam row into the reach above it (south, larger
    rows) and below it, and find each reach's lidar water separately: the
    dam holds them at different levels.

    Returns (above, water_up, water_down); above is a boolean row mask
    broadcast to the window.
    """
    above = np.broadcast_to(np.arange(a.shape[0])[:, None] > dam, a.shape)

    def river(mask):
        return flat_water(np.where(mask, a, np.nan), below=below,
                          min_cells=min_cells, largest_only=True)

    water_up = river(above)
    water_down = river(~above) if dam >= 0 else np.zeros(a.shape, bool)
    return above, water_up, water_down


def smooth_contours(a, levels, sigma=2.0):
    """Contour lines of a lightly smoothed DEM: {level: [arrays of (col, row)]}."""
    z = ndimage.gaussian_filter(np.nan_to_num(a, nan=np.nanmean(a)), sigma)
    gen = contourpy.contour_generator(z=z, name="serial")
    return {level: gen.lines(level) for level in levels}
