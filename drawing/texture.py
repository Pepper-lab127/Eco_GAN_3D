"""Surface detail for the rendered (composite) sheets.

`detail(name, shape, rng, scale_px, fallback)` returns a zero-mean, unit-std
detail field for a material. If a photo texture is on file as
data/textures/<name>.jpg (or .png) - e.g. from Texture Labs - its grey values,
high-passed so only the grain is kept, are tiled across the field at
`scale_px` pixels per tile width; otherwise `fallback()` (procedural noise) is
used. Photo textures therefore only add grain: colour and layer placement
still come from the data.

Expected names: snow, ice, ice_clear, water, clay, till, shale, limestone,
bark, sky. Any can be missing.
"""

import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage

TEXTURE_DIR = os.environ.get("ECO_TEXTURES", "data/textures")


def photo_path(name):
    for ext in ("jpg", "jpeg", "png", "tif", "webp"):
        hits = glob.glob(os.path.join(TEXTURE_DIR, f"{name}.{ext}"))
        if hits:
            return hits[0]
    return None


def photo(name, shape, scale_px, rng, highpass=0.12):
    """Tiled, high-passed grey texture from a photo, or None if not on file."""
    p = photo_path(name)
    if p is None:
        return None
    im = Image.open(p).convert("L")
    w = max(8, int(scale_px))
    h = max(8, int(im.height * w / im.width))
    t = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32)
    t = t - ndimage.gaussian_filter(t, max(w, h) * highpass, mode="wrap")
    # Mirror-tile so seams do not show, then crop at a random offset.
    tile = np.block([[t, t[:, ::-1]], [t[::-1], t[::-1, ::-1]]])
    H, W = shape
    reps = (H // tile.shape[0] + 2, W // tile.shape[1] + 2)
    big = np.tile(tile, reps)
    oy, ox = rng.integers(0, tile.shape[0]), rng.integers(0, tile.shape[1])
    out = big[oy:oy + H, ox:ox + W]
    return (out - out.mean()) / (out.std() + 1e-6)


def detail(name, shape, rng, scale_px, fallback):
    t = photo(name, shape, scale_px, rng)
    return t if t is not None else fallback()


def available():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TEXTURE_DIR, "*.*"))
                  if not p.endswith(".md"))
