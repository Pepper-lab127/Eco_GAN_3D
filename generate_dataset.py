import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

from config import CONFIG
from ecology_metrics import METRIC_NAMES, compute_metrics, exterior_air


# Dataset V4 - process simulation
# -------------------------------
# V3 drew shapes that resembled ecological phenomena (blobs, tubes,
# gaussian holes). V4 simulates the processes, so form is the record of
# what happened to the material:
#
#   branching     -> space colonisation growth toward light/space,
#                    branch thickness from the pipe model
#   clustering    -> competing colonies accreting toward light
#   layering      -> deposited strata of varying hardness, weathered
#                    differentially into ledges and recesses
#   erosion       -> rain (gully incision + drip), wind scour, or salt
#                    weathering with shelter feedback (tafoni)
#   cavitation    -> karst dissolution by water percolating from above
#   porosity      -> Gray-Scott reaction-diffusion pore networks
#   fragmentation -> jointed blocks, frost cracking and spalling
#   hybridization -> builders combined, then operators applied in turn
#
# Builders make material from nothing. Operators transform an existing
# mass; on their own they act on a host form (outcrop, boulder, wall or
# column) standing on the ground.
#
# Axis 2 (z) is vertical, index 0 is the ground, as in ecology_metrics.


PHENOMENA = [
    "branching",
    "cavitation",
    "porosity",
    "erosion",
    "layering",
    "clustering",
    "fragmentation",
]

BUILDERS = {"branching", "clustering", "layering"}

SIX = ndimage.generate_binary_structure(3, 1)


# ============================================================
# Fields and helpers
# ============================================================

def grid(n):
    a = np.linspace(-1, 1, n, dtype=np.float32)
    return np.meshgrid(a, a, a, indexing="ij")


def norm(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def unit_vector(rng):
    v = rng.normal(size=3).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-8)


def resample(field, shape):
    """Cubic resampling of a coarse lattice to an exact output shape."""
    coords = np.meshgrid(
        *[
            np.linspace(0, s_in - 1, s_out)
            for s_in, s_out in zip(field.shape, shape)
        ],
        indexing="ij",
    )
    return ndimage.map_coordinates(
        field, coords, order=3, mode="reflect"
    ).astype(np.float32)


def fbm(shape, rng, base=3, octaves=4, persistence=0.5, stretch=None):
    """
    Fractal noise in [0, 1]. stretch scales the frequency per axis,
    e.g. (1, 1, 4) gives horizontal bedding.
    """
    stretch = np.ones(len(shape)) if stretch is None else np.asarray(stretch)
    total = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0

    for octave in range(octaves):
        cells = [
            int(min(s, max(2, round(base * 2**octave * k)))) + 1
            for s, k in zip(shape, stretch)
        ]
        total += amplitude * resample(
            rng.standard_normal(cells).astype(np.float32),
            shape,
        )
        amplitude *= persistence

    return norm(total)


def keep_largest_component(volume):
    labels, count = ndimage.label(volume, structure=SIX)

    if count <= 1:
        return volume

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def connected_components(volume):
    if not volume.any():
        return 0
    return int(ndimage.label(volume, structure=SIX)[1])


def surface_material(volume, outside=None):
    """Material voxels touching exterior air."""
    if outside is None:
        outside = exterior_air(volume)
    return volume & ndimage.binary_dilation(outside, structure=SIX)


def material_above(volume):
    """Number of material voxels above each voxel (z is up)."""
    above = np.cumsum(volume[:, :, ::-1], axis=2)[:, :, ::-1]
    return above - volume


def shelter(volume, size=5):
    return ndimage.uniform_filter(
        volume.astype(np.float32), size=size, mode="constant"
    )


def to_index(points, n):
    return (np.asarray(points) + 1.0) * 0.5 * (n - 1)


def stamp_capsule(volume, a, b, radius):
    """Rasterise a capsule between voxel-space points a and b."""
    n = volume.shape[0]
    lo = np.maximum(np.floor(np.minimum(a, b) - radius - 1), 0).astype(int)
    hi = np.minimum(np.ceil(np.maximum(a, b) + radius + 2), n).astype(int)

    if np.any(hi <= lo):
        return

    gx, gy, gz = np.meshgrid(
        np.arange(lo[0], hi[0]),
        np.arange(lo[1], hi[1]),
        np.arange(lo[2], hi[2]),
        indexing="ij",
    )
    p = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
    d = (b - a).astype(np.float32)
    denom = float(np.dot(d, d))
    t = 0.0 if denom < 1e-8 else np.clip(((p - a) @ d) / denom, 0, 1)[..., None]
    dist = np.linalg.norm(p - (a + t * d), axis=-1)
    volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= dist <= radius


# ============================================================
# Host forms for operators
# ============================================================

def host(X, Y, Z, rng):
    """A rock-like mass standing on the ground."""
    n = X.shape[0]
    kind = str(rng.choice(["outcrop", "boulder", "wall", "column"]))
    warp = fbm(X.shape, rng, base=2, octaves=3) - 0.5

    if kind == "outcrop":
        radius = rng.uniform(0.6, 0.9)
        height = fbm((n, n), rng, base=2, octaves=3)
        r = np.sqrt(X**2 + Y**2) + 0.35 * warp
        top = -1 + rng.uniform(0.7, 1.5) * (0.4 + 0.6 * height[:, :, None])
        volume = (r < radius) & (Z < top)

    elif kind == "boulder":
        axes = rng.uniform([0.6, 0.6, 0.45], [0.9, 0.9, 0.8])
        cz = -1 + axes[2] * rng.uniform(0.6, 0.9)
        d = np.sqrt(
            (X / axes[0]) ** 2 + (Y / axes[1]) ** 2 + ((Z - cz) / axes[2]) ** 2
        )
        volume = d + 0.5 * warp < 1.0

    elif kind == "wall":
        angle = rng.uniform(0, np.pi)
        normal = np.array([np.cos(angle), np.sin(angle)])
        along = -normal[1] * X + normal[0] * Y
        across = normal[0] * X + normal[1] * Y
        thickness = rng.uniform(0.35, 0.65)
        length = rng.uniform(0.6, 0.95)
        top = rng.uniform(0.3, 0.9) + 0.3 * warp
        volume = (
            (np.abs(across + 0.15 * warp) < thickness / 2)
            & (np.abs(along) < length)
            & (Z < top)
        )

    else:
        radius = rng.uniform(0.35, 0.6)
        lean = rng.normal(scale=0.15, size=2)
        profile = fbm((n,), rng, base=3, octaves=2)[None, None, :]
        r = np.sqrt(
            (X - lean[0] * (Z + 1)) ** 2 + (Y - lean[1] * (Z + 1)) ** 2
        )
        top = rng.uniform(0.4, 0.95)
        volume = (r + 0.25 * warp < radius * (0.7 + 0.6 * profile)) & (Z < top)

    volume[:, :, 0] |= volume[:, :, 1]
    return keep_largest_component(volume), {"host": kind}


# ============================================================
# Builders
# ============================================================

def branching(X, Y, Z, rng):
    """
    Space colonisation (Runions et al. 2007): branches grow toward free
    space, competing for it. Thickness follows the pipe model, where a
    parent's cross-section carries those of its children.
    """
    n = X.shape[0]
    envelope = str(rng.choice(["crown", "canopy", "column", "reach"]))

    centre = {
        "crown": [0, 0, rng.uniform(0.1, 0.4)],
        "canopy": [0, 0, rng.uniform(0.4, 0.6)],
        "column": [0, 0, 0.0],
        "reach": [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.0, 0.4)],
    }[envelope]
    radii = {
        "crown": rng.uniform([0.5, 0.5, 0.35], [0.85, 0.85, 0.6]),
        "canopy": rng.uniform([0.7, 0.7, 0.2], [0.95, 0.95, 0.35]),
        "column": rng.uniform([0.3, 0.3, 0.7], [0.45, 0.45, 0.95]),
        "reach": rng.uniform([0.6, 0.3, 0.3], [0.95, 0.5, 0.5]),
    }[envelope]

    count = int(rng.integers(400, 900))
    candidates = rng.uniform(-1, 1, size=(count * 4, 3))
    inside = (
        np.sum(((candidates - centre) / radii) ** 2, axis=1) < 1
    ) & np.all(np.abs(candidates) < 0.94, axis=1)
    attractors = candidates[inside][:count]

    step = rng.uniform(0.05, 0.08)
    influence = rng.uniform(0.3, 0.55)
    kill = rng.uniform(0.09, 0.15)
    tropism = rng.uniform(0.0, 0.4)

    roots = int(rng.integers(1, 4))
    nodes = []
    parents = []

    for _ in range(roots):
        nodes.append([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), -1.0])
        parents.append(-1)

    # Trunks rise until they sense the envelope.
    for i in range(roots):
        current = i
        for _ in range(60):
            p = np.asarray(nodes[current])
            if len(attractors) == 0 or np.min(
                np.linalg.norm(attractors - p, axis=1)
            ) < influence:
                break
            nodes.append((p + [0, 0, step]).tolist())
            parents.append(current)
            current = len(nodes) - 1

    for _ in range(140):
        if len(attractors) == 0:
            break

        points = np.asarray(nodes)
        distance, nearest = cKDTree(points).query(
            attractors, distance_upper_bound=influence
        )
        active = np.isfinite(distance)

        if not active.any():
            break

        growth = {}
        for a, node in zip(attractors[active], nearest[active]):
            direction = a - points[node]
            growth.setdefault(int(node), []).append(
                direction / max(np.linalg.norm(direction), 1e-8)
            )

        for node, directions in growth.items():
            direction = np.sum(directions, axis=0) + [0, 0, tropism]
            direction /= max(np.linalg.norm(direction), 1e-8)
            new = np.clip(points[node] + step * direction, -0.97, 0.97)
            nodes.append(new.tolist())
            parents.append(node)

        near = cKDTree(np.asarray(nodes)).query(attractors)[0] < kill
        attractors = attractors[~near]

    points = np.asarray(nodes)
    parents = np.asarray(parents)

    # Pipe model: r_parent^e = sum(r_child^e), accumulated tips-first.
    exponent = rng.uniform(2.0, 2.8)
    area = np.ones(len(points))
    for i in range(len(points) - 1, -1, -1):
        if parents[i] >= 0:
            area[parents[i]] += area[i]
    radius = area ** (1.0 / exponent)
    radius *= rng.uniform(0.13, 0.22) / radius.max()

    volume = np.zeros(X.shape, dtype=bool)
    voxel = to_index(points, n)
    scale = 0.5 * (n - 1)

    for i in range(len(points)):
        j = parents[i] if parents[i] >= 0 else i
        stamp_capsule(
            volume,
            voxel[j],
            voxel[i],
            max(1.0, radius[i] * scale),
        )

    return keep_largest_component(volume), {
        "envelope": envelope,
        "roots": roots,
        "nodes": int(len(points)),
        "pipe_exponent": float(exponent),
        "tropism": float(tropism),
    }


def clustering(X, Y, Z, rng):
    """
    Colonies seeded on the ground accrete toward light. Tips that reach
    open sky grow fastest, so neighbours compete, shade one another and
    form knobbed, coral-like clusters.
    """
    n = X.shape[0]
    colonies = int(rng.integers(3, 9))
    labels = np.zeros(X.shape, dtype=np.int16)

    for c in range(1, colonies + 1):
        i, j = rng.integers(int(0.15 * n), int(0.85 * n), size=2)
        labels[i - 1:i + 2, j - 1:j + 2, 0:2] = c

    vigour = rng.uniform(0.5, 1.0, colonies + 1)
    vigour[0] = 0.0
    light_power = rng.uniform(1.5, 4.0)
    rate = rng.uniform(0.35, 0.6)
    texture = fbm(X.shape, rng, base=4, octaves=3)
    lateral = rng.uniform(0.12, 0.35)
    iterations = int(rng.uniform(16, 28) * n / 32)

    for _ in range(iterations):
        occupied = labels > 0
        light = np.exp(-0.6 * material_above(occupied))
        front = ndimage.binary_dilation(occupied, structure=SIX) & ~occupied
        neighbour = ndimage.grey_dilation(labels, footprint=SIX)

        # Growth is supported from below: voxels resting on the colony
        # grow freely, sideways growth is slower, overhangs rarer.
        below = np.zeros_like(occupied)
        below[:, :, 1:] = occupied[:, :, :-1]
        support = np.where(below, 1.0, lateral)

        p = (
            rate
            * vigour[neighbour]
            * light ** light_power
            * support
            * (0.4 + texture)
        )
        grow = front & (rng.random(X.shape) < p)
        labels[grow] = neighbour[grow]

    volume = labels > 0

    return volume, {
        "colonies": colonies,
        "light_power": float(light_power),
        "lateral_growth": float(lateral),
        "iterations": iterations,
    }


def weather(volume, weakness, rng, iterations, mode, rate, direction=None):
    """
    Remove exposed surface voxels, one layer of chance per iteration.

    mode
      "differential": exposure * weakness (soft beds retreat)
      "salt":         shelter feedback; hollows deepen (tafoni)
      "rain":         sky-facing surfaces and drip lines
      "wind":         surfaces facing the wind, abraded by grit
    """
    for _ in range(iterations):
        outside = exterior_air(volume)
        surface = surface_material(volume, outside)

        if not surface.any():
            break

        cover = shelter(volume)

        if mode == "differential":
            p = rate * weakness**2 * (1.4 - cover)
        elif mode == "salt":
            p = rate * weakness * (0.08 + 2.2 * cover**3)
        elif mode == "rain":
            sky = material_above(volume) == 0
            drip = ndimage.binary_dilation(sky & surface, iterations=1)
            p = rate * weakness * (0.15 + 0.85 * drip)
        else:
            facing = np.roll(outside, -1, axis=direction[0])
            if direction[1] < 0:
                facing = np.roll(outside, 1, axis=direction[0])
            low = np.clip(1.0 - (np.arange(volume.shape[2]) / volume.shape[2]), 0.3, 1)
            p = rate * weakness * (0.1 + facing * low[None, None, :]) * (1.2 - cover)

        volume = volume & ~(surface & (rng.random(volume.shape) < p))

    return keep_largest_component(volume)


def incise(volume, rng, strength):
    """
    Gully incision on the top surface (stream power law): rain falling on
    the heightmap is routed downslope, and each column is cut in
    proportion to sqrt(catchment area) * slope.
    """
    n = volume.shape[0]
    filled = volume.any(axis=2)
    height = np.where(
        filled, n - np.argmax(volume[:, :, ::-1], axis=2), 0
    ).astype(np.float32)

    area = np.ones((n, n), dtype=np.float32)
    slope = np.zeros((n, n), dtype=np.float32)
    order = np.argsort(-height, axis=None)

    for flat in order:
        i, j = divmod(int(flat), n)
        if not filled[i, j]:
            continue
        best, drop = None, 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < n and 0 <= b < n:
                    d = (height[i, j] - height[a, b]) / (1.414 if di and dj else 1.0)
                    if d > drop:
                        best, drop = (a, b), d
        slope[i, j] = drop
        if best is not None:
            area[best] += area[i, j]

    cut = strength * np.sqrt(area) * np.minimum(slope, 3.0)
    new_height = np.maximum(height - np.round(cut), 1)
    z = np.arange(n)[None, None, :]

    return keep_largest_component(volume & (z < new_height[:, :, None]))


def layering(X, Y, Z, rng):
    """
    Strata deposited over time, folded and tilted, then weathered
    differentially: soft beds retreat under hard caps, leaving ledges,
    overhangs and recesses.
    """
    n = X.shape[0]
    beds = int(rng.integers(4, 10))
    top = rng.uniform(0.0, 0.85)
    thickness = rng.dirichlet(np.ones(beds) * 2.0) * (top + 1.0)

    dip = rng.normal(scale=0.12, size=2)
    fold = (fbm((n, n), rng, base=2, octaves=2) - 0.5) * rng.uniform(0.0, 0.5)
    interfaces = -1.0 + np.cumsum(thickness)

    surface_z = Z - (dip[0] * X + dip[1] * Y + fold[:, :, None])
    bed = np.searchsorted(interfaces, surface_z)

    hardness = rng.uniform(0.0, 1.0, beds + 1)
    hardness[1::2] = 1.0 - hardness[1::2] * 0.5
    hardness[0::2] *= 0.6
    weakness = 1.0 - hardness[np.minimum(bed, beds)]
    weakness = np.clip(
        weakness + 0.25 * (fbm(X.shape, rng, base=4, octaves=3) - 0.5), 0, 1
    )

    footprint = str(rng.choice(["mesa", "ridge", "butte"]))
    warp = fbm((n, n), rng, base=2, octaves=3)[:, :, None] - 0.5
    if footprint == "ridge":
        angle = rng.uniform(0, np.pi)
        r = np.abs(-np.sin(angle) * X + np.cos(angle) * Y) / rng.uniform(0.3, 0.5)
    else:
        size = 0.85 if footprint == "mesa" else 0.5
        r = np.sqrt(X**2 + Y**2) / rng.uniform(size - 0.15, size)
    volume = (r + 0.6 * warp < 1.0) & (bed < beds)
    volume = keep_largest_component(volume)

    iterations = int(rng.uniform(8, 20) * n / 32)
    volume = weather(
        volume, weakness, rng, iterations, "differential", rng.uniform(0.25, 0.45)
    )

    if rng.random() < 0.5:
        volume = incise(volume, rng, rng.uniform(0.1, 0.35))

    return volume, {
        "beds": beds,
        "footprint": footprint,
        "dip": [float(d) for d in dip],
        "weathering_iterations": iterations,
    }


# ============================================================
# Operators
# ============================================================

def erosion(volume, X, Y, Z, rng):
    """Weathering by one agent: rain, wind or salt."""
    n = X.shape[0]
    agent = str(rng.choice(["rain", "wind", "salt"]))
    weakness = fbm(
        X.shape, rng, base=3, octaves=4,
        stretch=(1, 1, rng.uniform(1, 4)),
    )
    duration = rng.uniform(0.3, 1.0)
    iterations = max(2, int(duration * 22 * n / 32))
    params = {"agent": agent, "duration": float(duration)}

    if agent == "rain":
        volume = incise(volume, rng, rng.uniform(0.15, 0.5))
        volume = weather(volume, weakness, rng, iterations, "rain", rng.uniform(0.2, 0.35))
    elif agent == "wind":
        direction = (int(rng.integers(0, 2)), int(rng.choice([-1, 1])))
        params["direction"] = list(direction)
        volume = weather(
            volume, weakness, rng, iterations, "wind", rng.uniform(0.25, 0.4), direction
        )
    else:
        volume = weather(volume, weakness, rng, iterations, "salt", rng.uniform(0.2, 0.32))

    return volume, params


def cavitation(volume, X, Y, Z, rng):
    """
    Karst dissolution. Water enters at sky-facing surfaces and percolates
    down, dissolving soluble rock as it goes: vertical shafts, solution
    pans that hold water, and caves where it spreads along a bed.
    """
    n = X.shape[0]
    solubility = fbm(X.shape, rng, base=3, octaves=3, stretch=(1, 1, 3))
    joints = np.zeros(X.shape, dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        normal = unit_vector(rng)
        normal[2] *= 0.3
        plane = X * normal[0] + Y * normal[1] + Z * normal[2] - rng.uniform(-0.4, 0.4)
        joints = np.maximum(joints, np.exp(-(plane / 0.06) ** 2))
    solubility = np.clip(0.6 * solubility + 0.6 * joints, 0, 1)
    # Insoluble beds stop downward dissolution, so water pools on them.
    beds = fbm(X.shape, rng, base=2, octaves=2, stretch=(0.5, 0.5, 4))
    solubility[beds > rng.uniform(0.55, 0.7)] *= 0.05

    aggressiveness = rng.uniform(0.25, 0.55)
    walkers = int(rng.uniform(120, 320) * (n / 32) ** 2)
    volume = volume.copy()
    lateral = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    tops = None

    for w in range(walkers):
        # Entry points: sky-facing surface. Refreshed as the rock changes.
        if w % 25 == 0:
            tops = np.argwhere(
                surface_material(volume) & (material_above(volume) == 0)
            )
        if len(tops) == 0:
            break
        x, y, z = tops[int(rng.integers(len(tops)))]
        z = min(z + 1, n - 1)
        pooled = 0

        for _ in range(3 * n):
            if z == 0:
                break
            if not volume[x, y, z - 1]:
                z -= 1
                continue
            if rng.random() < aggressiveness * solubility[x, y, z - 1]:
                volume[x, y, z - 1] = False
                z -= 1
                continue
            dx, dy = lateral[int(rng.integers(4))]
            a, b = x + dx, y + dy
            if not (0 <= a < n and 0 <= b < n):
                break
            if not volume[a, b, z]:
                x, y = a, b
            elif rng.random() < 0.5 * aggressiveness * solubility[a, b, z]:
                volume[a, b, z] = False
                x, y = a, b
            else:
                # Trapped water pools and keeps dissolving its basin,
                # deepening and widening a solution pan.
                pooled += 1
                if pooled > 6:
                    break
                d = int(rng.integers(5))
                if d < 4:
                    a, b = x + lateral[d][0], y + lateral[d][1]
                    if 0 <= a < n and 0 <= b < n:
                        volume[a, b, z] = False
                else:
                    volume[x, y, z - 1] = False

    return keep_largest_component(volume), {
        "walkers": walkers,
        "aggressiveness": float(aggressiveness),
    }


GRAY_SCOTT = {
    "vesicular": (0.0367, 0.0649, 0.20),
    "spongy": (0.042, 0.059, 0.22),
    "sparse": (0.054, 0.063, 0.12),
}


def gray_scott(regime, rng, size=32, iterations=1200):
    feed, kill, threshold = GRAY_SCOTT[regime]
    u = np.ones((size,) * 3, dtype=np.float32)
    v = np.zeros_like(u)
    seeds = ndimage.binary_dilation(rng.random(u.shape) < 0.002, iterations=2)
    v[seeds], u[seeds] = 0.5, 0.25
    v += 0.02 * rng.random(u.shape).astype(np.float32)

    def laplacian(a):
        return sum(np.roll(a, s, axis) for axis in range(3) for s in (1, -1)) - 6 * a

    for _ in range(iterations):
        uvv = u * v * v
        u += 0.16 * laplacian(u) - uvv + feed * (1 - u)
        v += 0.08 * laplacian(v) + uvv - (feed + kill) * v

    return v, threshold


def porosity(volume, X, Y, Z, rng):
    """
    Pores from Gray-Scott reaction-diffusion: depending on feed and kill
    rates the reacting species settles into isolated vesicles or a
    sponge of interconnected tunnels. Simulated at 32^3 and resampled,
    so pore size stays fixed relative to the form at any resolution.
    """
    n = X.shape[0]
    regime = str(rng.choice(list(GRAY_SCOTT)))
    v, threshold = gray_scott(regime, rng)
    if n != v.shape[0]:
        v = resample(v, X.shape)

    pores = v > threshold
    depth = "surface" if rng.random() < 0.7 else "through"
    if depth == "surface":
        pores &= ndimage.distance_transform_edt(volume) < rng.uniform(2, 4) * n / 32

    return keep_largest_component(volume & ~pores), {
        "regime": regime,
        "depth": depth,
    }


def fragmentation(volume, X, Y, Z, rng):
    """
    Jointed rock: a Voronoi block structure (anisotropic for columnar or
    sheeted jointing). Cracks open from the surface inward, as frost and
    roots wedge them, and some surface blocks spall off entirely.
    """
    n = X.shape[0]
    jointing = str(rng.choice(["blocky", "columnar", "sheeted"]))
    stretch = {
        "blocky": [1, 1, 1],
        "columnar": [1, 1, 0.3],
        "sheeted": [1, 1, 2.5],
    }[jointing]
    blocks = int(rng.integers(15, 50))

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) * stretch
    seeds = rng.uniform(-1, 1, size=(blocks, 3)) * stretch
    distance, owner = cKDTree(seeds).query(points, k=2)
    gap = (distance[:, 1] - distance[:, 0]).reshape(X.shape)
    block = owner[:, 0].reshape(X.shape)

    width = rng.uniform(0.6, 1.4) * 2.0 / n
    depth = ndimage.distance_transform_edt(volume)
    reach = rng.uniform(1.5, 4) * n / 32
    # Cracks open in patches, leaving rock bridges between blocks.
    patches = fbm(X.shape, rng, base=3, octaves=2) > rng.uniform(0.35, 0.55)
    cracks = (gap < width) & (depth < reach) & patches
    volume = volume & ~cracks

    spall = rng.uniform(0.03, 0.15)
    exposed = np.unique(block[surface_material(volume)])
    falling = exposed[rng.random(len(exposed)) < spall]
    volume &= ~np.isin(block, falling)

    return keep_largest_component(volume), {
        "jointing": jointing,
        "blocks": blocks,
        "spalled_blocks": int(len(falling)),
    }


BUILDER_FUNCTIONS = {
    "branching": branching,
    "clustering": clustering,
    "layering": layering,
}

OPERATOR_FUNCTIONS = {
    "erosion": erosion,
    "cavitation": cavitation,
    "porosity": porosity,
    "fragmentation": fragmentation,
}


def simulate(names, X, Y, Z, rng):
    """
    Run a list of phenomena: builders first (united), then operators in
    the given order. With no builder, operators act on a host form.
    """
    volume = np.zeros(X.shape, dtype=bool)
    processes = []

    builders = [name for name in names if name in BUILDERS]
    operators = [name for name in names if name not in BUILDERS]

    for name in builders:
        built, params = BUILDER_FUNCTIONS[name](X, Y, Z, rng)
        volume |= built
        processes.append({"phenomenon": name, "parameters": params})

    if not builders:
        volume, params = host(X, Y, Z, rng)
        processes.append({"phenomenon": "host", "parameters": params})

    for name in operators:
        volume, params = OPERATOR_FUNCTIONS[name](volume, X, Y, Z, rng)
        processes.append({"phenomenon": name, "parameters": params})

    return volume, processes


def generate(phenomenon, X, Y, Z, rng):
    if phenomenon == "hybridization":
        names = list(rng.choice(PHENOMENA, size=int(rng.integers(2, 4)), replace=False))
    else:
        names = [phenomenon]

    volume, processes = simulate(names, X, Y, Z, rng)

    return volume, {"phenomena": [str(n) for n in names], "processes": processes}


def valid(volume, phenomenon, phenomena):
    occupancy = float(volume.mean())
    components = connected_components(volume)

    ok = 0.02 <= occupancy <= 0.6

    if "clustering" in phenomena:
        ok &= 1 <= components <= 12
    else:
        ok &= components == 1

    return bool(ok), {
        "occupancy": occupancy,
        "connected_components": components,
    }


# ============================================================
# Dataset assembly
# ============================================================

def to_tsdf(volume, truncation, smoothing):
    """
    Truncated signed distance field in [-1, 1], positive inside material.

    The surface sits at 0, half a voxel from the centres of the boundary
    voxels, so marching cubes at level 0 reproduces the binary surface.
    A light Gaussian on the distance field rounds voxel stair-steps.
    """
    inside = ndimage.distance_transform_edt(volume)
    outside = ndimage.distance_transform_edt(~volume)

    signed = np.where(
        volume,
        inside - 0.5,
        -(outside - 0.5),
    ).astype(np.float32)

    if smoothing > 0:
        signed = ndimage.gaussian_filter(
            signed,
            smoothing,
        )

    return np.clip(
        signed / truncation,
        -1.0,
        1.0,
    )


def build_labels(dataset_size, rng):
    """About 70% single-phenomenon samples, 30% hybrids."""
    single_count = int(
        round(dataset_size * 0.70)
    )

    labels = (
        PHENOMENA
        * (single_count // len(PHENOMENA))
    )

    labels += PHENOMENA[
        : single_count % len(PHENOMENA)
    ]

    labels += [
        "hybridization"
    ] * (
        dataset_size
        - single_count
    )

    rng.shuffle(labels)

    return labels


_GRID = None


def _init_worker(resolution):
    global _GRID
    _GRID = grid(resolution)


def make_sample(job):
    """Generate, validate and describe one sample. Runs in a worker."""
    index, phenomenon, seed = job
    X, Y, Z = _GRID

    last_validation = None

    for attempt in range(100):
        rng = np.random.default_rng(
            seed + attempt
        )

        volume, parameters = generate(
            phenomenon,
            X,
            Y,
            Z,
            rng,
        )

        ok, validation = valid(
            volume,
            phenomenon,
            parameters["phenomena"],
        )

        last_validation = validation

        if ok:
            break
    else:
        raise RuntimeError(
            "Could not generate a valid sample after 100 attempts. "
            f"Sample={index}, phenomenon={phenomenon}, "
            f"last_validation={last_validation}"
        )

    volume = volume.astype(bool)

    sdf = to_tsdf(
        volume,
        CONFIG.tsdf_truncation,
        CONFIG.tsdf_smoothing,
    )

    metrics = compute_metrics(volume)

    record = {
        "id": index,
        "seed": seed,
        "attempt": attempt,
        "phenomenon": phenomenon,
        "phenomena": parameters["phenomena"],
        "parameters": parameters,
        "validation": validation,
        "metrics": metrics,
    }

    return index, volume, sdf, record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the procedural ecological morphology dataset."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=CONFIG.dataset_size,
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=CONFIG.resolution,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.seed,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel processes (1 = no multiprocessing)",
    )

    parser.add_argument(
        "--output",
        default=CONFIG.dataset_path,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    resolution = int(args.resolution)
    dataset_size = int(args.count)

    master = np.random.default_rng(
        int(args.seed)
    )

    labels = build_labels(
        dataset_size,
        master,
    )

    jobs = [
        (
            index,
            phenomenon,
            int(master.integers(0, 2**32 - 1)),
        )
        for index, phenomenon in enumerate(labels)
    ]

    structures = np.zeros(
        (dataset_size, resolution, resolution, resolution, 1),
        dtype=np.uint8,
    )

    sdf = np.zeros(
        (dataset_size, resolution, resolution, resolution, 1),
        dtype=np.float16,
    )

    metadata = [None] * dataset_size

    progress = tqdm(
        total=dataset_size,
        desc="Generating ecological morphology dataset",
    )

    def collect(result):
        index, volume, field, record = result
        structures[index, ..., 0] = volume
        sdf[index, ..., 0] = field
        metadata[index] = record
        progress.update(1)

    if args.workers <= 1:
        _init_worker(resolution)

        for job in jobs:
            collect(make_sample(job))
    else:
        with Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(resolution,),
        ) as pool:
            for result in pool.imap_unordered(
                make_sample,
                jobs,
                chunksize=8,
            ):
                collect(result)

    progress.close()

    # Multi-hot phenomenon labels: hybrids switch on every process used.
    phenomenon_labels = np.zeros(
        (dataset_size, len(PHENOMENA)),
        dtype=np.float32,
    )

    for record in metadata:
        for name in record["phenomena"]:
            phenomenon_labels[
                record["id"],
                PHENOMENA.index(name),
            ] = 1.0

    metric_arrays = {
        f"metric_{name}": np.array(
            [record["metrics"][name] for record in metadata],
            dtype=np.float32,
        )
        for name in METRIC_NAMES
    }

    # structures: binary occupancy (uint8, 0/1), kept for inspection tools.
    # sdf: truncated signed distance in [-1, 1], positive inside. GAN target.
    np.savez_compressed(
        args.output,
        structures=structures,
        sdf=sdf,
        labels=phenomenon_labels,
        phenomena=np.array(PHENOMENA),
        sample_phenomenon=np.array(
            [record["phenomenon"] for record in metadata]
        ),
        metric_names=np.array(METRIC_NAMES),
        tsdf_truncation=np.float32(CONFIG.tsdf_truncation),
        **metric_arrays,
    )

    metadata_path = (
        os.path.splitext(args.output)[0]
        + "_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("Wrote", args.output)
    print("Shape:", sdf.shape)
    print(
        "Mean occupancy:",
        float(structures.mean()),
    )

    print()
    print("Metric ranges (10th / 50th / 90th percentile):")

    for name in METRIC_NAMES:
        values = metric_arrays[f"metric_{name}"]
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        print(f"  {name:24s} {p10:8.3f} {p50:8.3f} {p90:8.3f}")

    print()
    print("Wrote", metadata_path)


if __name__ == "__main__":
    main()
