import argparse
import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from config import CONFIG
from generate_dataset import PHENOMENA, build_labels


# Surface detail dataset
# ----------------------
# High-resolution height maps of weathered surfaces, the fine scale that
# the 3D volumes are too coarse to hold. Each map is the result of a 2D
# process simulation, labelled with the same phenomena as the volumes so
# detail can be matched to form:
#
#   branching     -> drainage / root networks from flow accumulation
#   clustering    -> colonies (lichen, coral) accreting into domes
#   layering      -> cross-bedded strata, soft beds recessed
#   erosion       -> rain droplets carving rills, or salt hollowing
#                    pits into honeycomb (tafoni)
#   cavitation    -> karst solution pits and runnels
#   porosity      -> Gray-Scott reaction-diffusion vesicles
#   fragmentation -> jointed blocks, open cracks, spalled blocks
#
# Every map is periodic (seamlessly tileable), so it can wrap around a
# form without seams. Values lie in [-1, 1]: positive stands proud of the
# surface, negative is carved in.


# ============================================================
# Periodic field helpers
# ============================================================

def spectral_noise(n, rng, beta=2.0, stretch=(1.0, 1.0)):
    """Periodic fractal noise by spectral synthesis, zero mean, unit std."""
    kx = np.fft.fftfreq(n)[:, None] * stretch[0]
    ky = np.fft.fftfreq(n)[None, :] * stretch[1]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amplitude = k ** (-beta / 2)
    amplitude[0, 0] = 0.0
    field = np.real(
        np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * amplitude)
    )
    return ((field - field.mean()) / (field.std() + 1e-8)).astype(np.float32)


def blur(h, sigma):
    return ndimage.gaussian_filter(h, sigma, mode="wrap")


def laplacian(h):
    return (
        np.roll(h, 1, 0) + np.roll(h, -1, 0)
        + np.roll(h, 1, 1) + np.roll(h, -1, 1) - 4 * h
    )


def sample(field, x, y):
    """Bilinear, wrapping lookup at fractional pixel coordinates."""
    return ndimage.map_coordinates(field, [x, y], order=1, mode="grid-wrap")


def coordinates(n):
    a = np.arange(n, dtype=np.float32)
    return np.meshgrid(a, a, indexing="ij")


def finish(h):
    """Zero mean, scaled into [-1, 1]."""
    h = h - np.median(h)
    return (h / (np.abs(h).max() + 1e-8)).astype(np.float32)


# ============================================================
# Processes
# ============================================================

def erosion(n, rng):
    """
    rain: droplets run downhill, picking up sediment where they speed up
          and dropping it where they slow, carving rills and gullies.
    salt: hollows erode fastest because salt crystallises in their
          shelter; pits deepen and meet in thin walls (tafoni, honeycomb).
    """
    agent = str(rng.choice(["rain", "salt"]))

    if agent == "salt":
        h = spectral_noise(n, rng, beta=rng.uniform(1.0, 1.6))
        hardness = 0.5 + 0.5 * np.tanh(spectral_noise(n, rng, beta=2.5))
        scale = rng.uniform(2.0, 4.0)
        floor = h.mean() - rng.uniform(1.5, 3.0)
        for _ in range(int(rng.integers(40, 80))):
            hollow = np.maximum(blur(h, scale) - h, 0)
            hollow /= hollow.std() + 1e-8
            # Feedback saturates, and deep pits slow as they lose exposure,
            # so neighbouring hollows grow until they meet in thin walls.
            slowing = np.clip((h - floor) / 1.5, 0, 1)
            h -= 0.06 * (0.2 + np.tanh(hollow)) * (1.3 - hardness) * slowing
            h = 0.9 * h + 0.1 * blur(h, 0.6)
        return finish(h), {"agent": agent, "cell_scale": float(scale)}

    h = spectral_noise(n, rng, beta=rng.uniform(2.0, 2.8))
    h *= rng.uniform(4, 10)
    tilt = rng.normal(size=2) * rng.uniform(0.0, 0.06)
    x, y = coordinates(n)
    # A periodic tilt: water drains in one direction overall.
    h += tilt[0] * n / (2 * np.pi) * np.sin(2 * np.pi * x / n)
    h += tilt[1] * n / (2 * np.pi) * np.sin(2 * np.pi * y / n)

    drops = 4000
    inertia = rng.uniform(0.05, 0.3)
    capacity = rng.uniform(4, 10)
    erode, deposit, evaporate = 0.3, 0.3, 0.02

    for _ in range(int(rng.integers(6, 12))):
        px = rng.uniform(0, n, drops)
        py = rng.uniform(0, n, drops)
        dx = np.zeros(drops)
        dy = np.zeros(drops)
        speed = np.ones(drops)
        water = np.ones(drops)
        sediment = np.zeros(drops)
        change = np.zeros_like(h)

        gx = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
        gy = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5

        for _ in range(30):
            ix, iy = px.astype(int) % n, py.astype(int) % n
            old = h[ix, iy] + change[ix, iy]
            dx = inertia * dx - (1 - inertia) * gx[ix, iy]
            dy = inertia * dy - (1 - inertia) * gy[ix, iy]
            length = np.sqrt(dx**2 + dy**2) + 1e-8
            dx, dy = dx / length, dy / length
            px, py = (px + dx) % n, (py + dy) % n
            jx, jy = px.astype(int) % n, py.astype(int) % n
            drop = h[jx, jy] + change[jx, jy] - old

            limit = np.maximum(-drop, 0.01) * speed * water * capacity
            settle = (sediment > limit) | (drop > 0)
            amount = np.where(
                settle,
                np.where(drop > 0, np.minimum(drop, sediment), (sediment - limit) * deposit),
                -np.minimum(np.minimum((limit - sediment) * erode, -drop), 0.3),
            )
            np.add.at(change, (ix, iy), amount)
            sediment -= amount
            speed = np.clip(np.sqrt(np.maximum(speed**2 - drop * 4.0, 0.01)), 0.1, 3.0)
            water *= 1 - evaporate

        h += blur(change, 0.6)

    return finish(h), {"agent": agent, "inertia": float(inertia)}


def layering(n, rng):
    """
    Cross-bedded strata: bedding planes warped by the currents that laid
    them down, then weathered so soft beds recess behind hard ledges.
    """
    x, y = coordinates(n)
    angle = rng.uniform(0, 2 * np.pi)
    # Integer wave numbers keep the bedding periodic.
    kx, ky = np.round(rng.uniform(1, 4) * np.array([np.cos(angle), np.sin(angle)]))
    if kx == 0 and ky == 0:
        kx = 1
    beds = rng.uniform(4, 14)
    warp = spectral_noise(n, rng, beta=rng.uniform(2.6, 3.4)) * rng.uniform(0.3, 1.2)
    phase = (kx * x + ky * y) / n + warp / beds
    bed = phase * beds

    index = np.floor(bed).astype(int)
    hardness_table = rng.uniform(0, 1, 64)
    hardness = hardness_table[index % 64]
    within = bed - np.floor(bed)
    # Ledges are rounded where hard beds overhang soft ones.
    profile = hardness - 0.35 * (1 - hardness) * np.sin(np.pi * within)
    h = blur(profile, rng.uniform(0.6, 1.4))
    h += 0.12 * spectral_noise(n, rng, beta=1.4) * (1.2 - hardness)

    return finish(h), {"beds": float(beds)}


def cavitation(n, rng):
    """
    Karst on an exposed surface: solution pits that deepen where water
    stands, linked by runnels (rillenkarren) running downslope.
    """
    x, y = coordinates(n)
    h = 0.15 * spectral_noise(n, rng, beta=2.2)

    pits = int(rng.integers(6, 30))
    for _ in range(pits):
        cx, cy = rng.uniform(0, n, 2)
        radius = rng.uniform(3, 14)
        ddx = (x - cx + n / 2) % n - n / 2
        ddy = (y - cy + n / 2) % n - n / 2
        r = np.sqrt(ddx**2 + ddy**2) / radius
        # Flat floor, steep walls: a solution pan.
        h = np.minimum(h, -rng.uniform(0.4, 1.0) * np.clip(1.3 - r**4, 0, 1) + 0.3)

    if rng.random() < 0.7:
        angle = rng.uniform(0, 2 * np.pi)
        k = np.round(rng.uniform(8, 20))
        across = (np.cos(angle) * x + np.sin(angle) * y) * k / n
        across += spectral_noise(n, rng, beta=3.0) * 0.4
        h -= rng.uniform(0.1, 0.3) * (0.5 + 0.5 * np.cos(2 * np.pi * across))

    return finish(blur(h, 0.8)), {"pits": pits}


GRAY_SCOTT_2D = {
    "vesicular": (0.0367, 0.0649),
    "spongy": (0.042, 0.059),
    "sparse": (0.054, 0.063),
}


def porosity(n, rng):
    """Gray-Scott reaction-diffusion pores, simulated at 64^2 and resampled."""
    regime = str(rng.choice(list(GRAY_SCOTT_2D)))
    feed, kill = GRAY_SCOTT_2D[regime]
    size = 64
    u = np.ones((size, size), dtype=np.float32)
    v = np.zeros_like(u)
    seeds = ndimage.binary_dilation(rng.random(u.shape) < 0.01, iterations=1)
    v[seeds], u[seeds] = 0.5, 0.25
    v += 0.02 * rng.random(u.shape).astype(np.float32)

    for _ in range(2500):
        uvv = u * v * v
        u += 0.2 * laplacian(u) - uvv + feed * (1 - u)
        v += 0.1 * laplacian(v) + uvv - (feed + kill) * v

    x, y = coordinates(n)
    pores = sample(v, x * size / n, y * size / n)
    h = -pores + 0.1 * spectral_noise(n, rng, beta=1.8)

    return finish(blur(h, 0.7)), {"regime": regime}


def fragmentation(n, rng):
    """Jointed blocks: open cracks, tilted blocks, some spalled away."""
    x, y = coordinates(n)
    blocks = int(rng.integers(6, 30))
    seeds = rng.uniform(0, n, size=(blocks, 2))
    stretch = rng.uniform(0.5, 1.0)

    d1 = np.full(x.shape, np.inf, dtype=np.float32)
    d2 = np.full(x.shape, np.inf, dtype=np.float32)
    owner = np.zeros(x.shape, dtype=int)
    for i, (sx, sy) in enumerate(seeds):
        ddx = (x - sx + n / 2) % n - n / 2
        ddy = ((y - sy + n / 2) % n - n / 2) * stretch
        d = np.sqrt(ddx**2 + ddy**2)
        closer = d < d1
        d2 = np.where(closer, d1, np.minimum(d2, d))
        owner = np.where(closer, i, owner)
        d1 = np.where(closer, d, d1)

    gap = d2 - d1
    width = rng.uniform(1.0, 3.0)
    h = -np.exp(-(gap / width) ** 2)

    tilt = rng.normal(scale=0.01, size=(blocks, 2))
    level = rng.normal(scale=0.15, size=blocks)
    level[rng.random(blocks) < rng.uniform(0.05, 0.25)] -= 0.6
    ddx = (x - seeds[owner, 0] + n / 2) % n - n / 2
    ddy = (y - seeds[owner, 1] + n / 2) % n - n / 2
    h += level[owner] + tilt[owner, 0] * ddx + tilt[owner, 1] * ddy
    # Block edges round over as they weather.
    h -= 0.25 * np.exp(-gap / (2 * width))
    h += 0.06 * spectral_noise(n, rng, beta=1.6)

    return finish(blur(h, 0.6)), {"blocks": blocks}


def clustering(n, rng):
    """Colonies spreading from seeds, each a knobbed dome."""
    texture = spectral_noise(n, rng, beta=1.6)
    colonies = int(rng.integers(8, 30))
    occupied = np.zeros((n, n), dtype=bool)
    for cx, cy in rng.integers(0, n, size=(colonies, 2)):
        occupied[cx, cy] = True

    cross = ndimage.generate_binary_structure(2, 1)
    for _ in range(int(rng.integers(18, 40) * n / 128)):
        padded = np.pad(occupied, 1, mode="wrap")
        front = ndimage.binary_dilation(padded, structure=cross)[1:-1, 1:-1] & ~occupied
        occupied |= front & (rng.random(occupied.shape) < 0.35 + 0.25 * np.tanh(texture))

    tiled = np.tile(occupied, (3, 3))
    depth = ndimage.distance_transform_edt(tiled)[n:2 * n, n:2 * n]
    h = np.sqrt(depth) * (1 + 0.25 * texture)
    h += 0.15 * occupied * spectral_noise(n, rng, beta=1.2)
    h += 0.1 * h.max() * spectral_noise(n, rng, beta=2.0) * ~occupied

    return finish(blur(h, 0.8)), {"colonies": colonies}


def branching(n, rng):
    """
    A dendritic network from flow accumulation: water from every cell is
    routed downhill, and channels form where enough of it gathers. Cut as
    rills, or raised as roots and veins.
    """
    h = spectral_noise(n, rng, beta=rng.uniform(3.0, 3.8))
    x, y = coordinates(n)
    angle = rng.uniform(0, 2 * np.pi)
    # A smooth periodic slope so water drains instead of pooling.
    h += rng.uniform(1.5, 3.0) * (
        np.cos(angle) * np.sin(2 * np.pi * x / n)
        + np.sin(angle) * np.sin(2 * np.pi * y / n)
    )
    h += 0.02 * spectral_noise(n, rng, beta=1.0)
    area = np.ones_like(h)
    order = np.argsort(-h, axis=None)
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for flat in order:
        i, j = divmod(int(flat), n)
        best, drop = None, 0.0
        for di, dj in steps:
            a, b = (i + di) % n, (j + dj) % n
            d = h[i, j] - h[a, b]
            if d > drop:
                best, drop = (a, b), d
        if best is not None:
            area[best] += area[i, j]

    channels = np.log1p(area)
    channels = np.where(area > rng.uniform(8, 25), channels, 0)
    channels = blur(channels, rng.uniform(0.6, 1.2))
    raised = rng.random() < 0.4
    h = (1 if raised else -1) * channels + 0.15 * h

    return finish(h), {"raised": bool(raised)}


PROCESSES = {
    "branching": branching,
    "cavitation": cavitation,
    "porosity": porosity,
    "erosion": erosion,
    "layering": layering,
    "clustering": clustering,
    "fragmentation": fragmentation,
}


def generate_texture(phenomenon, n, rng):
    if phenomenon == "hybridization":
        names = list(rng.choice(PHENOMENA, size=2, replace=False))
        first, _ = PROCESSES[names[0]](n, rng)
        second, _ = PROCESSES[names[1]](n, rng)
        mix = rng.uniform(0.3, 0.7)
        return finish(mix * first + (1 - mix) * second), names

    h, _ = PROCESSES[phenomenon](n, rng)
    return h, [phenomenon]


# ============================================================
# Dataset assembly
# ============================================================

def make_texture(job):
    index, phenomenon, seed, size = job
    rng = np.random.default_rng(seed)
    h, names = generate_texture(phenomenon, size, rng)
    return index, h, names


def save_gallery(textures, names, path):
    """Hill-shaded preview of a handful of height maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    count = min(len(textures), 16)
    figure, axes = plt.subplots(2, (count + 1) // 2, figsize=(2.6 * ((count + 1) // 2), 5.6))
    shade = LightSource(azdeg=315, altdeg=40)

    for axis, h, name in zip(axes.ravel(), textures, names):
        axis.imshow(shade.shade(h, cmap=plt.cm.gist_earth, vert_exag=12, blend_mode="soft"))
        axis.set_title("+".join(name), fontsize=8)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=80)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the surface-detail height map dataset."
    )
    parser.add_argument("--count", type=int, default=CONFIG.texture_dataset_size)
    parser.add_argument("--size", type=int, default=CONFIG.texture_size)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--output", default=CONFIG.texture_dataset_path)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    master = np.random.default_rng(args.seed)
    labels = build_labels(args.count, master)
    jobs = [
        (index, phenomenon, int(master.integers(0, 2**32 - 1)), args.size)
        for index, phenomenon in enumerate(labels)
    ]

    textures = np.zeros((args.count, args.size, args.size, 1), dtype=np.float16)
    multi_hot = np.zeros((args.count, len(PHENOMENA)), dtype=np.float32)
    names = [None] * args.count

    progress = tqdm(total=args.count, desc="Generating surface textures")

    def collect(result):
        index, h, used = result
        textures[index, ..., 0] = h
        names[index] = used
        for name in used:
            multi_hot[index, PHENOMENA.index(name)] = 1.0
        progress.update(1)

    if args.workers <= 1:
        for job in jobs:
            collect(make_texture(job))
    else:
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(make_texture, jobs, chunksize=4):
                collect(result)

    progress.close()

    np.savez_compressed(
        args.output,
        textures=textures,
        labels=multi_hot,
        phenomena=np.array(PHENOMENA),
        sample_phenomenon=np.array(labels),
    )

    gallery = os.path.splitext(args.output)[0] + "_gallery.png"
    picks = np.random.default_rng(0).choice(args.count, size=min(16, args.count), replace=False)
    save_gallery(
        [textures[i, ..., 0].astype(np.float32) for i in picks],
        [names[i] for i in picks],
        gallery,
    )

    print()
    print("Wrote", args.output, textures.shape)
    print("Wrote", gallery)


if __name__ == "__main__":
    main()
