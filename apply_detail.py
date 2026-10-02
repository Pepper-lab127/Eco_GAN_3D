import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

from export_obj import volume_to_mesh, write_obj
from generate_dataset import PHENOMENA


# Surface detail
# --------------
# The 3D GAN gives the form; the texture GAN (or the procedural texture
# processes directly) gives the fine weathering. This script joins them:
#
#   1. upsample the generated signed distance field to a fine grid
#   2. project tileable height maps onto it from three directions
#      (triplanar mapping, blended by surface orientation)
#   3. push the surface in and out by the height, then mesh
#
# Detail is carved into the field before meshing, so pits, ridges and
# grooves are real geometry, not a texture painted on a smooth mesh.


def upsample(sdf, resolution, truncation):
    """
    Cubic resampling of a coarse truncated SDF to resolution^3, returned
    in voxel units of the fine grid (positive inside).
    """
    n = sdf.shape[0]
    axis = np.linspace(0, n - 1, resolution)
    coords = np.meshgrid(axis, axis, axis, indexing="ij")
    fine = ndimage.map_coordinates(sdf, coords, order=3, mode="nearest")
    return (fine * truncation * (resolution - 1) / (n - 1)).astype(np.float32)


def triplanar(fine, maps, tile, weathering):
    """
    Height at every voxel, blending three projected maps by how much the
    surface faces each axis. weathering > 0 strengthens detail on
    surfaces exposed to the sky (upward facing), as rain and frost would.
    """
    r = fine.shape[0]
    gx, gy, gz = np.gradient(fine)
    length = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
    # The SDF is positive inside, so the outward normal is -gradient.
    nx, ny, nz = -gx / length, -gy / length, -gz / length

    weights = np.stack([np.abs(nx), np.abs(ny), np.abs(nz)]) ** 4
    weights /= weights.sum(axis=0) + 1e-8

    i, j, k = np.meshgrid(*(np.arange(r, dtype=np.float32),) * 3, indexing="ij")
    height = np.zeros_like(fine)

    for axis, (u, v) in enumerate([(j, k), (i, k), (i, j)]):
        texture = maps[axis]
        scale = texture.shape[0] * tile / r
        projected = ndimage.map_coordinates(
            texture,
            [u * scale, v * scale],
            order=1,
            mode="grid-wrap",
        )
        height += weights[axis] * projected

    if weathering > 0:
        height *= 1.0 + weathering * np.clip(nz, 0, 1)

    return height


def keep_largest(fine):
    labels, count = ndimage.label(fine > 0)
    if count <= 1:
        return fine
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = labels == int(np.argmax(sizes))
    # Detached crumbs become empty space.
    return np.where(keep | (fine <= 0), fine, -np.abs(fine))


# ============================================================
# Texture sources
# ============================================================

def latest_texture_checkpoint(directory):
    paths = sorted(glob.glob(os.path.join(directory, "texture_ema_*.weights.h5")))
    return paths[-1] if paths else None


def gan_textures(checkpoint, phenomena, count, seed):
    from models import load_weights
    from texture_models import build_texture_generator

    stats_path = os.path.join(os.path.dirname(checkpoint), "texture_stats.json")
    with open(stats_path, "r", encoding="utf-8") as file:
        stats = json.load(file)

    generator = build_texture_generator(
        resolution=stats["resolution"],
        latent_dim=stats["latent_dim"],
        condition_dim=len(stats["phenomena"]),
    )
    load_weights(generator, checkpoint)

    condition = np.zeros((count, len(stats["phenomena"])), dtype=np.float32)
    for name in phenomena:
        condition[:, stats["phenomena"].index(name)] = 1.0

    z = np.random.default_rng(seed).normal(
        size=(count, stats["latent_dim"])
    ).astype(np.float32)

    return list(generator([z, condition], training=False).numpy()[..., 0])


def procedural_textures(phenomena, count, seed, size):
    from generate_textures import PROCESSES, finish

    rng = np.random.default_rng(seed)
    maps = []
    for _ in range(count):
        layers = [PROCESSES[name](size, rng)[0] for name in phenomena]
        maps.append(finish(np.mean(layers, axis=0)))
    return maps


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add fine weathering detail to a generated structure and export "
            "a high-resolution OBJ. Example: python apply_detail.py --input "
            "outputs/generated_0000.npy --amplitude 2.5"
        )
    )
    parser.add_argument("--input", required=True, help="Generated .npy volume")
    parser.add_argument("--output", default=None, help="OBJ path (default: <input>_detail.obj)")
    parser.add_argument(
        "--phenomena",
        default=None,
        help="Comma-separated texture phenomena (default: read from the "
        "generated sample's .json, else erosion)",
    )
    parser.add_argument("--resolution", type=int, default=128, help="Fine grid size")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=2.5,
        help="Detail depth in fine-grid voxels",
    )
    parser.add_argument(
        "--tile",
        type=float,
        default=1.5,
        help="Texture repeats across the volume (higher = finer detail)",
    )
    parser.add_argument(
        "--weathering",
        type=float,
        default=0.5,
        help="Extra detail on upward-facing surfaces (0 = uniform)",
    )
    parser.add_argument(
        "--texture-checkpoint",
        default=None,
        help="texture_ema_*.weights.h5 (default: latest in checkpoints_texture/)",
    )
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="Use the procedural texture processes instead of the texture GAN",
    )
    parser.add_argument("--truncation", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    from config import CONFIG

    sdf = np.load(args.input).astype(np.float32)

    phenomena = None
    if args.phenomena:
        phenomena = [p.strip() for p in args.phenomena.split(",") if p.strip()]
    else:
        sidecar = os.path.splitext(args.input)[0] + ".json"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as file:
                phenomena = json.load(file).get("phenomena")
    phenomena = phenomena or ["erosion"]

    for name in phenomena:
        if name not in PHENOMENA:
            raise SystemExit(f"Unknown phenomenon '{name}'. Options: {PHENOMENA}")

    checkpoint = args.texture_checkpoint or latest_texture_checkpoint(
        CONFIG.texture_checkpoint_dir
    )

    if args.procedural or checkpoint is None:
        source = "procedural"
        maps = procedural_textures(phenomena, 3, args.seed, CONFIG.texture_size)
    else:
        source = checkpoint
        maps = gan_textures(checkpoint, phenomena, 3, args.seed)

    print(f"Texture: {'+'.join(phenomena)} from {source}")

    fine = upsample(sdf, args.resolution, args.truncation)
    height = triplanar(fine, maps, args.tile, args.weathering)
    fine = keep_largest(fine + args.amplitude * height)

    vertices, faces = volume_to_mesh(fine, 0.0)

    output = args.output or os.path.splitext(args.input)[0] + "_detail.obj"
    write_obj(output, vertices, faces)
    np.save(os.path.splitext(output)[0] + ".npy", fine)

    print(f"Grid: {args.resolution}^3 | vertices: {len(vertices)} | faces: {len(faces)}")
    print("Saved:", output)


if __name__ == "__main__":
    main()
