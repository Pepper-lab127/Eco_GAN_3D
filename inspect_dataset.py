import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DATA_PATH = Path("data/procedural.npz")
METADATA_PATH = Path("data/procedural_metadata.json")
N_SAMPLES = 24
GALLERY_SEED = 42
ISO_LEVEL = 0.5


def load_dataset():
    """Return (volumes, iso_level). Prefers the smooth signed distance field."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run generate_dataset.py first.")
    data = np.load(DATA_PATH)
    if "sdf" in data.files:
        volumes, iso = data["sdf"], 0.0
    else:
        volumes, iso = data["structures"], ISO_LEVEL
    if volumes.ndim != 5 or volumes.shape[-1] != 1:
        raise ValueError(f"Expected shape (N, X, Y, Z, 1), got {volumes.shape}")
    return volumes, iso


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {METADATA_PATH}. "
            "Run the new generate_dataset.py so it creates the metadata file."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def metadata_by_id(metadata):
    if isinstance(metadata, dict):
        records = metadata.get("samples", metadata.get("metadata", []))
    else:
        records = metadata
    if not isinstance(records, list):
        raise ValueError("Could not find a sample list in procedural_metadata.json.")
    return {
        int(r["id"]): r for r in records
        if isinstance(r, dict) and "id" in r
    }


def phenomenon_label(record):
    if not record:
        return "UNKNOWN"
    phenomena = record.get("phenomena")
    if isinstance(phenomena, list) and phenomena:
        return " + ".join(str(x).upper() for x in phenomena)
    if isinstance(phenomena, str):
        return phenomena.upper()
    for key in ("phenomenon", "type", "category", "process"):
        if key in record:
            return str(record[key]).upper()
    return "UNKNOWN"


def short_description(record):
    if not record:
        return ""
    parts = []
    for key in ("agent", "duration", "intensity", "sequence"):
        if key in record:
            value = record[key]
            parts.append(f"{key}={value}")
    metrics = record.get("metrics", {})
    for key, short in (("water_retention", "water"), ("crevice_fraction", "crevice")):
        if key in metrics:
            parts.append(f"{short}={metrics[key]:.3f}")
    return " | ".join(parts)


def render_sample(ax, volume, iso=ISO_LEVEL):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.max() <= iso or volume.min() >= iso:
        ax.text2D(0.5, 0.5, "EMPTY", transform=ax.transAxes,
                  ha="center", va="center")
        return

    verts, faces, _, _ = marching_cubes(volume, level=iso)
    mesh = Poly3DCollection(verts[faces], alpha=0.80)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, volume.shape[0] - 1)
    ax.set_ylim(0, volume.shape[1] - 1)
    ax.set_zlim(0, volume.shape[2] - 1)
    ax.set_box_aspect(volume.shape)


def main():
    structures, iso = load_dataset()
    metadata_lookup = metadata_by_id(load_metadata())

    n_total = structures.shape[0]
    n_show = min(N_SAMPLES, n_total)
    rng = np.random.default_rng(GALLERY_SEED)
    indices = rng.choice(n_total, size=n_show, replace=False)

    n_cols = 6
    n_rows = int(np.ceil(n_show / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 4.5 * n_rows))

    print("\n" + "=" * 70)
    print("ECO_GAN_3D — ECOLOGICAL MORPHOLOGY DATASET")
    print("=" * 70)
    print(f"Dataset:       {DATA_PATH}")
    print(f"Total samples: {n_total}")
    print(f"Showing:       {n_show}\n")

    for plot_number, index in enumerate(indices):
        ax = fig.add_subplot(n_rows, n_cols, plot_number + 1, projection="3d")
        volume = structures[index, ..., 0]
        render_sample(ax, volume, iso)

        record = metadata_lookup.get(int(index), {})
        label = phenomenon_label(record)
        details = short_description(record)
        occupancy = float(np.mean(volume > iso))

        title = f"#{index}  {label}\noccupancy={occupancy:.2f}"
        if details:
            title += f"\n{details}"
        ax.set_title(title, fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        print(f"Sample {index:4d} | {label:35s} | occupancy={occupancy:.3f}"
              + (f" | {details}" if details else ""))

    fig.suptitle("Eco_GAN_3D — Ecological Morphology Dataset", fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
