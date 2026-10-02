import argparse
import json
import os

import numpy as np
import tensorflow as tf

from config import CONFIG
from ecology_metrics import compute_metrics
from export_obj import volume_to_mesh, write_obj
from models import build_generator, load_weights


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate structures from a trained EMA generator. "
            "Example: python generate.py --checkpoint "
            "checkpoints/generator_ema_030000.weights.h5 "
            "--phenomena erosion,porosity --metric water_retention=0.02 "
            "--count 8 --obj"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Generator weights file (generator_ema_*.weights.h5)"
    )

    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="condition_stats.json (default: next to the checkpoint)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10
    )

    parser.add_argument(
        "--phenomena",
        type=str,
        default=None,
        help=(
            "Comma-separated processes to combine, e.g. erosion,porosity. "
            "Default: a random single phenomenon per sample."
        )
    )

    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Target value for a conditioning metric in its own units, "
            "e.g. water_retention=0.02. Repeatable. Unset metrics use "
            "the dataset median. Ranges are listed in condition_stats.json."
        )
    )

    parser.add_argument(
        "--truncation",
        type=float,
        default=1.0,
        help="Scale latent vectors (<1 = more typical, less varied)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--obj",
        action="store_true",
        help="Also export an OBJ mesh for each sample"
    )

    return parser.parse_args()


def build_condition(stats, phenomena, targets):

    labels = np.zeros(len(stats["phenomena"]), dtype=np.float32)

    for name in phenomena:
        labels[stats["phenomena"].index(name)] = 1.0

    mean = np.asarray(stats["metric_mean"], dtype=np.float32)
    std = np.asarray(stats["metric_std"], dtype=np.float32)

    metrics = np.asarray(stats["metric_p50"], dtype=np.float32).copy()

    for name, value in targets.items():
        metrics[stats["condition_metrics"].index(name)] = value

    standardised = np.clip((metrics - mean) / std, -4.0, 4.0)

    return np.concatenate([labels, standardised]), metrics


def main():

    args = parse_args()

    stats_path = args.stats or os.path.join(
        os.path.dirname(args.checkpoint) or ".",
        "condition_stats.json"
    )

    with open(stats_path, "r", encoding="utf-8") as file:
        stats = json.load(file)

    targets = {}

    for item in args.metric:
        name, _, value = item.partition("=")
        name = name.strip()

        if name not in stats["condition_metrics"]:
            raise SystemExit(
                f"Unknown metric '{name}'. "
                f"Conditioning metrics: {stats['condition_metrics']}"
            )

        targets[name] = float(value)

    requested = None

    if args.phenomena:
        requested = [p.strip() for p in args.phenomena.split(",") if p.strip()]

        for name in requested:
            if name not in stats["phenomena"]:
                raise SystemExit(
                    f"Unknown phenomenon '{name}'. "
                    f"Options: {stats['phenomena']}"
                )

    condition_dim = (
        len(stats["phenomena"]) + len(stats["condition_metrics"])
    )

    generator = build_generator(
        resolution=stats["resolution"],
        latent_dim=stats["latent_dim"],
        condition_dim=condition_dim,
    )

    load_weights(generator, args.checkpoint)

    print("Loaded:", args.checkpoint)

    rng = np.random.default_rng(args.seed)

    for i in range(args.count):

        phenomena = requested or [str(rng.choice(stats["phenomena"]))]

        condition, target_metrics = build_condition(
            stats,
            phenomena,
            targets,
        )

        z = (
            rng.normal(size=(1, stats["latent_dim"])) * args.truncation
        ).astype(np.float32)

        volume = generator(
            [z, condition[None].astype(np.float32)],
            training=False
        )[0].numpy()[..., 0]

        filename = os.path.join(
            CONFIG.output_dir,
            f"generated_{i:04d}.npy"
        )

        np.save(filename, volume)

        achieved = compute_metrics(volume > 0)

        # Sidecar so apply_detail.py can match surface detail to the form.
        with open(filename.replace(".npy", ".json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "phenomena": phenomena,
                    "targets": dict(zip(stats["condition_metrics"], map(float, target_metrics))),
                    "achieved": {k: float(v) for k, v in achieved.items()},
                },
                file,
                indent=2,
            )

        comparison = ", ".join(
            f"{name} {target:.3f}->{achieved[name]:.3f}"
            for name, target in zip(stats["condition_metrics"], target_metrics)
        )

        print(f"Saved: {filename} | {'+'.join(phenomena)} | target->got: {comparison}")

        if args.obj:
            try:
                vertices, faces = volume_to_mesh(volume, 0.0)
            except ValueError as error:
                print("  OBJ skipped:", error)
                continue

            obj_path = filename.replace(".npy", ".obj")
            write_obj(obj_path, vertices, faces)
            print("  OBJ:", obj_path)


if __name__ == "__main__":
    main()
