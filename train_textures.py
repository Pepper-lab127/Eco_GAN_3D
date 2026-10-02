import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from config import CONFIG
from generate_textures import save_gallery
from texture_models import build_texture_discriminator, build_texture_generator
from train import EcologicalGAN


# ============================================================
# Dataset
# ============================================================

class TextureDataset:
    """Height maps and their multi-hot phenomenon labels."""

    def __init__(self, path, seed):
        data = np.load(path)
        self.textures = data["textures"]
        self.labels = data["labels"].astype(np.float32)
        self.phenomena = [str(p) for p in data["phenomena"]]
        self.sample_phenomenon = [str(p) for p in data["sample_phenomenon"]]
        self.size = len(self.textures)
        self.resolution = self.textures.shape[1]
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size):
        indices = self.rng.integers(0, self.size, batch_size)
        maps = self.textures[indices].astype(np.float32)
        n = self.resolution

        # The maps are periodic, so any shift, rotation or mirror is
        # another valid sample of the same surface.
        for i in range(batch_size):
            maps[i] = np.roll(maps[i], self.rng.integers(0, n, 2), axis=(0, 1))
            maps[i] = np.rot90(maps[i], k=int(self.rng.integers(4)), axes=(0, 1))
            if self.rng.random() < 0.5:
                maps[i] = maps[i, ::-1]

        return maps, self.labels[indices]


def preview(generator, dataset, z, step, writer=None):
    """One sample per phenomenon plus a hybrid, same z every time."""
    names = dataset.phenomena + [dataset.phenomena[1] + "+" + dataset.phenomena[4]]
    conditions = np.zeros((len(names), len(dataset.phenomena)), dtype=np.float32)

    for i, name in enumerate(names):
        for part in name.split("+"):
            conditions[i, dataset.phenomena.index(part)] = 1.0

    maps = generator([z[:len(names)], conditions], training=False).numpy()[..., 0]

    path = os.path.join(CONFIG.texture_output_dir, f"texture_preview_{step:06d}.png")
    save_gallery(list(maps), [[n] for n in names], path)
    print(f"  preview saved: {path}")


# ============================================================
# Main training loop
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=CONFIG.texture_dataset_path)
    parser.add_argument("--steps", type=int, default=CONFIG.texture_total_steps)
    parser.add_argument("--batch-size", type=int, default=CONFIG.texture_batch_size)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved training state and start from scratch"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tf.random.set_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(
        "GPUs:",
        [g.name for g in tf.config.list_physical_devices("GPU")]
        or "none (training on CPU)"
    )

    os.makedirs(CONFIG.texture_checkpoint_dir, exist_ok=True)
    os.makedirs(CONFIG.texture_output_dir, exist_ok=True)

    dataset = TextureDataset(args.data, CONFIG.seed)
    condition_dim = len(dataset.phenomena)

    print(
        f"Textures: {dataset.size} maps at {dataset.resolution}^2, "
        f"condition = {condition_dim} phenomenon labels"
    )

    gan = EcologicalGAN(
        dataset.resolution,
        condition_dim,
        generator_fn=build_texture_generator,
        discriminator_fn=build_texture_discriminator,
    )

    with open(
        os.path.join(CONFIG.texture_checkpoint_dir, "texture_stats.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "resolution": int(dataset.resolution),
                "latent_dim": int(CONFIG.latent_dim),
                "phenomena": dataset.phenomena,
            },
            file,
            indent=2,
        )

    step = tf.Variable(0, dtype=tf.int64, name="step")
    state = tf.train.Checkpoint(
        step=step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        ema=gan.ema,
        g_optimizer=gan.g_optimizer,
        d_optimizer=gan.d_optimizer,
    )
    manager = tf.train.CheckpointManager(
        state,
        os.path.join(CONFIG.texture_checkpoint_dir, "train_state"),
        max_to_keep=3,
    )

    if manager.latest_checkpoint and not args.fresh:
        state.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint} at step {int(step)}")

    z = np.random.default_rng(CONFIG.seed).normal(
        size=(16, CONFIG.latent_dim)
    ).astype(np.float32)

    print()
    print(f"Training to step {args.steps} (batch {args.batch_size})")
    print()

    totals = {}
    started = time.time()
    last_step = int(step)

    while int(step) < args.steps:
        current = int(step) + 1
        real, condition = dataset.batch(args.batch_size)
        logs = gan.train_step(real, condition, current)
        step.assign(current)

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)

        if current % CONFIG.log_every == 0:
            count = current - last_step
            means = {k: v / count for k, v in totals.items()}
            means["r1"] *= CONFIG.r1_interval
            rate = count / (time.time() - started)
            print(
                f"Step {current:6d} | "
                f"G {means['g_loss']:.3f} | "
                f"D {means['d_loss']:.3f} | "
                f"R1 {means['r1']:.3f} | "
                f"real {means['real_logit']:+.2f} "
                f"fake {means['fake_logit']:+.2f} | "
                f"{rate:.1f} steps/s"
            )
            totals = {}
            started = time.time()
            last_step = current

        if current == 1 or current % CONFIG.preview_every == 0:
            preview(gan.ema, dataset, z, current)

        if current % CONFIG.checkpoint_every == 0 or current == args.steps:
            manager.save(checkpoint_number=current)
            path = os.path.join(
                CONFIG.texture_checkpoint_dir,
                f"texture_ema_{current:06d}.weights.h5",
            )
            gan.ema.save_weights(path)
            print(f"  checkpoint saved: {path}")

    print()
    print("Texture training complete.")


if __name__ == "__main__":
    main()
