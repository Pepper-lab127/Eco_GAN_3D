import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from config import CONFIG
from ecology_metrics import compute_metrics
from models import (
    build_generator,
    build_discriminator
)


# ============================================================
# Dataset
# ============================================================

class ConditionedDataset:
    """
    Signed distance volumes plus their condition vectors.

    condition = [multi-hot phenomenon labels, standardised metrics]

    Batches are drawn in NumPy so the dataset is never embedded in the
    TensorFlow graph (which caps at 2 GB and would break at 64^3).
    """

    def __init__(self, path, condition_metrics, seed):

        data = np.load(path)

        if "sdf" not in data.files:
            raise ValueError(
                f"{path} has no 'sdf' array. Regenerate it with "
                "the current generate_dataset.py."
            )

        self.volumes = data["sdf"]
        self.phenomena = [str(p) for p in data["phenomena"]]
        self.sample_phenomenon = [
            str(p) for p in data["sample_phenomenon"]
        ]
        self.tsdf_truncation = float(data["tsdf_truncation"])
        self.condition_metrics = list(condition_metrics)

        metrics = np.stack(
            [
                data[f"metric_{name}"]
                for name in self.condition_metrics
            ],
            axis=1,
        ).astype(np.float32)

        self.metrics = metrics
        self.metric_mean = metrics.mean(axis=0)
        self.metric_std = metrics.std(axis=0) + 1e-6

        self.conditions = np.concatenate(
            [
                data["labels"].astype(np.float32),
                self.standardise(metrics),
            ],
            axis=1,
        )

        self.size = len(self.volumes)
        self.resolution = self.volumes.shape[1]
        self.rng = np.random.default_rng(seed)

    def standardise(self, metrics):
        return np.clip(
            (metrics - self.metric_mean) / self.metric_std,
            -4.0,
            4.0,
        ).astype(np.float32)

    def batch(self, batch_size):
        indices = self.rng.integers(0, self.size, batch_size)

        volumes = self.volumes[indices].astype(np.float32)

        # Rotations about the vertical axis and horizontal mirroring.
        # These leave gravity, and so every metric, unchanged.
        for i in range(batch_size):
            volumes[i] = np.rot90(
                volumes[i],
                k=int(self.rng.integers(4)),
                axes=(0, 1),
            )

            if self.rng.random() < 0.5:
                volumes[i] = volumes[i, ::-1]

        return volumes, self.conditions[indices]

    def stats(self):
        """Everything generate.py needs to rebuild conditions."""
        percentiles = np.percentile(
            self.metrics,
            [10, 50, 90],
            axis=0,
        )

        return {
            "resolution": int(self.resolution),
            "latent_dim": int(CONFIG.latent_dim),
            "phenomena": self.phenomena,
            "condition_metrics": self.condition_metrics,
            "metric_mean": self.metric_mean.tolist(),
            "metric_std": self.metric_std.tolist(),
            "metric_p10": percentiles[0].tolist(),
            "metric_p50": percentiles[1].tolist(),
            "metric_p90": percentiles[2].tolist(),
            "tsdf_truncation": self.tsdf_truncation,
        }


# ============================================================
# GAN
# ============================================================

class EcologicalGAN:

    def __init__(
        self,
        resolution,
        condition_dim,
        generator_fn=build_generator,
        discriminator_fn=build_discriminator,
    ):
        # The builders are swappable so the same training logic serves
        # the 3D volume GAN and the 2D surface texture GAN.

        self.generator = generator_fn(
            resolution=resolution,
            condition_dim=condition_dim,
        )

        self.discriminator = discriminator_fn(
            resolution=resolution,
            condition_dim=condition_dim,
        )

        # Slow-moving average of the generator: smoother, more reliable
        # samples than the raw generator at any single step.
        self.ema = generator_fn(
            resolution=resolution,
            condition_dim=condition_dim,
        )
        self.ema.set_weights(self.generator.get_weights())

        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=CONFIG.learning_rate_generator,
            beta_1=CONFIG.beta_1,
            beta_2=CONFIG.beta_2
        )

        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=CONFIG.learning_rate_discriminator,
            beta_1=CONFIG.beta_1,
            beta_2=CONFIG.beta_2
        )

        # Create optimizer slots now so a checkpoint restore fills them.
        for optimizer, model in (
            (self.g_optimizer, self.generator),
            (self.d_optimizer, self.discriminator),
        ):
            if hasattr(optimizer, "build"):
                optimizer.build(model.trainable_variables)

        self.d_step_plain = tf.function(
            lambda real, condition: self._d_step(real, condition, False)
        )
        self.d_step_r1 = tf.function(
            lambda real, condition: self._d_step(real, condition, True)
        )
        self.g_step = tf.function(self._g_step)

    # --------------------------------------------------------
    # Discriminator: non-saturating logistic loss + lazy R1
    # --------------------------------------------------------

    def _d_step(self, real, condition, apply_r1):

        batch_size = tf.shape(real)[0]

        z = tf.random.normal([batch_size, CONFIG.latent_dim])

        fake = self.generator([z, condition], training=True)

        with tf.GradientTape() as d_tape:

            with tf.GradientTape() as r1_tape:
                r1_tape.watch(real)
                real_logits = self.discriminator(
                    [real, condition],
                    training=True
                )
                # Summed inside the tape so the sum is recorded too.
                real_logit_sum = tf.reduce_sum(real_logits)

            fake_logits = self.discriminator(
                [fake, condition],
                training=True
            )

            d_loss = tf.reduce_mean(
                tf.nn.softplus(fake_logits)
                + tf.nn.softplus(-real_logits)
            )

            r1_penalty = tf.constant(0.0)

            if apply_r1:
                gradients = r1_tape.gradient(
                    real_logit_sum,
                    real
                )
                r1_penalty = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(gradients),
                        axis=list(range(1, len(real.shape)))
                    )
                )
                d_loss += (
                    0.5
                    * CONFIG.r1_gamma
                    * CONFIG.r1_interval
                    * r1_penalty
                )

        d_gradients = d_tape.gradient(
            d_loss,
            self.discriminator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(
                d_gradients,
                self.discriminator.trainable_variables
            )
        )

        return {
            "d_loss": d_loss,
            "r1": r1_penalty,
            "real_logit": tf.reduce_mean(real_logits),
            "fake_logit": tf.reduce_mean(fake_logits),
        }

    # --------------------------------------------------------
    # Generator: non-saturating loss, then EMA update
    # --------------------------------------------------------

    def _g_step(self, condition):

        batch_size = tf.shape(condition)[0]

        z = tf.random.normal([batch_size, CONFIG.latent_dim])

        with tf.GradientTape() as g_tape:

            fake = self.generator([z, condition], training=True)

            fake_logits = self.discriminator(
                [fake, condition],
                training=True
            )

            g_loss = tf.reduce_mean(
                tf.nn.softplus(-fake_logits)
            )

        g_gradients = g_tape.gradient(
            g_loss,
            self.generator.trainable_variables
        )

        self.g_optimizer.apply_gradients(
            zip(
                g_gradients,
                self.generator.trainable_variables
            )
        )

        for ema_weight, weight in zip(
            self.ema.weights,
            self.generator.weights
        ):
            ema_weight.assign(
                CONFIG.ema_decay * ema_weight
                + (1.0 - CONFIG.ema_decay) * weight
            )

        return {"g_loss": g_loss}

    def train_step(self, real, condition, step):

        real = tf.convert_to_tensor(real)
        condition = tf.convert_to_tensor(condition)

        if step % CONFIG.r1_interval == 0:
            logs = self.d_step_r1(real, condition)
        else:
            logs = self.d_step_plain(real, condition)

        logs.update(self.g_step(condition))

        return logs


# ============================================================
# Previews
# ============================================================

class Preview:
    """
    A fixed set of (z, condition) pairs rendered throughout training,
    so successive previews are directly comparable.

    Conditions are copied from real samples: one per phenomenon plus
    hybrids. The metrics of each generated shape are measured and
    compared with the metrics it was asked for.
    """

    def __init__(self, dataset, count=8):

        rng = np.random.default_rng(CONFIG.seed)

        indices = []

        for name in dataset.phenomena + ["hybridization"]:
            matches = [
                i for i, p in enumerate(dataset.sample_phenomenon)
                if p == name
            ]
            if matches:
                indices.append(int(rng.choice(matches)))

        while len(indices) < count:
            indices.append(int(rng.integers(dataset.size)))

        self.indices = indices[:count]
        self.dataset = dataset
        self.labels = [
            dataset.sample_phenomenon[i] for i in self.indices
        ]
        self.conditions = dataset.conditions[self.indices]
        self.requested = dataset.metrics[self.indices]
        self.z = rng.normal(
            size=(len(self.indices), CONFIG.latent_dim)
        ).astype(np.float32)

    def run(self, generator, step, writer):

        volumes = generator(
            [self.z, self.conditions],
            training=False
        ).numpy()[..., 0]

        np.savez_compressed(
            os.path.join(CONFIG.output_dir, f"preview_{step:06d}.npz"),
            sdf=volumes,
            conditions=self.conditions,
            labels=np.array(self.labels),
        )

        achieved = np.array(
            [
                [
                    compute_metrics(volume > 0)[name]
                    for name in self.dataset.condition_metrics
                ]
                for volume in volumes
            ],
            dtype=np.float32,
        )

        # Error in units of the dataset's standard deviation.
        error = np.abs(
            achieved - self.requested
        ) / self.dataset.metric_std

        with writer.as_default():
            for j, name in enumerate(self.dataset.condition_metrics):
                tf.summary.scalar(
                    f"condition_error/{name}",
                    float(error[:, j].mean()),
                    step=step,
                )

        render_preview(
            volumes,
            self.labels,
            os.path.join(CONFIG.output_dir, f"preview_{step:06d}.png"),
        )

        summary = ", ".join(
            f"{name}={error[:, j].mean():.2f}"
            for j, name in enumerate(self.dataset.condition_metrics)
        )

        print(f"  preview saved | condition error (std units): {summary}")


def render_preview(volumes, labels, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    columns = 4
    rows = int(np.ceil(len(volumes) / columns))

    figure = plt.figure(figsize=(3 * columns, 3 * rows))

    for i, (volume, label) in enumerate(zip(volumes, labels)):

        axis = figure.add_subplot(rows, columns, i + 1, projection="3d")
        axis.set_axis_off()
        axis.set_title(label.upper(), fontsize=8)

        if volume.max() <= 0 or volume.min() >= 0:
            axis.text2D(0.5, 0.5, "EMPTY", transform=axis.transAxes,
                        ha="center")
            continue

        vertices, faces, _, _ = marching_cubes(volume, level=0.0)

        axis.add_collection3d(
            Poly3DCollection(vertices[faces], alpha=0.85)
        )

        n = volume.shape[0] - 1
        axis.set_xlim(0, n)
        axis.set_ylim(0, n)
        axis.set_zlim(0, n)

    figure.tight_layout()
    figure.savefig(path, dpi=80)
    plt.close(figure)


# ============================================================
# Main training loop
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=CONFIG.dataset_path)
    parser.add_argument("--steps", type=int, default=CONFIG.total_steps)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)

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

    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        # Allocate GPU memory as needed instead of all at once.
        tf.config.experimental.set_memory_growth(gpu, True)

    print("GPUs:", [gpu.name for gpu in gpus] or "none (training on CPU)")

    dataset = ConditionedDataset(
        args.data,
        CONFIG.condition_metrics,
        CONFIG.seed,
    )

    condition_dim = dataset.conditions.shape[1]

    print(
        f"Dataset: {dataset.size} samples at {dataset.resolution}^3, "
        f"condition = {len(dataset.phenomena)} labels + "
        f"{len(dataset.condition_metrics)} metrics"
    )

    gan = EcologicalGAN(dataset.resolution, condition_dim)

    with open(
        os.path.join(CONFIG.checkpoint_dir, "condition_stats.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(dataset.stats(), file, indent=2)

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
        os.path.join(CONFIG.checkpoint_dir, "train_state"),
        max_to_keep=3,
    )

    if manager.latest_checkpoint and not args.fresh:
        state.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint} at step {int(step)}")

    writer = tf.summary.create_file_writer(CONFIG.log_dir)
    preview = Preview(dataset)

    print()
    print(
        f"Training to step {args.steps} "
        f"(batch {args.batch_size}). "
        f"TensorBoard: tensorboard --logdir {CONFIG.log_dir}"
    )
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
            # R1 is only evaluated every r1_interval steps.
            means["r1"] *= CONFIG.r1_interval
            rate = count / (time.time() - started)

            with writer.as_default():
                for key, value in means.items():
                    tf.summary.scalar(f"train/{key}", value, step=current)

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
            preview.run(gan.ema, current, writer)

        if (
            current % CONFIG.checkpoint_every == 0
            or current == args.steps
        ):
            manager.save(checkpoint_number=current)

            path = os.path.join(
                CONFIG.checkpoint_dir,
                f"generator_ema_{current:06d}.weights.h5"
            )
            gan.ema.save_weights(path)

            print(f"  checkpoint saved: {path}")

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
