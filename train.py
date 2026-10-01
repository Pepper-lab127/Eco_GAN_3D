import os
import numpy as np
import tensorflow as tf

from config import CONFIG
from models import (
    build_generator,
    build_discriminator
)


# ============================================================
# Reproducibility
# ============================================================

tf.random.set_seed(
    CONFIG.seed
)

np.random.seed(
    CONFIG.seed
)


# ============================================================
# Load dataset
# ============================================================

def load_dataset():

    data = np.load(
        "data/procedural.npz"
    )

    structures = data[
        "structures"
    ].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        structures
    )

    dataset = dataset.shuffle(
        min(len(structures), 1000)
    )

    dataset = dataset.batch(
        CONFIG.batch_size,
        drop_remainder=True
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# GAN
# ============================================================

class EcologicalGAN:

    def __init__(self):

        self.generator = (
            build_generator()
        )

        self.discriminator = (
            build_discriminator()
        )

        self.g_optimizer = (
            tf.keras.optimizers.Adam(
                learning_rate=CONFIG.learning_rate_generator,
                beta_1=CONFIG.beta_1,
                beta_2=CONFIG.beta_2
            )
        )

        self.d_optimizer = (
            tf.keras.optimizers.Adam(
                learning_rate=CONFIG.learning_rate_discriminator,
                beta_1=CONFIG.beta_1,
                beta_2=CONFIG.beta_2
            )
        )

        self.loss_function = (
            tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )
        )

    # --------------------------------------------------------
    # Discriminator loss
    # --------------------------------------------------------

    def discriminator_loss(
        self,
        real_logits,
        fake_logits
    ):

        # One-sided label smoothing
        real_labels = tf.ones_like(
            real_logits
        ) * 0.9

        fake_labels = tf.zeros_like(
            fake_logits
        )

        real_loss = (
            self.loss_function(
                real_labels,
                real_logits
            )
        )

        fake_loss = (
            self.loss_function(
                fake_labels,
                fake_logits
            )
        )

        return real_loss + fake_loss

    # --------------------------------------------------------
    # Generator loss
    # --------------------------------------------------------

    def generator_loss(
        self,
        fake_logits,
        fake_volume
    ):

        real_labels = tf.ones_like(
            fake_logits
        )

        adversarial_loss = (
            self.loss_function(
                real_labels,
                fake_logits
            )
        )

        # Encourage sparse architectural fields.
        occupancy = tf.reduce_mean(
            fake_volume
        )

        sparsity_loss = tf.abs(
            occupancy -
            CONFIG.target_occupancy
        )

        return (
            adversarial_loss +
            CONFIG.sparsity_weight *
            sparsity_loss
        )

    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    @tf.function
    def train_step(
        self,
        real_volume
    ):

        batch_size = tf.shape(
            real_volume
        )[0]

        z = tf.random.normal(
            [
                batch_size,
                CONFIG.latent_dim
            ]
        )

        # ----------------------------------------------
        # Generator forward pass
        # ----------------------------------------------

        with tf.GradientTape() as g_tape:

            fake_volume = (
                self.generator(
                    z,
                    training=True
                )
            )

            fake_logits = (
                self.discriminator(
                    fake_volume,
                    training=True
                )
            )

            g_loss = (
                self.generator_loss(
                    fake_logits,
                    fake_volume
                )
            )

        g_gradients = (
            g_tape.gradient(
                g_loss,
                self.generator.trainable_variables
            )
        )

        self.g_optimizer.apply_gradients(
            zip(
                g_gradients,
                self.generator.trainable_variables
            )
        )

        # ----------------------------------------------
        # Discriminator
        # ----------------------------------------------

        with tf.GradientTape() as d_tape:

            real_logits = (
                self.discriminator(
                    real_volume,
                    training=True
                )
            )

            fake_logits = (
                self.discriminator(
                    tf.stop_gradient(fake_volume),
                    training=True
                )
            )

            d_loss = (
                self.discriminator_loss(
                    real_logits,
                    fake_logits
                )
            )

        d_gradients = (
            d_tape.gradient(
                d_loss,
                self.discriminator.trainable_variables
            )
        )

        self.d_optimizer.apply_gradients(
            zip(
                d_gradients,
                self.discriminator.trainable_variables
            )
        )

        return g_loss, d_loss


# ============================================================
# Save generated sample
# ============================================================

def save_sample(
    generator,
    epoch
):

    z = tf.random.normal(
        [1, CONFIG.latent_dim]
    )

    generated = generator(
        z,
        training=False
    )[0]

    generated = (
        generated.numpy()
        [..., 0]
    )

    path = os.path.join(
        CONFIG.output_dir,
        f"epoch_{epoch:04d}.npy"
    )

    np.save(
        path,
        generated
    )


# ============================================================
# Main training function
# ============================================================

def main():

    dataset = load_dataset()

    gan = EcologicalGAN()

    print()
    print("Generator:")
    gan.generator.summary()

    print()
    print("Discriminator:")
    gan.discriminator.summary()

    print()
    print("Beginning training...")
    print()

    for epoch in range(
        1,
        CONFIG.epochs + 1
    ):

        generator_losses = []
        discriminator_losses = []

        for real_batch in dataset:

            g_loss, d_loss = (
                gan.train_step(
                    real_batch
                )
            )

            generator_losses.append(
                float(g_loss)
            )

            discriminator_losses.append(
                float(d_loss)
            )

        g_mean = np.mean(
            generator_losses
        )

        d_mean = np.mean(
            discriminator_losses
        )

        print(
            f"Epoch {epoch:04d} | "
            f"G: {g_mean:.4f} | "
            f"D: {d_mean:.4f}"
        )

        # ----------------------------------------------
        # Save sample
        # ----------------------------------------------

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            save_sample(
                gan.generator,
                epoch
            )

        # ----------------------------------------------
        # Checkpoint
        # ----------------------------------------------

        if epoch % 10 == 0:

            gan.generator.save_weights(
                os.path.join(
                    CONFIG.checkpoint_dir,
                    f"generator_{epoch:04d}.weights.h5"
                )
            )

            gan.discriminator.save_weights(
                os.path.join(
                    CONFIG.checkpoint_dir,
                    f"discriminator_{epoch:04d}.weights.h5"
                )
            )

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
