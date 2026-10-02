import math

import tensorflow as tf

from config import CONFIG


# ============================================================
# Conditioning layers
# ============================================================

class FiLM(tf.keras.layers.Layer):
    """
    Feature-wise linear modulation: the condition vector scales and
    shifts every channel. Lets phenomenon labels and ecological targets
    steer every resolution of the generator, not just its input.
    Initialised to the identity.
    """

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.projection = tf.keras.layers.Dense(
            2 * channels,
            kernel_initializer="zeros",
        )

    def call(self, inputs):
        x, condition = inputs
        gamma, beta = tf.split(
            self.projection(condition),
            2,
            axis=-1,
        )
        # Broadcast over every spatial axis (2D maps or 3D volumes).
        shape = [-1] + [1] * (len(x.shape) - 2) + [self.channels]
        gamma = tf.reshape(gamma, shape)
        beta = tf.reshape(beta, shape)
        return x * (1.0 + gamma) + beta


class MinibatchStdDev(tf.keras.layers.Layer):
    """
    Appends the batch-wide feature standard deviation as one channel.
    The discriminator can then see when the generator produces the same
    shape for every input, which counters mode collapse.
    """

    def call(self, x):
        _mean, variance = tf.nn.moments(x, axes=[0])
        std = tf.reduce_mean(tf.sqrt(variance + 1e-8))
        feature = tf.fill(
            tf.concat([tf.shape(x)[:-1], [1]], axis=0),
            std,
        )
        return tf.concat([x, feature], axis=-1)


class ProjectionLogit(tf.keras.layers.Layer):
    """
    Projection discriminator output (Miyato & Koyama 2018):
    logit = w . h + <embed(condition), h>
    The inner product rewards samples whose features agree with the
    requested phenomena and metrics.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.unconditional = tf.keras.layers.Dense(1)
        self.embedding = tf.keras.layers.Dense(
            units,
            use_bias=False,
        )

    def call(self, inputs):
        features, condition = inputs
        projection = tf.reduce_sum(
            self.embedding(condition) * features,
            axis=-1,
            keepdims=True,
        )
        return self.unconditional(features) + projection


def upsampling_blocks(resolution):
    blocks = int(round(math.log2(resolution / 4)))

    if 4 * 2 ** blocks != resolution:
        raise ValueError(
            f"resolution must be 4 * 2^n (32, 64, ...), got {resolution}"
        )

    return blocks


def channels_at(level):
    """Channel width at 4 * 2^level voxels: 256, 128, 64, 32, 16."""
    return max(16, 256 >> level)


# ============================================================
# Generator
# ============================================================

def build_generator(
    resolution=None,
    latent_dim=None,
    condition_dim=None,
):
    """
    z + condition -> truncated signed distance field in [-1, 1]
    (positive inside material, surface at 0).
    """
    resolution = resolution or CONFIG.resolution
    latent_dim = latent_dim or CONFIG.latent_dim

    if condition_dim is None:
        raise ValueError("condition_dim is required (labels + metrics)")

    z = tf.keras.Input(
        shape=(latent_dim,),
        name="latent_vector"
    )

    condition = tf.keras.Input(
        shape=(condition_dim,),
        name="condition"
    )

    c = tf.keras.layers.Dense(128, activation="gelu")(condition)
    c = tf.keras.layers.Dense(128, activation="gelu")(c)

    x = tf.keras.layers.Concatenate()([z, c])

    x = tf.keras.layers.Dense(
        4 * 4 * 4 * channels_at(0),
        use_bias=False
    )(x)

    x = tf.keras.layers.Reshape(
        (4, 4, 4, channels_at(0))
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)
    x = FiLM(channels_at(0))([x, c])
    x = tf.keras.layers.Activation("gelu")(x)

    for level in range(1, upsampling_blocks(resolution) + 1):
        channels = channels_at(level)

        # Nearest-neighbour upsampling + convolution avoids the
        # checkerboard artefacts of strided transposed convolutions.
        x = tf.keras.layers.UpSampling3D(2)(x)

        for _ in range(2):
            x = tf.keras.layers.Conv3D(
                channels,
                kernel_size=3,
                padding="same",
                use_bias=False
            )(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = FiLM(channels)([x, c])
            x = tf.keras.layers.Activation("gelu")(x)

    output = tf.keras.layers.Conv3D(
        1,
        kernel_size=3,
        padding="same",
        activation="tanh",
        name="signed_distance"
    )(x)

    return tf.keras.Model(
        [z, condition],
        output,
        name="3D_Ecological_Generator"
    )


# ============================================================
# Discriminator
# ============================================================

def build_discriminator(
    resolution=None,
    condition_dim=None,
):
    resolution = resolution or CONFIG.resolution

    if condition_dim is None:
        raise ValueError("condition_dim is required (labels + metrics)")

    blocks = upsampling_blocks(resolution)

    volume = tf.keras.Input(
        shape=(resolution, resolution, resolution, 1),
        name="voxel_volume"
    )

    condition = tf.keras.Input(
        shape=(condition_dim,),
        name="condition"
    )

    x = tf.keras.layers.Conv3D(
        channels_at(blocks),
        kernel_size=3,
        padding="same"
    )(volume)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    for level in range(blocks - 1, -1, -1):
        x = tf.keras.layers.Conv3D(
            channels_at(level + 1),
            kernel_size=3,
            padding="same"
        )(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv3D(
            channels_at(level),
            kernel_size=4,
            strides=2,
            padding="same"
        )(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    # 4 x 4 x 4
    x = MinibatchStdDev()(x)

    x = tf.keras.layers.Conv3D(
        channels_at(0),
        kernel_size=3,
        padding="same"
    )(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    features = tf.keras.layers.Dense(256)(x)
    features = tf.keras.layers.LeakyReLU(0.2)(features)

    c = tf.keras.layers.Dense(128)(condition)
    c = tf.keras.layers.LeakyReLU(0.2)(c)

    # Logit, no sigmoid: the losses use softplus on logits.
    output = ProjectionLogit(256, name="real_fake_logit")(
        [features, c]
    )

    return tf.keras.Model(
        [volume, condition],
        output,
        name="3D_Ecological_Discriminator"
    )


# ============================================================
# Loading weights across TensorFlow versions
# ============================================================

def load_weights(model, path):
    """
    model.load_weights with a clear message for the one common failure:
    weights saved by Keras 3 (TensorFlow 2.16+, e.g. Kaggle or Colab)
    cannot be read by Keras 2 (TensorFlow 2.15 and older, e.g. the
    TensorFlow 2.10 needed for native Windows GPU support).
    """
    try:
        model.load_weights(path)
    except ValueError as error:
        if tf.__version__ < "2.16":
            raise SystemExit(
                f"Could not load {path} with TensorFlow {tf.__version__}.\n"
                "These weights were most likely saved by a newer TensorFlow "
                "(Keras 3, e.g. on Kaggle or Colab), which TensorFlow 2.10 "
                "cannot read. Run this script in an environment with a "
                "current TensorFlow instead; generating and adding detail "
                "run fine on the CPU. See README, 'Using weights trained "
                "elsewhere'."
            ) from error
        raise


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    condition_dim = 7 + len(CONFIG.condition_metrics)

    generator = build_generator(condition_dim=condition_dim)
    discriminator = build_discriminator(condition_dim=condition_dim)

    generator.summary()
    discriminator.summary()

    z = tf.random.normal([2, CONFIG.latent_dim])
    condition = tf.random.normal([2, condition_dim])

    fake = generator([z, condition])

    print("Generated shape:", fake.shape)

    prediction = discriminator([fake, condition])

    print("Discriminator shape:", prediction.shape)
