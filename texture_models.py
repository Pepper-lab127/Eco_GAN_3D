import math

import tensorflow as tf

from config import CONFIG
from models import FiLM, MinibatchStdDev, ProjectionLogit


# ============================================================
# 2D surface texture GAN
# ============================================================
#
# Same design as the 3D models (FiLM conditioning, projection
# discriminator, minibatch std-dev), in 2D, with circular padding on
# every convolution. The networks therefore see each height map as a
# torus: what they generate tiles seamlessly, which is what lets one
# texture wrap around a 3D form without visible seams.


class CircularPad(tf.keras.layers.Layer):
    """Wrap-around padding of the two spatial axes."""

    def __init__(self, pad, **kwargs):
        super().__init__(**kwargs)
        self.pad = pad

    def call(self, x):
        p = self.pad
        x = tf.concat([x[:, -p:], x, x[:, :p]], axis=1)
        return tf.concat([x[:, :, -p:], x, x[:, :, :p]], axis=2)


def wrapped_conv(x, channels, kernel_size=3, strides=1, **kwargs):
    x = CircularPad(1)(x)
    return tf.keras.layers.Conv2D(
        channels,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        **kwargs
    )(x)


def texture_levels(resolution):
    levels = int(round(math.log2(resolution / 4)))

    if 4 * 2 ** levels != resolution:
        raise ValueError(
            f"texture resolution must be 4 * 2^n (64, 128, 256), got {resolution}"
        )

    return levels


TEXTURE_CHANNELS = (512, 256, 128, 96, 64, 32, 32)


def texture_channels(level):
    """Channel width at 4 * 2^level pixels (4, 8, 16, ... 256)."""
    return TEXTURE_CHANNELS[min(level, len(TEXTURE_CHANNELS) - 1)]


def build_texture_generator(
    resolution=None,
    latent_dim=None,
    condition_dim=None,
):
    """z + phenomenon labels -> tileable height map in [-1, 1]."""
    resolution = resolution or CONFIG.texture_size
    latent_dim = latent_dim or CONFIG.latent_dim

    if condition_dim is None:
        raise ValueError("condition_dim is required")

    z = tf.keras.Input(shape=(latent_dim,), name="latent_vector")
    condition = tf.keras.Input(shape=(condition_dim,), name="condition")

    c = tf.keras.layers.Dense(128, activation="gelu")(condition)
    c = tf.keras.layers.Dense(128, activation="gelu")(c)

    x = tf.keras.layers.Concatenate()([z, c])
    x = tf.keras.layers.Dense(4 * 4 * texture_channels(0), use_bias=False)(x)
    x = tf.keras.layers.Reshape((4, 4, texture_channels(0)))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = FiLM(texture_channels(0))([x, c])
    x = tf.keras.layers.Activation("gelu")(x)

    for level in range(1, texture_levels(resolution) + 1):
        channels = texture_channels(level)
        x = tf.keras.layers.UpSampling2D(2)(x)

        for _ in range(2):
            x = wrapped_conv(x, channels, use_bias=False)
            x = tf.keras.layers.LayerNormalization()(x)
            x = FiLM(channels)([x, c])
            x = tf.keras.layers.Activation("gelu")(x)

    output = wrapped_conv(x, 1, activation="tanh", name="height")

    return tf.keras.Model([z, condition], output, name="Texture_Generator")


def build_texture_discriminator(
    resolution=None,
    condition_dim=None,
):
    resolution = resolution or CONFIG.texture_size

    if condition_dim is None:
        raise ValueError("condition_dim is required")

    levels = texture_levels(resolution)

    height = tf.keras.Input(shape=(resolution, resolution, 1), name="height_map")
    condition = tf.keras.Input(shape=(condition_dim,), name="condition")

    x = wrapped_conv(height, texture_channels(levels))
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    for level in range(levels - 1, -1, -1):
        x = wrapped_conv(x, texture_channels(level + 1))
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = wrapped_conv(x, texture_channels(level), kernel_size=4, strides=2)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = MinibatchStdDev()(x)
    x = wrapped_conv(x, texture_channels(0))
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)

    features = tf.keras.layers.Dense(256)(x)
    features = tf.keras.layers.LeakyReLU(0.2)(features)

    c = tf.keras.layers.Dense(128)(condition)
    c = tf.keras.layers.LeakyReLU(0.2)(c)

    output = ProjectionLogit(256, name="real_fake_logit")([features, c])

    return tf.keras.Model([height, condition], output, name="Texture_Discriminator")


if __name__ == "__main__":
    generator = build_texture_generator(condition_dim=7)
    discriminator = build_texture_discriminator(condition_dim=7)
    generator.summary()
    discriminator.summary()
