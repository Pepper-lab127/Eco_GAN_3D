import tensorflow as tf

from config import CONFIG


# ============================================================
# Generator
# ============================================================

def build_generator():

    z = tf.keras.Input(
        shape=(CONFIG.latent_dim,),
        name="latent_vector"
    )

    x = tf.keras.layers.Dense(
        4 * 4 * 4 * 256,
        use_bias=False
    )(z)

    x = tf.keras.layers.Reshape(
        (4, 4, 4, 256)
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Activation(
        "gelu"
    )(x)

    # 4 -> 8
    x = tf.keras.layers.Conv3DTranspose(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Activation(
        "gelu"
    )(x)

    # 8 -> 16
    x = tf.keras.layers.Conv3DTranspose(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Activation(
        "gelu"
    )(x)

    # 16 -> 32
    x = tf.keras.layers.Conv3DTranspose(
        32,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Activation(
        "gelu"
    )(x)

    # Refinement
    x = tf.keras.layers.Conv3D(
        32,
        kernel_size=3,
        padding="same",
        activation="gelu"
    )(x)

    # Output occupancy field
    output = tf.keras.layers.Conv3D(
        1,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        name="occupancy"
    )(x)

    return tf.keras.Model(
        z,
        output,
        name="3D_Ecological_Generator"
    )


# ============================================================
# Discriminator
# ============================================================

def build_discriminator():

    volume = tf.keras.Input(
        shape=(
            CONFIG.resolution,
            CONFIG.resolution,
            CONFIG.resolution,
            1
        ),
        name="voxel_volume"
    )

    x = tf.keras.layers.Conv3D(
        32,
        kernel_size=4,
        strides=2,
        padding="same"
    )(volume)

    x = tf.keras.layers.LeakyReLU(
        0.2
    )(x)

    # 32 -> 16

    x = tf.keras.layers.Conv3D(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.LeakyReLU(
        0.2
    )(x)

    # 16 -> 8

    x = tf.keras.layers.Conv3D(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.LeakyReLU(
        0.2
    )(x)

    # 8 -> 4

    x = tf.keras.layers.Conv3D(
        256,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.LeakyReLU(
        0.2
    )(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(
        256
    )(x)

    x = tf.keras.layers.LeakyReLU(
        0.2
    )(x)

    # No sigmoid here.
    # We use logits + BinaryCrossentropy(from_logits=True).

    output = tf.keras.layers.Dense(
        1,
        name="real_fake_logit"
    )(x)

    return tf.keras.Model(
        volume,
        output,
        name="3D_Ecological_Discriminator"
    )


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    generator = build_generator()
    discriminator = build_discriminator()

    generator.summary()
    discriminator.summary()

    z = tf.random.normal(
        [2, CONFIG.latent_dim]
    )

    fake = generator(z)

    print(
        "Generated shape:",
        fake.shape
    )

    prediction = discriminator(
        fake
    )

    print(
        "Discriminator shape:",
        prediction.shape
    )
