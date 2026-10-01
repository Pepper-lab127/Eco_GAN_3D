import argparse
import os

import numpy as np
import tensorflow as tf

from config import CONFIG
from models import build_generator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Generator weights file"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10
    )

    args = parser.parse_args()

    generator = build_generator()

    generator.load_weights(
        args.checkpoint
    )

    print(
        "Loaded:",
        args.checkpoint
    )

    for i in range(
        args.count
    ):

        z = tf.random.normal(
            [1, CONFIG.latent_dim]
        )

        volume = generator(
            z,
            training=False
        )[0]

        volume = (
            volume.numpy()
            [..., 0]
        )

        filename = os.path.join(
            CONFIG.output_dir,
            f"generated_{i:04d}.npy"
        )

        np.save(
            filename,
            volume
        )

        print(
            "Saved:",
            filename
        )


if __name__ == "__main__":
    main()
