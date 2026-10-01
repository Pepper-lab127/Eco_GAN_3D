import argparse

import numpy as np

from skimage.measure import marching_cubes


def default_level(volume):
    """Signed distance volumes (with negatives) mesh at 0, occupancy at 0.5."""
    return 0.0 if float(volume.min()) < 0.0 else 0.5


def volume_to_mesh(volume, level=None):
    """
    Marching cubes, with vertices scaled to normalized coordinates
    in [-1, 1]. Axis 2 (z) is up.
    """
    if level is None:
        level = default_level(volume)

    if not (volume.min() < level < volume.max()):
        raise ValueError(
            f"Volume does not cross level {level} "
            f"(range {volume.min():.3f} to {volume.max():.3f}); "
            "nothing to mesh."
        )

    vertices, faces, normals, values = (
        marching_cubes(
            volume,
            level=level
        )
    )

    resolution = volume.shape[0]

    vertices = (
        vertices /
        (resolution - 1)
    )

    vertices = (
        vertices * 2.0 - 1.0
    )

    return vertices, faces


def write_obj(
    filename,
    vertices,
    faces
):

    with open(
        filename,
        "w"
    ) as f:

        for vertex in vertices:

            x, y, z = vertex

            f.write(
                f"v {x:.6f} "
                f"{y:.6f} "
                f"{z:.6f}\n"
            )

        for face in faces:

            a, b, c = face

            # OBJ indices are 1-based
            f.write(
                f"f {a + 1} "
                f"{b + 1} "
                f"{c + 1}\n"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Iso level. Default: 0 for signed distance, 0.5 for occupancy"
    )

    args = parser.parse_args()

    volume = np.load(
        args.input
    )

    level = (
        default_level(volume)
        if args.threshold is None
        else args.threshold
    )

    print(
        "Volume shape:",
        volume.shape
    )

    print(
        "Occupancy:",
        float((volume > level).mean())
    )

    vertices, faces = volume_to_mesh(
        volume,
        level
    )

    write_obj(
        args.output,
        vertices,
        faces
    )

    print()
    print(
        "Vertices:",
        len(vertices)
    )

    print(
        "Faces:",
        len(faces)
    )

    print(
        "Saved:",
        args.output
    )


if __name__ == "__main__":
    main()
