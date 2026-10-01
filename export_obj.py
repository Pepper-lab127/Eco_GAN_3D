import argparse

import numpy as np

from skimage.measure import marching_cubes


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
        default=0.5
    )

    args = parser.parse_args()

    volume = np.load(
        args.input
    )

    print(
        "Volume shape:",
        volume.shape
    )

    print(
        "Occupancy:",
        volume.mean()
    )

    vertices, faces, normals, values = (
        marching_cubes(
            volume,
            level=args.threshold
        )
    )

    # Convert voxel coordinates to approximately
    # normalized architectural coordinates.
    resolution = volume.shape[0]

    vertices = (
        vertices /
        (resolution - 1)
    )

    vertices = (
        vertices * 2.0 - 1.0
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
