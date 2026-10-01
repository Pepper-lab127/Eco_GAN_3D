"""
Ecological performance metrics for voxel morphologies.

These measure the qualities that make a weathered surface habitable:
pockets that hold water, sheltered crevices where seeds and spores lodge,
overhangs that shade, and the amount of surface exposed to the environment.

They are computed for every training sample (stored in the dataset and
used as GAN conditioning) and for generated samples (to check that the
generator delivers what was requested).

Convention: axis 2 (z) is vertical. Index 0 is the ground, increasing
index is up. Water falls in -z.
"""

import numpy as np
from scipy import ndimage
from skimage.measure import euler_number


METRIC_NAMES = [
    "occupancy",
    "surface_to_volume",
    "water_retention",
    "crevice_fraction",
    "overhang_fraction",
    "enclosed_void_fraction",
    "euler_number",
    "fractal_dimension",
]

SIX_NEIGHBOURS = ndimage.generate_binary_structure(
    3,
    1,
)


def exposed_faces(volume):
    """Number of voxel faces between material and air (domain edge = air)."""
    padded = np.pad(
        volume,
        1,
        constant_values=False,
    )

    count = 0

    for axis in range(3):
        count += int(
            np.count_nonzero(
                np.diff(
                    padded.astype(np.int8),
                    axis=axis,
                )
            )
        )

    return count


def exterior_air(volume):
    """Air connected to the domain boundary (reachable by weather)."""
    air = ~volume

    labels, _count = ndimage.label(
        air,
        structure=SIX_NEIGHBOURS,
    )

    boundary = np.unique(
        np.concatenate(
            [
                labels[[0, -1], :, :].ravel(),
                labels[:, [0, -1], :].ravel(),
                labels[:, :, [0, -1]].ravel(),
            ]
        )
    )

    boundary = boundary[boundary > 0]

    return np.isin(labels, boundary)


def drainable_air(volume):
    """
    Air from which water can flow out of the domain.

    Water moves sideways or down, never up. It leaves through the four
    side faces or the bottom face. Propagating backwards from those exits:
    a voxel drains if a horizontal neighbour drains, or if the voxel
    directly below it drains.
    """
    air = ~volume

    exits = np.zeros_like(air)
    exits[[0, -1], :, :] = True
    exits[:, [0, -1], :] = True
    exits[:, :, 0] = True
    exits &= air

    # Kernel offsets: horizontal neighbours and the voxel above (+z).
    structure = np.zeros(
        (3, 3, 3),
        dtype=bool,
    )
    structure[1, 1, 1] = True
    structure[0, 1, 1] = structure[2, 1, 1] = True
    structure[1, 0, 1] = structure[1, 2, 1] = True
    structure[1, 1, 2] = True

    return ndimage.binary_dilation(
        exits,
        structure=structure,
        iterations=0,
        mask=air,
    )


def surface_air(volume, outside):
    """Exterior air voxels touching material."""
    touching = ndimage.binary_dilation(
        volume,
        structure=SIX_NEIGHBOURS,
    )

    return touching & outside


def box_counting_dimension(shell):
    """Box-counting dimension of the surface shell."""
    n = shell.shape[0]
    sizes = []
    counts = []

    size = 1

    while size <= n // 4:
        m = n // size
        trimmed = shell[
            : m * size,
            : m * size,
            : m * size,
        ]

        boxes = trimmed.reshape(
            m, size, m, size, m, size
        ).any(axis=(1, 3, 5))

        occupied = int(boxes.sum())

        if occupied > 0:
            sizes.append(size)
            counts.append(occupied)

        size *= 2

    if len(sizes) < 2:
        return 0.0

    slope, _intercept = np.polyfit(
        np.log(1.0 / np.asarray(sizes, dtype=np.float64)),
        np.log(np.asarray(counts, dtype=np.float64)),
        1,
    )

    return float(slope)


def compute_metrics(volume):
    """
    Ecological metrics for one boolean volume of shape (X, Y, Z).

    Returns a dict keyed by METRIC_NAMES. Fractions are relative to the
    material volume unless stated otherwise.
    """
    volume = np.asarray(volume, dtype=bool)
    material = int(volume.sum())

    if material == 0:
        return {name: 0.0 for name in METRIC_NAMES}

    outside = exterior_air(volume)
    drains = drainable_air(volume)
    contact = surface_air(volume, outside)

    # Water held in exterior pockets that cannot drain.
    retained = outside & ~drains

    # Sheltered air next to the surface: more than half of the
    # surrounding 5x5x5 neighbourhood is material. A flat face scores
    # about 0.4, a concave corner or crevice scores higher.
    shelter = ndimage.uniform_filter(
        volume.astype(np.float32),
        size=5,
        mode="constant",
    )

    crevices = contact & (shelter > 0.5)

    # Material surface that faces down onto exterior air (shade, shelter).
    below_is_air = np.zeros_like(volume)
    below_is_air[:, :, 1:] = outside[:, :, :-1]
    overhang = volume & below_is_air

    shell = volume & ndimage.binary_dilation(
        ~volume,
        structure=SIX_NEIGHBOURS,
    )

    enclosed = (~volume) & ~outside

    return {
        "occupancy": material / volume.size,
        "surface_to_volume": exposed_faces(volume) / material,
        "water_retention": float(retained.sum()) / material,
        "crevice_fraction": float(crevices.sum())
        / max(int(contact.sum()), 1),
        "overhang_fraction": float(overhang.sum())
        / max(int(shell.sum()), 1),
        "enclosed_void_fraction": float(enclosed.sum()) / material,
        "euler_number": float(
            euler_number(volume, connectivity=1)
        ),
        "fractal_dimension": box_counting_dimension(shell),
    }
