import numpy as np
from tqdm import tqdm

from config import CONFIG


# ------------------------------------------------------------
# Coordinate grid
# ------------------------------------------------------------

def make_grid(resolution):

    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    z = np.linspace(-1.0, 1.0, resolution)

    X, Y, Z = np.meshgrid(
        x, y, z,
        indexing="ij"
    )

    return X, Y, Z


# ------------------------------------------------------------
# Gaussian mass
# ------------------------------------------------------------

def gaussian_mass(X, Y, Z, center, scale):

    cx, cy, cz = center
    sx, sy, sz = scale

    return np.exp(
        -(
            ((X - cx) / sx) ** 2 +
            ((Y - cy) / sy) ** 2 +
            ((Z - cz) / sz) ** 2
        )
    )


# ------------------------------------------------------------
# Cylinder
# ------------------------------------------------------------

def cylinder(
    X,
    Y,
    Z,
    cx,
    cy,
    radius,
    zmin,
    zmax
):

    radial = np.sqrt(
        (X - cx) ** 2 +
        (Y - cy) ** 2
    )

    vertical = (
        (Z >= zmin) &
        (Z <= zmax)
    )

    return (
        (radial < radius) &
        vertical
    ).astype(np.float32)


# ------------------------------------------------------------
# Branch between two points
# ------------------------------------------------------------

def branch(
    X,
    Y,
    Z,
    p0,
    p1,
    radius
):

    x0, y0, z0 = p0
    x1, y1, z1 = p1

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    length_sq = dx * dx + dy * dy + dz * dz

    if length_sq < 1e-8:
        return np.zeros_like(X)

    t = (
        (X - x0) * dx +
        (Y - y0) * dy +
        (Z - z0) * dz
    ) / length_sq

    t = np.clip(t, 0.0, 1.0)

    closest_x = x0 + t * dx
    closest_y = y0 + t * dy
    closest_z = z0 + t * dz

    distance = np.sqrt(
        (X - closest_x) ** 2 +
        (Y - closest_y) ** 2 +
        (Z - closest_z) ** 2
    )

    return (
        distance < radius
    ).astype(np.float32)


# ------------------------------------------------------------
# Generate one architectural/ecological object
# ------------------------------------------------------------

def generate_structure(resolution):

    X, Y, Z = make_grid(resolution)

    structure = np.zeros(
        (resolution, resolution, resolution),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Main central mass
    # --------------------------------------------------------

    base_radius = np.random.uniform(0.20, 0.42)

    central_height = np.random.uniform(
        0.35,
        0.85
    )

    central = cylinder(
        X,
        Y,
        Z,
        0,
        0,
        base_radius,
        -0.95,
        central_height
    )

    structure = np.maximum(
        structure,
        central
    )

    # --------------------------------------------------------
    # Secondary vertical structures
    # --------------------------------------------------------

    number_of_towers = np.random.randint(
        2,
        8
    )

    for _ in range(number_of_towers):

        angle = np.random.uniform(
            0,
            2 * np.pi
        )

        distance = np.random.uniform(
            0.15,
            0.75
        )

        cx = np.cos(angle) * distance
        cy = np.sin(angle) * distance

        radius = np.random.uniform(
            0.06,
            0.18
        )

        height = np.random.uniform(
            -0.4,
            0.85
        )

        tower = cylinder(
            X,
            Y,
            Z,
            cx,
            cy,
            radius,
            -0.95,
            height
        )

        structure = np.maximum(
            structure,
            tower
        )

    # --------------------------------------------------------
    # Branching network
    # --------------------------------------------------------

    number_of_branches = np.random.randint(
        3,
        10
    )

    for _ in range(number_of_branches):

        angle = np.random.uniform(
            0,
            2 * np.pi
        )

        length = np.random.uniform(
            0.25,
            0.9
        )

        start_z = np.random.uniform(
            -0.3,
            0.6
        )

        end_z = start_z + np.random.uniform(
            -0.2,
            0.4
        )

        p0 = (
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
            start_z
        )

        p1 = (
            p0[0] + np.cos(angle) * length,
            p0[1] + np.sin(angle) * length,
            end_z
        )

        radius = np.random.uniform(
            0.025,
            0.09
        )

        b = branch(
            X,
            Y,
            Z,
            p0,
            p1,
            radius
        )

        structure = np.maximum(
            structure,
            b
        )

    # --------------------------------------------------------
    # Terraced / floating masses
    # --------------------------------------------------------

    number_of_terraces = np.random.randint(
        2,
        7
    )

    for _ in range(number_of_terraces):

        cx = np.random.uniform(
            -0.55,
            0.55
        )

        cy = np.random.uniform(
            -0.55,
            0.55
        )

        cz = np.random.uniform(
            -0.4,
            0.65
        )

        sx = np.random.uniform(
            0.08,
            0.35
        )

        sy = np.random.uniform(
            0.08,
            0.35
        )

        sz = np.random.uniform(
            0.025,
            0.12
        )

        mass = gaussian_mass(
            X,
            Y,
            Z,
            (cx, cy, cz),
            (sx, sy, sz)
        )

        mass = (
            mass > np.random.uniform(
                0.35,
                0.60
            )
        ).astype(np.float32)

        structure = np.maximum(
            structure,
            mass
        )

    # --------------------------------------------------------
    # Organic distortion
    # --------------------------------------------------------

    noise = (
        np.sin(
            X * np.random.uniform(5, 12)
            + Y * np.random.uniform(2, 8)
        )
        *
        np.cos(
            Z * np.random.uniform(3, 10)
        )
    )

    # Slightly erode some regions
    erosion_probability = np.random.uniform(
        0.0,
        0.35
    )

    structure = (
        structure *
        (noise > -erosion_probability)
    )

    # --------------------------------------------------------
    # Create cavities / voids
    # --------------------------------------------------------

    number_of_voids = np.random.randint(
        1,
        6
    )

    for _ in range(number_of_voids):

        cx = np.random.uniform(
            -0.55,
            0.55
        )

        cy = np.random.uniform(
            -0.55,
            0.55
        )

        cz = np.random.uniform(
            -0.5,
            0.7
        )

        sx = np.random.uniform(
            0.06,
            0.22
        )

        sy = np.random.uniform(
            0.06,
            0.22
        )

        sz = np.random.uniform(
            0.08,
            0.3
        )

        void = gaussian_mass(
            X,
            Y,
            Z,
            (cx, cy, cz),
            (sx, sy, sz)
        )

        void = (
            void > 0.45
        )

        structure[void] = 0.0

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    structure = (
        structure > 0.35
    ).astype(np.float32)

    # Avoid completely empty objects
    if structure.mean() < 0.01:

        structure[
            resolution // 3:
            2 * resolution // 3,

            resolution // 3:
            2 * resolution // 3,

            resolution // 4:
            3 * resolution // 4
        ] = 1.0

    return structure


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    np.random.seed(CONFIG.seed)

    print(
        f"Generating {CONFIG.dataset_size} "
        f"3D structures..."
    )

    dataset = np.zeros(
        (
            CONFIG.dataset_size,
            CONFIG.resolution,
            CONFIG.resolution,
            CONFIG.resolution
        ),
        dtype=np.float32
    )

    for i in tqdm(
        range(CONFIG.dataset_size)
    ):

        dataset[i] = generate_structure(
            CONFIG.resolution
        )

    # Add channel dimension
    dataset = dataset[..., np.newaxis]

    np.savez_compressed(
        "data/procedural.npz",
        structures=dataset
    )

    print()
    print("Dataset written to:")
    print("data/procedural.npz")
    print("Shape:", dataset.shape)
    print(
        "Mean occupancy:",
        dataset.mean()
    )


if __name__ == "__main__":
    main()
