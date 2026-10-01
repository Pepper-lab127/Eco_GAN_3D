import json
import os

import numpy as np
from tqdm import tqdm
from config import CONFIG


# Dataset V3
# ----------------
# The central change from V2:
# there is NO universal ellipsoid/blob base.
#
# Phenomena are generated from morphology/process fields:
#   branching   -> branching network
#   clustering  -> field-driven collection of masses
#   layering    -> deposition bands
#   cavitation  -> cavities carved from varied scaffolds
#   porosity    -> surface pores on varied scaffolds
#   erosion     -> surface removal from varied scaffolds
#   fragmentation -> fracture fields through varied scaffolds
#   hybridization -> combinations of those processes


PHENOMENA = [
    "branching",
    "cavitation",
    "porosity",
    "erosion",
    "layering",
    "clustering",
    "fragmentation",
]


def grid(n):
    a = np.linspace(-1, 1, n, dtype=np.float32)
    return np.meshgrid(a, a, a, indexing="ij")


def norm(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def unit_vector(rng):
    v = rng.normal(size=3).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-8)


def noise(X, Y, Z, rng, lo=1.2, hi=4.0):
    q = np.zeros_like(X, dtype=np.float32)

    for _ in range(4):
        s = rng.uniform(lo, hi)
        p = rng.uniform(-np.pi, np.pi, 3)
        q += (
            np.sin(s * X + p[0])
            * np.cos(0.83 * s * Y + p[1])
            * np.sin(0.71 * s * Z + p[2])
        )

    return norm(q)


def segment_field(X, Y, Z, a, b, radius):
    """Soft field around a line segment."""
    a = np.asarray(a, dtype=np.float32)
    d = np.asarray(b, dtype=np.float32) - a
    denom = float(np.dot(d, d))

    if denom < 1e-8:
        dist = np.sqrt(
            (X - a[0]) ** 2
            + (Y - a[1]) ** 2
            + (Z - a[2]) ** 2
        )
    else:
        t = np.clip(
            (
                (X - a[0]) * d[0]
                + (Y - a[1]) * d[1]
                + (Z - a[2]) * d[2]
            )
            / denom,
            0,
            1,
        )
        px = a[0] + t * d[0]
        py = a[1] + t * d[1]
        pz = a[2] + t * d[2]

        dist = np.sqrt(
            (X - px) ** 2
            + (Y - py) ** 2
            + (Z - pz) ** 2
        )

    return np.exp(-(dist / max(radius, 1e-4)) ** 2).astype(np.float32)


def branching_field(X, Y, Z, rng, depth=None):
    """
    Hierarchical branching structure.

    This deliberately begins with a line/branch network rather than a blob.
    """
    if depth is None:
        depth = int(rng.integers(2, 4))

    fields = []
    branches = []

    root = rng.uniform(-0.22, 0.22, 3)
    direction = unit_vector(rng)

    def grow(start, direction, length, radius, level):
        direction = direction + rng.normal(scale=0.18, size=3)
        direction /= max(np.linalg.norm(direction), 1e-8)

        end = np.clip(
            start + direction * length,
            -0.90,
            0.90,
        )

        fields.append(
            segment_field(X, Y, Z, start, end, radius)
        )

        branches.append(
            {
                "level": level,
                "start": start.tolist(),
                "end": end.tolist(),
                "radius": float(radius),
            }
        )

        if level < depth:
            for _ in range(int(rng.integers(2, 4))):
                child_direction = direction + rng.normal(
                    scale=0.38,
                    size=3,
                )
                child_direction /= max(
                    np.linalg.norm(child_direction),
                    1e-8,
                )

                grow(
                    end,
                    child_direction,
                    length * rng.uniform(0.62, 0.84),
                    max(
                        0.075,
                        radius * rng.uniform(0.62, 0.78),
                    ),
                    level + 1,
                )

    grow(
        root,
        direction,
        rng.uniform(0.40, 0.65),
        rng.uniform(0.10, 0.17),
        0,
    )

    return np.max(fields, axis=0), {
        "depth": depth,
        "branches": branches,
    }


def branching(X, Y, Z, rng):
    """Generate a branching morphology without a blob scaffold."""
    field, details = branching_field(
        X,
        Y,
        Z,
        rng,
        depth=int(rng.integers(2, 4)),
    )

    volume = field > rng.uniform(0.40, 0.52)

    return volume, details


def deposition_field(X, Y, Z, rng):
    """Curved spatial coordinate representing accumulated layers."""
    direction = unit_vector(rng)

    coordinate = (
        X * direction[0]
        + Y * direction[1]
        + Z * direction[2]
    )

    coordinate += (
        noise(X, Y, Z, rng, 0.8, 2.5) - 0.5
    ) * rng.uniform(0.15, 0.38)

    return coordinate, direction


def angular_mass(X, Y, Z, rng):
    """
    Irregular hard mass made from intersecting directional planes.

    This is intentionally angular/irregular rather than spherical.
    """
    center = rng.uniform(-0.10, 0.10, 3)

    coords = np.stack(
        [
            X - center[0],
            Y - center[1],
            Z - center[2],
        ],
        axis=-1,
    )

    field = np.full(
        X.shape,
        10.0,
        dtype=np.float32,
    )

    planes = []

    for _ in range(int(rng.integers(5, 9))):
        direction = unit_vector(rng)
        offset = rng.uniform(0.48, 0.78)

        plane_distance = (
            coords[..., 0] * direction[0]
            + coords[..., 1] * direction[1]
            + coords[..., 2] * direction[2]
        )

        field = np.minimum(
            field,
            offset - plane_distance,
        )

        planes.append(
            {
                "direction": direction.tolist(),
                "offset": float(offset),
            }
        )

    irregularity = (
        noise(X, Y, Z, rng, 1.2, 3.0) - 0.5
    ) * rng.uniform(0.08, 0.18)

    return (
        (field + irregularity) > 0,
        {
            "construction": "directional_planes",
            "planes": planes,
        },
    )


def network_mass(X, Y, Z, rng):
    """Several substantial intersecting linear masses."""
    fields = []
    segments = []

    for _ in range(int(rng.integers(3, 6))):
        start = rng.uniform(-0.75, 0.75, 3)
        end = np.clip(
            start + unit_vector(rng) * rng.uniform(0.45, 1.0),
            -0.92,
            0.92,
        )
        radius = rng.uniform(0.08, 0.16)

        fields.append(
            segment_field(
                X,
                Y,
                Z,
                start,
                end,
                radius,
            )
        )

        segments.append(
            {
                "start": start.tolist(),
                "end": end.tolist(),
                "radius": float(radius),
            }
        )

    return (
        np.max(fields, axis=0)
        > rng.uniform(0.38, 0.52),
        {
            "construction": "intersecting_network",
            "segments": segments,
        },
    )


def field_cluster(X, Y, Z, rng):
    """
    Multiple masses responding to a shared environmental field.

    Coalescence is allowed, but a single blob is NOT required.
    """
    count = int(rng.integers(4, 10))

    positions = [
        rng.uniform(-0.72, 0.72, 3)
        for _ in range(count)
    ]

    radii = [
        rng.uniform(0.09, 0.18)
        for _ in positions
    ]

    field_direction = unit_vector(rng)

    field = norm(
        X * field_direction[0]
        + Y * field_direction[1]
        + Z * field_direction[2]
    )

    field = np.clip(
        field
        + 0.35 * (
            noise(X, Y, Z, rng, 1.0, 2.6)
            - 0.5
        ),
        0,
        1,
    )

    attraction = rng.uniform(0.08, 0.55)

    for _ in range(int(rng.integers(5, 12))):
        new_positions = []

        for i, position in enumerate(positions):
            position = np.asarray(
                position,
                dtype=np.float32,
            )

            index = np.clip(
                ((position + 1) * 0.5 * (len(X) - 1))
                .astype(int),
                0,
                len(X) - 1,
            )

            growth = (
                0.010
                + 0.035 * float(field[tuple(index)])
            )

            radii[i] = min(
                0.30,
                radii[i] + growth,
            )

            if len(positions) > 1:
                distances = [
                    (
                        np.linalg.norm(
                            positions[j] - position
                        )
                        if j != i
                        else 1e9
                    )
                    for j in range(len(positions))
                ]

                nearest = int(np.argmin(distances))
                delta = (
                    np.asarray(positions[nearest])
                    - position
                )
                distance = np.linalg.norm(delta)

                if distance > 1e-6:
                    position = (
                        position
                        + delta / distance
                        * attraction
                        * 0.014
                    )

            new_positions.append(position)

        positions = new_positions

    fields = []

    for position, radius in zip(
        positions,
        radii,
    ):
        fields.append(
            np.exp(
                -(
                    ((X - position[0]) / radius) ** 2
                    + ((Y - position[1]) / radius) ** 2
                    + ((Z - position[2]) / radius) ** 2
                )
            ).astype(np.float32)
        )

    return np.max(fields, axis=0), {
        "construction": "shared_environmental_field",
        "initial_forms": count,
        "field_direction": field_direction.tolist(),
        "attraction": float(attraction),
        "coalescence_allowed": True,
    }


def scaffold(X, Y, Z, rng):
    """
    Select a morphology instead of selecting a universal base object.
    """
    kind = str(
        rng.choice(
            [
                "angular",
                "network",
                "cluster",
                "branching",
                "layers",
            ]
        )
    )

    if kind == "angular":
        volume, details = angular_mass(
            X, Y, Z, rng
        )

    elif kind == "network":
        volume, details = network_mass(
            X, Y, Z, rng
        )

    elif kind == "cluster":
        field, details = field_cluster(
            X, Y, Z, rng
        )
        volume = field > rng.uniform(
            0.38,
            0.50,
        )

    elif kind == "branching":
        field, details = branching_field(
            X,
            Y,
            Z,
            rng,
            depth=int(rng.integers(2, 4)),
        )
        volume = field > rng.uniform(
            0.40,
            0.52,
        )

    else:
        coordinate, direction = deposition_field(
            X, Y, Z, rng
        )

        bands = np.sin(
            coordinate
            * np.pi
            * rng.uniform(2.5, 5.5)
        )

        volume = bands > rng.uniform(
            -0.15,
            0.18,
        )

        details = {
            "construction": "layered_scaffold",
            "orientation": direction.tolist(),
        }

    return volume, {
        "scaffold_type": kind,
        "scaffold": details,
    }


def surface(mass):
    """Exterior voxel shell."""
    interior = mass.copy()

    for axis in range(3):
        interior &= (
            np.roll(mass, 1, axis)
            & np.roll(mass, -1, axis)
        )

    shell = mass & ~interior

    shell[[0, -1], :, :] = False
    shell[:, [0, -1], :] = False
    shell[:, :, [0, -1]] = False

    return shell


def cavitation(X, Y, Z, rng):
    """
    Cavities carved into a varied mass.

    Cavities can be internal or break through the surface.
    """
    mass, base = scaffold(
        X, Y, Z, rng
    )

    cavity_fields = []
    cavity_meta = []

    for _ in range(
        int(rng.integers(12, 24))
    ):
        material = np.argwhere(mass)

        if len(material) == 0:
            break

        voxel = material[
            int(rng.integers(len(material)))
        ]

        center = np.array(
            [
                X[tuple(voxel)],
                Y[tuple(voxel)],
                Z[tuple(voxel)],
            ]
        )

        scale = rng.uniform(
            0.06,
            0.20,
            3,
        )

        smoothness = rng.uniform(
            0.65,
            1.35,
        )

        field = np.exp(
            -(
                ((X - center[0]) / scale[0]) ** 2
                + ((Y - center[1]) / scale[1]) ** 2
                + ((Z - center[2]) / scale[2]) ** 2
            )
            * smoothness
        )

        cavity_fields.append(field)

        cavity_meta.append(
            {
                "center": center.tolist(),
                "scale": scale.tolist(),
                "smoothness": float(smoothness),
            }
        )

    if cavity_fields:
        holes = (
            np.max(cavity_fields, axis=0)
            > rng.uniform(0.40, 0.58)
        )
        mass[holes] = False

    return mass, {
        "base": base,
        "cavity_count": len(cavity_fields),
        "connectivity_bias": "high",
        "cavities": cavity_meta,
    }


def porosity(X, Y, Z, rng):
    """
    Surface-condition porosity on a varied morphology.
    """
    mass, base = scaffold(
        X, Y, Z, rng
    )

    surface_voxels = np.argwhere(
        surface(mass)
    )

    holes = np.zeros_like(
        mass,
        dtype=bool,
    )

    pore_meta = []

    if len(surface_voxels):
        for _ in range(
            int(rng.integers(20, 42))
        ):
            voxel = surface_voxels[
                int(rng.integers(len(surface_voxels)))
            ]

            center = np.array(
                [
                    X[tuple(voxel)],
                    Y[tuple(voxel)],
                    Z[tuple(voxel)],
                ]
            )

            scale = rng.uniform(
                0.07,
                0.16,
                3,
            )

            field = np.exp(
                -(
                    ((X - center[0]) / scale[0]) ** 2
                    + ((Y - center[1]) / scale[1]) ** 2
                    + ((Z - center[2]) / scale[2]) ** 2
                )
            )

            holes |= (
                field
                > rng.uniform(0.52, 0.70)
            )

            pore_meta.append(
                {
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                }
            )

        mass[holes] = False

    return mass, {
        "base": base,
        "pore_count": len(pore_meta),
        "surface_condition": True,
        "pores": pore_meta,
    }


def erosion(X, Y, Z, rng):
    """
    Surface-only environmental erosion.

    The underlying morphology is varied; erosion is the process applied to it.
    """
    mass, base = scaffold(
        X, Y, Z, rng
    )

    agent = str(
        rng.choice(
            ["water", "wind"]
        )
    )

    direction = unit_vector(rng)
    duration = rng.uniform(
        0.08,
        0.92,
    )

    intensity = float(
        np.clip(
            duration * rng.uniform(
                0.75,
                1.25,
            ),
            0.05,
            1.0,
        )
    )

    directional = norm(
        X * direction[0]
        + Y * direction[1]
        + Z * direction[2]
    )

    roughness = noise(
        X,
        Y,
        Z,
        rng,
        1.5,
        4.5,
    )

    if agent == "water":
        erosion_field = (
            0.62 * (1 - directional)
            + 0.38
            * (
                np.sin(
                    directional
                    * np.pi
                    * rng.uniform(3, 6)
                    + roughness * np.pi
                )
                + 1
            )
            / 2
        )
    else:
        erosion_field = (
            0.68 * directional
            + 0.32
            * np.abs(
                np.sin(
                    directional
                    * np.pi
                    * rng.uniform(4, 8)
                    + roughness * 2
                )
            )
        )

    erosion_field = norm(
        erosion_field
    )

    for step in range(
        1 + int(duration * 4)
    ):
        shell = surface(mass)

        probability = (
            erosion_field
            * intensity
            * (
                0.58
                if step == 0
                else 0.18
            )
        )

        mass[
            shell
            & (
                rng.random(mass.shape)
                < probability
            )
        ] = False

    return mass, {
        "base": base,
        "agent": agent,
        "duration": float(duration),
        "intensity": intensity,
        "direction": direction.tolist(),
        "surface_only_primary": True,
    }


def layering(X, Y, Z, rng):
    """
    Accumulation/deposition rather than bands cut into a blob.
    """
    coordinate, direction = deposition_field(
        X, Y, Z, rng
    )

    layer_count = int(
        rng.integers(3, 8)
    )

    edges = np.linspace(
        coordinate.min(),
        coordinate.max(),
        layer_count + 1,
    )

    field_noise = noise(
        X,
        Y,
        Z,
        rng,
        1.0,
        3.0,
    )

    material = np.zeros_like(
        X,
        dtype=bool,
    )

    layers = []

    for i in range(
        layer_count
    ):
        lo = edges[i]
        hi = edges[i + 1]

        thickness_variation = (
            field_noise - 0.5
        ) * rng.uniform(
            0.08,
            0.20,
        )

        band = (
            (coordinate >= lo + thickness_variation)
            & (coordinate < hi + thickness_variation)
        )

        persistence = rng.uniform(
            0.55,
            1.0,
        )

        material |= (
            band
            & (field_noise < persistence)
        )

        layers.append(
            {
                "index": i,
                "persistence": float(
                    persistence
                ),
            }
        )

    # Finite spatial extent without an ellipsoidal envelope.
    envelope = (
        np.abs(X)
        + np.abs(Y)
        + np.abs(Z)
        < rng.uniform(
            1.35,
            1.75,
        )
    )

    material &= envelope

    return material, {
        "layer_count": layer_count,
        "orientation": direction.tolist(),
        "continuity": "variable",
        "overlap": float(
            rng.uniform(0.25, 0.85)
        ),
        "layers": layers,
    }


def clustering(X, Y, Z, rng):
    field, metadata = field_cluster(
        X, Y, Z, rng
    )

    return (
        field
        > rng.uniform(
            0.36,
            0.49,
        ),
        metadata,
    )


def fragmentation(X, Y, Z, rng):
    """
    Partial fracture without displacement or detached fragments.
    """
    mass, base = scaffold(
        X, Y, Z, rng
    )

    fracture_field = np.zeros_like(
        mass,
        dtype=np.float32,
    )

    fractures = []

    for _ in range(
        int(rng.integers(3, 8))
    ):
        direction = unit_vector(rng)
        offset = rng.uniform(
            -0.45,
            0.45,
        )

        plane = (
            X * direction[0]
            + Y * direction[1]
            + Z * direction[2]
            - offset
        )

        warp = (
            noise(
                X,
                Y,
                Z,
                rng,
                1.5,
                4.0,
            )
            - 0.5
        ) * rng.uniform(
            0.04,
            0.12,
        )

        width = rng.uniform(
            0.018,
            0.055,
        )

        fracture_field = np.maximum(
            fracture_field,
            np.exp(
                -((plane + warp) / width) ** 2
            ),
        )

        fractures.append(
            {
                "direction": direction.tolist(),
                "offset": float(offset),
                "width": float(width),
            }
        )

    strength = rng.uniform(
        0.15,
        0.72,
    )

    mass[
        mass
        & (
            fracture_field
            > rng.uniform(0.45, 0.68)
        )
        & (
            rng.random(mass.shape)
            < strength * 0.58
        )
    ] = False

    return mass, {
        "base": base,
        "fracture_count": len(fractures),
        "fracture_intensity": float(
            strength
        ),
        "displacement": 0.0,
        "complete_fracture": False,
        "fractures": fractures,
    }


GENERATORS = {
    "branching": branching,
    "cavitation": cavitation,
    "porosity": porosity,
    "erosion": erosion,
    "layering": layering,
    "clustering": clustering,
    "fragmentation": fragmentation,
}


def hybrid(X, Y, Z, rng):
    names = list(
        rng.choice(
            PHENOMENA,
            size=int(
                rng.integers(2, 4)
            ),
            replace=False,
        )
    )

    mode = str(
        rng.choice(
            ["sequential", "simultaneous"]
        )
    )

    processes = []

    if mode == "sequential":
        volume = None

        for name in names:
            generated, parameters = GENERATORS[name](
                X,
                Y,
                Z,
                rng,
            )

            if volume is None:
                volume = generated.copy()

            elif name in {
                "cavitation",
                "porosity",
                "erosion",
                "fragmentation",
            }:
                volume &= generated

            else:
                volume |= generated

            processes.append(
                {
                    "phenomenon": name,
                    "parameters": parameters,
                }
            )

    else:
        fields = []

        for name in names:
            generated, parameters = GENERATORS[name](
                X,
                Y,
                Z,
                rng,
            )

            fields.append(
                generated.astype(
                    np.float32
                )
            )

            processes.append(
                {
                    "phenomenon": name,
                    "parameters": parameters,
                }
            )

        volume = (
            np.mean(fields, axis=0)
            > rng.uniform(
                0.30,
                0.48,
            )
        )

    return volume, {
        "mode": mode,
        "phenomena": names,
        "processes": processes,
    }


def connected_components(volume):
    """
    6-neighbour connected-component count.
    Kept intentionally simple so no scipy dependency is required.
    """
    if not volume.any():
        return 0

    seen = np.zeros_like(
        volume,
        dtype=bool,
    )

    count = 0

    for start in np.argwhere(volume):
        start = tuple(
            map(int, start)
        )

        if seen[start]:
            continue

        count += 1
        stack = [start]
        seen[start] = True

        while stack:
            x, y, z = stack.pop()

            for dx, dy, dz in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ):
                q = (
                    x + dx,
                    y + dy,
                    z + dz,
                )

                if (
                    0 <= q[0] < volume.shape[0]
                    and 0 <= q[1] < volume.shape[1]
                    and 0 <= q[2] < volume.shape[2]
                    and volume[q]
                    and not seen[q]
                ):
                    seen[q] = True
                    stack.append(q)

    return count


def valid(volume, phenomenon):
    occupancy = float(
        volume.mean()
    )

    components = connected_components(
        volume
    )

    valid_sample = (
        0.015
        <= occupancy
        <= 0.55
    )

    # These are intended to remain spatially associated.
    if phenomenon in {
        "cavitation",
        "porosity",
        "erosion",
        "layering",
        "fragmentation",
    }:
        valid_sample &= (
            components == 1
        )

    return bool(valid_sample), {
        "occupancy": occupancy,
        "connected_components": components,
    }


def main():
    os.makedirs(
        "data",
        exist_ok=True,
    )

    resolution = int(
        CONFIG.resolution
    )

    dataset_size = int(
        CONFIG.dataset_size
    )

    X, Y, Z = grid(
        resolution
    )

    master = np.random.default_rng(
        int(CONFIG.seed)
    )

    # Approximately 70% single-phenomenon examples,
    # 30% hybrids, as in the previous dataset design.
    single_count = int(
        round(dataset_size * 0.70)
    )

    labels = (
        PHENOMENA
        * (single_count // len(PHENOMENA))
    )

    labels += PHENOMENA[
        : single_count % len(PHENOMENA)
    ]

    labels += [
        "hybridization"
    ] * (
        dataset_size
        - single_count
    )

    master.shuffle(labels)

    data = np.zeros(
        (
            dataset_size,
            resolution,
            resolution,
            resolution,
            1,
        ),
        dtype=np.float32,
    )

    metadata = []

    for index, phenomenon in enumerate(
        tqdm(
            labels,
            desc="Generating morphology-first dataset",
        )
    ):
        seed = int(
            master.integers(
                0,
                2**32 - 1,
            )
        )

        accepted = False

        for attempt in range(40):
            rng = np.random.default_rng(
                seed + attempt
            )

            if phenomenon == "hybridization":
                volume, parameters = hybrid(
                    X,
                    Y,
                    Z,
                    rng,
                )
            else:
                volume, parameters = GENERATORS[
                    phenomenon
                ](
                    X,
                    Y,
                    Z,
                    rng,
                )

            ok, validation = valid(
                volume,
                phenomenon,
            )

            if ok:
                accepted = True
                break

        data[
            index,
            ...,
            0,
        ] = volume.astype(
            np.float32
        )

        validation[
            "validation_exhausted"
        ] = not accepted

        metadata.append(
            {
                "id": index,
                "seed": seed,
                "phenomenon": phenomenon,
                "phenomena": parameters.get(
                    "phenomena",
                    [phenomenon],
                ),
                "parameters": parameters,
                "validation": validation,
            }
        )

    # Same training-data interface as the existing GAN.
    np.savez_compressed(
        "data/procedural.npz",
        structures=data,
    )

    with open(
        "data/procedural_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    np.savez_compressed(
        "data/procedural_metadata.npz",
        ids=np.arange(
            dataset_size,
            dtype=np.int32,
        ),
        seeds=np.array(
            [
                item["seed"]
                for item in metadata
            ],
            dtype=np.uint64,
        ),
        phenomena=np.array(
            [
                item["phenomenon"]
                for item in metadata
            ]
        ),
        occupancy=np.array(
            [
                item["validation"][
                    "occupancy"
                ]
                for item in metadata
            ],
            dtype=np.float32,
        ),
        connected_components=np.array(
            [
                item["validation"][
                    "connected_components"
                ]
                for item in metadata
            ],
            dtype=np.int32,
        ),
    )

    print()
    print("Wrote data/procedural.npz")
    print("Shape:", data.shape)
    print(
        "Mean occupancy:",
        float(data.mean()),
    )
    print(
        "Wrote data/procedural_metadata.json"
    )
    print(
        "Wrote data/procedural_metadata.npz"
    )


if __name__ == "__main__":
    main()
