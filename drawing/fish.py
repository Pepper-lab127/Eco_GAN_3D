"""Fish of the Red River at Lockport, drawn as fine outline profiles.

Six species: the most important for the river's ecology and fishery, plus
those with a conservation status. Presence is from Manitoba's waterbody data;
lengths are typical adults; winter positions are by known habit (fish of the
Red River overwinter in deep, slow water), not observations.
"""

import numpy as np

# name, typical adult length (m), body depth / length, winter habit,
# preferred height above the bed as a share of the water depth, status, profile
SPECIES = [
    dict(name="LAKE STURGEON", latin="Acipenser fulvescens", length=1.3, depth=0.14, zone=(0.0, 0.12),
         habit="on the bed of the deepest pools", status="ENDANGERED (COSEWIC, Western Hudson Bay populations)",
         tail="shark", barbels=True, scutes=True),
    dict(name="CHANNEL CATFISH", latin="Ictalurus punctatus", length=0.75, depth=0.17, zone=(0.0, 0.2),
         habit="near the bed in deep holes, nearly dormant", status="", tail="fork", barbels=True),
    dict(name="WALLEYE", latin="Sander vitreus", length=0.5, depth=0.19, zone=(0.15, 0.45),
         habit="deep pool below the dam", status="", tail="fork", spiny=True),
    dict(name="BIGMOUTH BUFFALO", latin="Ictiobus cyprinellus", length=0.6, depth=0.33, zone=(0.25, 0.6),
         habit="slow, deeper water", status="SPECIAL CONCERN (COSEWIC, Saskatchewan–Nelson populations)",
         tail="fork"),
    dict(name="FRESHWATER DRUM", latin="Aplodinotus grunniens", length=0.45, depth=0.3, zone=(0.05, 0.35),
         habit="near the bed", status="", tail="round", spiny=True),
    dict(name="GOLDEYE", latin="Hiodon alosoides", length=0.35, depth=0.26, zone=(0.55, 0.85),
         habit="mid-water to under the ice", status="", tail="fork"),
]


def outline(x, y, length, depth_ratio, facing=1, tail="fork", spiny=False, barbels=False, scutes=False):
    """Path data for a fish profile, nose at (x, y) heading `facing` (+1 east),
    in sheet mm. Returns a list of path strings (body, fins, details)."""
    L, D = length, length * depth_ratio
    t = np.linspace(0, 1, 40)
    # Body: a teardrop, deepest a third of the way back, narrowing to the tail.
    half = D / 2 * np.sin(np.pi * np.clip(t, 0, 1) ** 0.75) * (1 - 0.65 * t ** 2)
    half = np.maximum(half, D * 0.05 * t)
    bx = x - facing * t * L * 0.82
    top = " L".join(f"{a:.2f},{y - h:.2f}" for a, h in zip(bx, half))
    bot = " L".join(f"{a:.2f},{y + h * 0.85:.2f}" for a, h in zip(bx[::-1], half[::-1]))
    parts = [f"M{top} L{bot} Z"]
    # Tail.
    tx = x - facing * L * 0.82
    th = D * 0.08
    if tail == "fork":
        parts.append(f"M{tx:.2f},{y - th:.2f} L{tx - facing * L * 0.18:.2f},{y - D * 0.48:.2f} "
                     f"L{tx - facing * L * 0.11:.2f},{y:.2f} L{tx - facing * L * 0.18:.2f},{y + D * 0.42:.2f} "
                     f"L{tx:.2f},{y + th:.2f}")
    elif tail == "round":
        parts.append(f"M{tx:.2f},{y - th:.2f} Q{tx - facing * L * 0.2:.2f},{y - D * 0.35:.2f} "
                     f"{tx - facing * L * 0.18:.2f},{y:.2f} Q{tx - facing * L * 0.2:.2f},{y + D * 0.35:.2f} "
                     f"{tx:.2f},{y + th:.2f}")
    else:  # shark: long upper lobe (sturgeon)
        parts.append(f"M{tx:.2f},{y - th:.2f} L{tx - facing * L * 0.2:.2f},{y - D * 0.7:.2f} "
                     f"L{tx - facing * L * 0.08:.2f},{y + D * 0.05:.2f} L{tx - facing * L * 0.1:.2f},{y + D * 0.3:.2f} "
                     f"L{tx:.2f},{y + th:.2f}")
    # Dorsal fin(s).
    d0 = x - facing * L * 0.35
    if spiny:
        parts.append(f"M{d0:.2f},{y - D * 0.47:.2f} L{d0 - facing * L * 0.05:.2f},{y - D * 0.85:.2f} "
                     f"L{d0 - facing * L * 0.16:.2f},{y - D * 0.48:.2f}")
        parts.append(f"M{d0 - facing * L * 0.18:.2f},{y - D * 0.43:.2f} L{d0 - facing * L * 0.22:.2f},{y - D * 0.72:.2f} "
                     f"L{d0 - facing * L * 0.32:.2f},{y - D * 0.33:.2f}")
    else:
        parts.append(f"M{d0:.2f},{y - D * 0.47:.2f} L{d0 - facing * L * 0.07:.2f},{y - D * 0.8:.2f} "
                     f"L{d0 - facing * L * 0.18:.2f},{y - D * 0.42:.2f}")
    # Pectoral fin and gill line.
    g = x - facing * L * 0.17
    parts.append(f"M{g:.2f},{y - D * 0.3:.2f} Q{g + facing * L * 0.03:.2f},{y:.2f} {g:.2f},{y + D * 0.3:.2f}")
    parts.append(f"M{g - facing * L * 0.02:.2f},{y + D * 0.15:.2f} L{g - facing * L * 0.12:.2f},{y + D * 0.42:.2f}")
    # Eye.
    e = x - facing * L * 0.06
    r = max(D * 0.05, 0.12)
    parts.append(f"M{e + r:.2f},{y - D * 0.12:.2f} a{r:.2f},{r:.2f} 0 1 0 {-2 * r:.2f},0 a{r:.2f},{r:.2f} 0 1 0 {2 * r:.2f},0")
    if barbels:
        for k in (0.25, 0.4):
            parts.append(f"M{x - facing * L * 0.02:.2f},{y + D * 0.15:.2f} "
                         f"q{facing * L * 0.03:.2f},{D * k:.2f} {facing * L * 0.01:.2f},{D * (k + 0.25):.2f}")
    if scutes:
        for k in np.linspace(0.2, 0.75, 9):
            sx = x - facing * L * k
            parts.append(f"M{sx:.2f},{y - D * 0.3:.2f} l{-facing * L * 0.015:.2f},{-D * 0.06:.2f} "
                         f"l{-facing * L * 0.015:.2f},{D * 0.06:.2f}")
    return parts
