"""Stones on the shore of the Red River at Lockport, and why they are there.

  - LIMESTONE SLABS: the Selkirk Member of the Red River Formation (Ordovician,
    the mottled limestone also quarried as Tyndall stone) outcropped as ledges
    along the banks and across the river at the St. Andrews Rapids, just
    upstream of this section. River ice breaks slabs off and shoves them
    downstream and up the bank: flat, angular, bedded, mottled.
  - CARBONATE COBBLES AND BOULDERS from the till: the region's till is
    carbonate-rich (ground from the Paleozoic bedrock by southward-flowing
    ice): subangular, pale.
  - GRANITE AND GNEISS ERRATICS from the till: carried by the ice from the
    Precambrian Shield to the east and northeast; hard, so they outlast the
    softer stones and end up rounded.

As the river and its spring ice wash the clay and till out of the bank, the
stones are left behind as a lag along the shore. They are drawn from below
the low-water line up to the reach of the spring ice (the highest April
level on record), thinning upward. Proportions and sizes are typical, not
counted.
"""

import math

import numpy as np

KINDS = [  # name, share, size range (m, lognormal median, sigma), aspect
    ("limestone_slab", 0.35, (0.7, 0.45), 0.35),
    ("carbonate", 0.35, (0.4, 0.5), 0.7),
    ("granite", 0.2, (0.55, 0.5), 0.8),
    ("gneiss", 0.1, (0.5, 0.5), 0.65),
]


def outline(rng, cx, cy, rx, ry, kind, n=18):
    """Points of a stone's outline (mm)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if kind == "limestone_slab":
        # Angular: few corners, flat top and bottom.
        k = rng.integers(5, 8)
        a = np.sort(rng.uniform(0, 2 * math.pi, k))
        r = rng.uniform(0.85, 1.1, k)
        pts = np.stack([np.cos(a) * rx * r, np.clip(np.sin(a) * ry * 1.6, -ry, ry)], 1)
    else:
        rough = dict(carbonate=0.12, granite=0.05, gneiss=0.06)[kind]
        r = 1 + rng.normal(0, rough, n)
        r = (r + np.roll(r, 1) + np.roll(r, -1)) / 3
        pts = np.stack([np.cos(a) * rx * r, np.sin(a) * ry * r], 1)
    tilt = rng.normal(0, 0.12)
    c, s = math.cos(tilt), math.sin(tilt)
    pts = pts @ np.array([[c, s], [-s, c]])
    return pts + [cx, cy]


def texture(rng, pts, kind, rx, ry):
    """Inner marks: bedding and burrow mottles (limestone), speckle (granite),
    banding (gneiss). Returns (line paths, dots)."""
    cx, cy = pts.mean(0)
    lines, dots = [], []
    if kind == "limestone_slab":
        for k in range(1, 3):
            y = cy - ry + 2 * ry * k / 3
            lines.append(f"M{cx - rx * 0.8:.2f},{y:.2f}L{cx + rx * 0.8:.2f},{y + rng.normal(0, 0.05):.2f}")
        m = int(rx * ry * 2)
        dots += [(cx + rng.uniform(-0.7, 0.7) * rx, cy + rng.uniform(-0.6, 0.6) * ry, 0.18) for _ in range(m)]
    elif kind == "granite":
        m = int(rx * ry * 10)
        dots += [(cx + rng.uniform(-0.8, 0.8) * rx, cy + rng.uniform(-0.7, 0.7) * ry, 0.08) for _ in range(m)]
    elif kind == "gneiss":
        for k in range(-2, 3):
            y0 = cy + k * ry * 0.3
            xs = np.linspace(cx - rx * 0.8, cx + rx * 0.8, 8)
            ys = y0 + np.sin(xs * 1.7 + k) * ry * 0.08
            lines.append("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in zip(xs, ys)))
    else:
        m = int(rx * ry * 2)
        dots += [(cx + rng.uniform(-0.7, 0.7) * rx, cy + rng.uniform(-0.6, 0.6) * ry, 0.1) for _ in range(m)]
    return lines, dots


def shore(svg, layers, sec, ch, ground, lo_z, hi_z, rng, per_m=1.6, ink="#111111"):
    """Draw the shore lag between levels lo_z and hi_z. Returns counts by kind."""
    names = [k[0] for k in KINDS]
    p = np.array([k[1] for k in KINDS])
    counts = {}
    outlines, lines, dots = [], [], []
    xs = np.nonzero((ground >= lo_z) & (ground <= hi_z))[0]
    if not len(xs):
        return counts
    x_lo, x_hi = xs.min(), xs.max() + 1
    n = int((x_hi - x_lo) * per_m)
    for _ in range(n):
        x = rng.uniform(x_lo, x_hi)
        gz = float(np.interp(x, ch, ground))
        up = (gz - lo_z) / max(hi_z - lo_z, 1e-6)
        if rng.random() > 1.1 - up:          # thinner toward the ice's reach
            continue
        i = rng.choice(len(KINDS), p=p / p.sum())
        kind, _, (med, sig), asp = KINDS[i]
        size = float(np.clip(rng.lognormal(math.log(med), sig), 0.08, 1.6))
        rx, ry = size / 2 * sec.sh, size / 2 * asp * sec.sv
        cx = sec.X(x)
        cy = sec.Y(gz) - ry * rng.uniform(-0.5, 0.6)       # partly buried
        pts = outline(rng, cx, cy, rx, ry, kind)
        outlines.append("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + "Z")
        ln, dt = texture(rng, pts, kind, rx, ry)
        lines += ln
        dots += dt
        counts[names[i]] = counts.get(names[i], 0) + 1
    svg.layer(layers["stones"])
    svg.path("".join(outlines), fill="#ffffff", stroke=ink, stroke_width=0.08, stroke_linejoin="round")
    svg.layer(layers["marks"])
    svg.path("".join(lines), fill="none", stroke=ink, stroke_width=0.04, opacity=0.8)
    for size in sorted({d[2] for d in dots}):
        svg.path("".join(f"M{a:.2f},{b:.2f}h0" for a, b, s in dots if s == size), fill="none", stroke=ink,
                 stroke_width=size, stroke_linecap="round", opacity=0.75)
    return counts
