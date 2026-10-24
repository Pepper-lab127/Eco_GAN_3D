"""Riparian trees of the Red River at Lockport, drawn by species.

Each tree is grown from its species' habit (how high the trunk runs before it
forks, how many leaders, the angle and droop of its limbs, how crooked they
are, how its leaves gather) and then fitted to the tree measured by the lidar:
its height and its crown width. Trunk and limbs are filled, tapering by the
pipe model; leaves are stippled at the twig ends.

Species come from the Lockport plant list (data/lockport/ecology/
lockport_flora.csv); which tree is which is placed by that list's habitat
preferences against how often the ground at its base is under water and
whether it stands at the woodland margin (see zone_species). Other species
here (willow, cottonwood, ash, basswood) are kept for other sites.
"""

import math

import numpy as np

# name, latin, habit parameters
SPECIES = {
    "willow": dict(name="PEACH-LEAVED WILLOW", latin="Salix amygdaloides", trunk=0.12, leaders=3, spread=0.35,
                   angle=0.45, ratio=0.72, depth=7, droop=0.10, crook=0.10, twig=0.8, leaf=(10, 0.03, 0.15),
                   root=(0.6, 1.4)),
    "cottonwood": dict(name="PLAINS COTTONWOOD", latin="Populus deltoides", trunk=0.42, leaders=3, spread=0.45,
                       angle=0.55, ratio=0.70, depth=7, droop=-0.05, crook=0.12, twig=0.6, leaf=(22, 0.06, 0.19),
                       root=(0.7, 1.6)),
    "maple": dict(name="MANITOBA MAPLE", latin="Acer negundo", trunk=0.10, leaders=3, spread=0.65, angle=0.6,
                  ratio=0.68, depth=7, droop=0.0, crook=0.22, twig=0.7, leaf=(14, 0.045, 0.17), root=(0.6, 1.1),
                  lean=0.25),
    "ash": dict(name="GREEN ASH", latin="Fraxinus pennsylvanica", trunk=0.38, leaders=2, spread=0.35, angle=0.5,
                ratio=0.72, depth=7, droop=0.02, crook=0.06, twig=0.8, leaf=(14, 0.04, 0.16), root=(0.8, 1.0)),
    "elm": dict(name="AMERICAN ELM", latin="Ulmus americana", trunk=0.28, leaders=4, spread=0.55, angle=0.38,
                ratio=0.76, depth=8, droop=0.18, crook=0.05, twig=1.0, leaf=(9, 0.035, 0.14), root=(0.7, 1.5)),
    "oak": dict(name="BUR OAK", latin="Quercus macrocarpa", trunk=0.34, leaders=3, spread=0.9, angle=0.95,
                ratio=0.66, depth=7, droop=0.0, crook=0.35, twig=0.6, leaf=(18, 0.05, 0.18), root=(1.3, 1.0)),
    "aspen": dict(name="TREMBLING ASPEN", latin="Populus tremuloides", trunk=0.55, leaders=1, spread=0.0, angle=0.55,
                  ratio=0.6, depth=6, droop=0.0, crook=0.07, twig=0.6, leaf=(11, 0.03, 0.14), root=(0.6, 1.4),
                  excurrent=True, pale_bark=True),
    "basswood": dict(name="BASSWOOD", latin="Tilia americana", trunk=0.9, leaders=1, spread=0.0, angle=0.85,
                     ratio=0.62, depth=7, droop=0.08, crook=0.04, twig=0.7, leaf=(16, 0.045, 0.19), root=(1.0, 1.0),
                     excurrent=True),
}


def zone_species(flood_share, above_crest, at_edge, rng):
    """Species for a tree, from the Lockport plant list (data/lockport/ecology/
    lockport_flora.csv) and its habitat preferences: Manitoba maple on
    floodplains (ground the open-water levels reach), American elm on the
    riverbank below the 1997 crest, bur oak on the well-drained ground above
    it, trembling aspen at the woodland margin."""
    if flood_share > 0.0:
        return "maple"
    if not above_crest:
        return "elm"
    if at_edge:
        return "aspen"
    return "oak"


def grow(sp, rng):
    """Skeleton in unit space (ground at 0, up is +y): list of segments
    (x0, y0, x1, y1, order, is_tip)."""
    p = SPECIES[sp]
    segs = []

    def limb(x, y, ang, length, order, depth):
        steps = 3
        for k in range(steps):
            ang += rng.normal(0, p["crook"]) + p["droop"] * order * 0.12 * math.sin(ang)
            ang = max(-1.75, min(1.75, ang))                 # limbs never turn back into the ground
            x2 = x + math.sin(ang) * length / steps
            y2 = y + math.cos(ang) * length / steps
            segs.append((x, y, x2, y2, order, depth == 0 and k == steps - 1))
            x, y = x2, y2
            # Side shoots along the limb, so limbs read as branches, not forks.
            if depth >= 2 and order >= 1 and rng.random() < p.get("lateral", 0.45):
                side = rng.choice([-1, 1])
                limb(x, y, ang + side * p["angle"] * rng.uniform(0.9, 1.4), length * 0.42, order + 1, depth - 2)
        if depth == 0:
            return
        if p.get("excurrent") and order == 0:
            # A continuing leader with whorls of side limbs (basswood).
            for side in (-1, 1):
                limb(x, y, side * p["angle"] + rng.normal(0, 0.15), length * 0.55, order + 1, depth - 1)
            limb(x, y, ang * 0.5 + rng.normal(0, 0.05), length * 0.8, order, depth - 1)
            return
        n = 2 if rng.random() < 0.75 else 3
        for i in range(n):
            a2 = ang + (i - (n - 1) / 2) * p["angle"] * rng.uniform(0.7, 1.2)
            a2 = a2 * (1 - 0.08) if order < 2 else a2
            limb(x, y, a2, length * p["ratio"] * rng.uniform(0.85, 1.1), order + 1, depth - 1)

    trunk = p["trunk"]
    lean = p.get("lean", 0.0) * rng.choice([-1, 1])
    if p.get("excurrent"):
        segs.append((0, 0, 0, trunk * 0.35, 0, False))
        limb(0, trunk * 0.35, rng.normal(0, 0.04), 0.22, 0, p["depth"])
    else:
        segs.append((0, 0, math.sin(lean) * trunk, math.cos(lean) * trunk, 0, False))
        x0, y0 = segs[-1][2], segs[-1][3]
        L = max(p["leaders"], 1)
        for i in range(L):
            a = lean + (i - (L - 1) / 2) * p["spread"] / max(L - 1, 1) * 2 + rng.normal(0, 0.08)
            limb(x0, y0, a, 0.24, 1, p["depth"] - 1)
    return segs


def draw(svg, layers, sp, x_mm, y_ground_mm, height_mm, crown_w_mm, rng, ink="#111111", dbh_mm=None):
    """Draw one tree; returns the drawn crown's (left, right, top) in mm."""
    p = SPECIES[sp]
    segs = grow(sp, rng)
    xs = np.array([[s[0], s[2]] for s in segs]).ravel()
    ys = np.array([[s[1], s[3]] for s in segs]).ravel()
    sy = height_mm / max(ys.max(), 1e-6)
    half = max(abs(xs.min()), abs(xs.max()), 1e-6)
    sx = (crown_w_mm / 2) / half
    sx = min(sx, sy * 1.6)

    def P(x, y):
        return x_mm + x * sx, y_ground_mm - y * sy

    dbh = dbh_mm or height_mm / 42
    width = {}
    # Pipe model: each order thinner by sqrt of the number of its siblings.
    for x0, y0, x1, y1, order, tip in segs:
        width[order] = dbh * (0.6 ** order)

    svg.layer(layers["wood"])
    polys = []
    for x0, y0, x1, y1, order, tip in segs:
        w0 = width[order]
        if w0 < 0.1:
            continue
        ax, ay = P(x0, y0)
        bx, by = P(x1, y1)
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy) or 1
        nx, ny = -dy / n, dx / n
        w1 = w0 * 0.88
        polys.append(f"M{ax + nx * w0 / 2:.2f},{ay + ny * w0 / 2:.2f}L{bx + nx * w1 / 2:.2f},{by + ny * w1 / 2:.2f}"
                     f"L{bx - nx * w1 / 2:.2f},{by - ny * w1 / 2:.2f}L{ax - nx * w0 / 2:.2f},{ay - ny * w0 / 2:.2f}Z")
    if p.get("pale_bark"):
        svg.path("".join(polys), fill="#ffffff", stroke=ink, stroke_width=0.06)
        marks = []
        for x0, y0, x1, y1, order, tip in segs:
            w0 = width[order]
            if w0 < 0.4:
                continue
            ax, ay = P(x0, y0)
            bx, by = P(x1, y1)
            for t in rng.uniform(0, 1, int(math.hypot(bx - ax, by - ay) / 1.2) + 1):
                mx, my = ax + (bx - ax) * t, ay + (by - ay) * t
                ln = w0 * rng.uniform(0.3, 0.7)
                marks.append(f"M{mx - ln / 2:.2f},{my:.2f}h{ln:.2f}")
        svg.path("".join(marks), fill="none", stroke=ink, stroke_width=0.12, stroke_linecap="round")
    else:
        svg.path("".join(polys), fill=ink, stroke="none")
    svg.layer(layers["twigs"])
    twigs = []
    for x0, y0, x1, y1, order, tip in segs:
        if width[order] < 0.1:
            ax, ay = P(x0, y0)
            bx, by = P(x1, y1)
            twigs.append(f"M{ax:.2f},{ay:.2f}L{bx:.2f},{by:.2f}")
    svg.path("".join(twigs), fill="none", stroke=ink, stroke_width=0.06 * p["twig"], stroke_linecap="round",
             opacity=0.8)

    # Leaves: clusters at the twig ends, the species' size and density.
    n_dots, r_unit, dot = p["leaf"]
    pts = []
    for x0, y0, x1, y1, order, tip in segs:
        if not tip:
            continue
        cx, cy = P(x1, y1)
        r = r_unit * sy
        k = rng.poisson(n_dots)
        a = rng.uniform(0, 2 * math.pi, k)
        rr = r * np.sqrt(rng.uniform(0, 1, k))
        hang = p["droop"] * r * 1.5
        pts.append(np.stack([cx + np.cos(a) * rr, cy + np.sin(a) * rr * 0.8 + hang], 1))
    svg.layer(layers["leaves"])
    if pts:
        pts = np.concatenate(pts)
        svg.path("".join(f"M{x:.2f},{y:.2f}h0" for x, y in pts), fill="none", stroke=ink, stroke_width=dot,
                 stroke_linecap="round", opacity=0.8)
    lx = [P(s[2], s[3])[0] for s in segs]
    ty = min(P(s[2], s[3])[1] for s in segs)
    return min(lx), max(lx), ty
