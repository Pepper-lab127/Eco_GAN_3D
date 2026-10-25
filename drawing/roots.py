"""Root systems by species, shaped by where each tree stands.

Roots need air: below the water table the soil is saturated and roots
largely stop, so how deep a tree roots depends on how far its ground sits
above the water. The water table is taken as the river's median open-water
level near the shore, rising gently inland (an assumption, not measured).

  - Manitoba maple and American elm, on the wet lower bank: shallow, fibrous
    and wide (elm wider than the tree is tall), packed into a ROOT PLATE
    above the water table - the mat a floodplain tree lifts when it falls.
  - Bur oak, on well-drained ground: a deep taproot (4-5 m in old trees) with
    heart roots and very wide shallow laterals; no plate.
  - Trembling aspen: long shallow laterals, sinker roots about every metre
    down to about 3 m, and suckers rising from the laterals; neighbouring
    aspens share roots as one clone.

Sources: USDA plant guide (bur oak); NDSU tree handbook and UF/IFAS (American
elm); BC tree species compendium (trembling aspen).
"""

import math



def _wander(rng, x, z, ang, length, step=0.25, jitter=0.12, z_min=None, z_max=None, pull=0.0):
    """A root path from (x, z) heading `ang` (0 = straight down, +pi/2 = +x),
    in metres. pull: tendency back toward horizontal (laterals)."""
    pts = [(x, z)]
    n = max(2, int(length / step))
    for _ in range(n):
        ang += rng.normal(0, jitter)
        if pull:
            target = math.copysign(math.pi / 2 * 0.92, math.sin(ang) or 1)
            ang += (target - ang) * pull
        x, z = x + math.sin(ang) * step, z - math.cos(ang) * step
        if z_max is not None:
            z = min(z, z_max)
        if z_min is not None and z < z_min:
            z = z_min
            ang = math.copysign(math.pi / 2, math.sin(ang) or 1)
        pts.append((x, z))
    return pts


def system(sp, rng, x0, ground_at, height, crown_r, water_table_at, floor_at):
    """Root paths for one tree: list of (points [(x, z)], width_mm_at_base).
    ground_at / water_table_at / floor_at: functions of x (m) giving ground,
    water table and the deepest rootable level (e.g. the shale)."""
    g0 = ground_at(x0)
    out = []

    def limit(x):
        return max(water_table_at(x), floor_at(x))

    if sp in ("maple", "elm"):
        reach = crown_r * (2.2 if sp == "elm" else 1.4)
        plate_depth = min(1.1, max(0.35, g0 - water_table_at(x0) - 0.1))
        # Root plate: a dense mat of short roots around the base.
        for _ in range(int(26 if sp == "elm" else 20)):
            a = rng.uniform(0.9, 2.0) * rng.choice([-1, 1])
            ln = rng.uniform(0.8, 2.6)
            out.append((_wander(rng, x0, g0 - 0.05, a, ln, step=0.15, jitter=0.25, z_min=g0 - plate_depth,
                                z_max=g0 - 0.05), rng.uniform(0.05, 0.12)))
        # Main laterals, near the surface, far out; fine sinkers that stop at the water table.
        for side in (-1, 1):
            for _ in range(3 if sp == "elm" else 2):
                pts = _wander(rng, x0 + side * 0.2, g0 - 0.15, side * rng.uniform(1.25, 1.45), reach * rng.uniform(0.6, 1.0),
                              jitter=0.08, pull=0.15)
                pts = [(x, min(ground_at(x) - 0.1, max(z, ground_at(x) - plate_depth))) for x, z in pts]
                out.append((pts, rng.uniform(0.2, 0.32)))
                for k in range(3, len(pts), 6):
                    x, z = pts[k]
                    d = max(0.2, z - limit(x))
                    out.append((_wander(rng, x, z, rng.normal(0, 0.3), d, step=0.12, jitter=0.2,
                                        z_min=limit(x)), 0.04))
        if sp == "maple" and g0 - water_table_at(x0) < 1.5:
            # Flood-tolerant: adventitious roots from the lower trunk, into the surface soil.
            for side in (-1, 1):
                out.append((_wander(rng, x0, g0 + 0.3, side * 2.4, 0.9, step=0.1, jitter=0.1, z_max=g0 + 0.3), 0.06))
    elif sp == "oak":
        tap = min(4.6, max(0.5, g0 - floor_at(x0) - 0.2))
        out.append((_wander(rng, x0, g0, rng.normal(0, 0.05), tap, step=0.2, jitter=0.05), 0.55))
        for side in (-1, 1):
            # Heart roots, oblique.
            for _ in range(2):
                out.append((_wander(rng, x0, g0 - 0.2, side * rng.uniform(0.4, 0.8), rng.uniform(1.2, 2.2),
                                    jitter=0.1, z_min=floor_at(x0)), 0.25))
            # Wide shallow laterals.
            pts = _wander(rng, x0, g0 - 0.3, side * 1.35, crown_r * 2.0, jitter=0.07, pull=0.12)
            pts = [(x, min(ground_at(x) - 0.15, max(z, ground_at(x) - 0.7))) for x, z in pts]
            out.append((pts, 0.22))
    elif sp == "aspen":
        for side in (-1, 1):
            for _ in range(2):
                pts = _wander(rng, x0, g0 - 0.2, side * rng.uniform(1.35, 1.5), rng.uniform(6.0, 9.0), jitter=0.1,
                              pull=0.2)
                pts = [(x, min(ground_at(x) - 0.12, max(z, ground_at(x) - 0.5))) for x, z in pts]
                out.append((pts, 0.18))
                # Sinkers about every metre, to about 3 m.
                for k in range(4, len(pts), 4):
                    x, z = pts[k]
                    d = min(3.0, max(0.3, z - floor_at(x)))
                    out.append((_wander(rng, x, z, rng.normal(0, 0.12), d * rng.uniform(0.5, 1.0), step=0.15,
                                        jitter=0.08, z_min=floor_at(x)), 0.06))
    return out


def suckers(rng, roots, ground_at, every=2.5):
    """Aspen suckers: small shoots rising from shallow laterals. Returns
    [(x, ground, height)] in m."""
    shoots = []
    for pts, w in roots:
        if w < 0.15:
            continue
        for k in range(6, len(pts), int(every / 0.25)):
            x, z = pts[k]
            if ground_at(x) - z < 0.4 and rng.random() < 0.6:
                shoots.append((x, ground_at(x), rng.uniform(0.4, 1.6)))
    return shoots


def draw(svg, layer, sec, roots, ink="#111111"):
    """Roots as tapering strokes: each path split into a few pieces, thinner
    toward its tip."""
    svg.layer(layer)
    by_w = {}
    for pts, w in roots:
        n = len(pts)
        pieces = 3 if n > 6 else 1
        for p in range(pieces):
            a, b = p * n // pieces, min(n, (p + 1) * n // pieces + 1)
            seg = pts[a:b]
            if len(seg) < 2:
                continue
            wp = round(max(0.03, w * (1 - p / (pieces + 0.5))), 2)
            by_w.setdefault(wp, []).append("M" + "L".join(f"{sec.X(x):.2f},{sec.Y(z):.2f}" for x, z in seg))
    for wp, paths in sorted(by_w.items()):
        svg.path("".join(paths), fill="none", stroke=ink, stroke_width=wp, stroke_linecap="round",
                 stroke_linejoin="round", opacity=0.9)
