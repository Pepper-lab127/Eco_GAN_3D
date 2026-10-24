"""The ground layer of the bank: shrubs, wildflowers and grasses from the
Lockport plant list (data/lockport/ecology/lockport_flora.csv), drawn by
species and placed by the list's habitat and sun preferences.

Each metre of the section gets a moisture class from how often the ground is
under water (wet: reached by the open-water levels; moist: below the 1997
crest; dry: above it) and a light class from the lidar canopy overhead
(shade under the crowns, part shade within a few metres of their edge, sun
in the open). A plant can grow there if both match its preferences.
"""

import csv
import math

import numpy as np

# Moisture and light each species takes, read from the list's habitat and sun
# columns; height range in m; form drawn.
FLORA = {
    "Prunus americana": dict(moist={"moist", "dry"}, light={"sun", "part"}, h=(2.0, 4.0), form="plum", edge=True),
    "Symphoricarpos occidentalis": dict(moist={"moist", "dry"}, light={"sun", "part", "shade"}, h=(0.5, 1.0),
                                        form="snowberry"),
    "Amelanchier alnifolia": dict(moist={"dry"}, light={"sun", "part"}, h=(1.5, 3.5), form="saskatoon"),
    "Cornus sericea": dict(moist={"wet", "moist"}, light={"sun", "part"}, h=(1.2, 2.5), form="dogwood"),
    "Anemone canadensis": dict(moist={"wet", "moist"}, light={"part", "shade"}, h=(0.2, 0.5), form="anemone"),
    "Solidago canadensis": dict(moist={"moist", "dry"}, light={"sun"}, h=(0.6, 1.4), form="goldenrod"),
    "Asclepias incarnata": dict(moist={"wet"}, light={"sun"}, h=(0.8, 1.4), form="milkweed"),
    "Zizia aurea": dict(moist={"moist"}, light={"sun", "part"}, h=(0.3, 0.7), form="alexander"),
    "Trillium cernuum": dict(moist={"moist"}, light={"shade"}, h=(0.2, 0.45), form="trillium"),
    "Monarda fistulosa": dict(moist={"dry"}, light={"sun"}, h=(0.5, 1.0), form="bergamot"),
    "Rudbeckia hirta": dict(moist={"dry"}, light={"sun", "part"}, h=(0.3, 0.8), form="susan"),
    "Viola canadensis": dict(moist={"moist", "dry"}, light={"part", "shade"}, h=(0.15, 0.35), form="violet"),
    "Spartina pectinata": dict(moist={"wet"}, light={"sun"}, h=(1.2, 2.2), form="cordgrass"),
    "Elymus canadensis": dict(moist={"wet", "moist", "dry"}, light={"sun", "part"}, h=(0.7, 1.3), form="wildrye"),
}
SHRUBS = {"plum", "snowberry", "saskatoon", "dogwood"}


def load_list(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["Scientific Name"]: r for r in csv.DictReader(f)}


def classes(ground, canopy_h, levels, crest, edge_m=3):
    """Per-metre moisture and light classes along a section."""
    n = len(ground)
    wet = np.array([(levels >= g).mean() > 0 for g in ground])
    moist = ground <= crest
    moisture = np.where(wet, "wet", np.where(moist, "moist", "dry"))
    under = np.nan_to_num(canopy_h) > 3
    near = np.zeros(n, bool)
    for i in np.nonzero(under)[0]:
        near[max(0, i - edge_m):i + edge_m + 1] = True
    light = np.where(under, "shade", np.where(near, "part", "sun"))
    return moisture, light


def plant(svg, layers, form, x, y, h, rng, ink="#111111"):
    """Draw one plant of `form`, base at (x, y) mm, h mm tall."""
    lines, dots = [], []

    def stem(x0, y0, ang, ln, bend=0.0, k=4):
        pts = [(x0, y0)]
        for _ in range(k):
            ang += bend + rng.normal(0, 0.05)
            x0, y0 = x0 + math.sin(ang) * ln / k, y0 - math.cos(ang) * ln / k
            pts.append((x0, y0))
        lines.append("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in pts))
        return x0, y0

    def cloud(cx, cy, rx, ry, n, size=0.12):
        a = rng.uniform(0, 2 * math.pi, n)
        r = np.sqrt(rng.uniform(0, 1, n))
        dots.append((cx + np.cos(a) * r * rx, cy + np.sin(a) * r * ry, size))

    if form in SHRUBS:
        n_st = dict(plum=6, snowberry=8, saskatoon=5, dogwood=7)[form]
        spread = dict(plum=0.5, snowberry=0.9, saskatoon=0.3, dogwood=0.25)[form]
        tops = []
        for _ in range(n_st):
            ang = rng.normal(0, spread)
            ln = h * rng.uniform(0.7, 1.0)
            tops.append(stem(x, y, ang, ln, bend=0.0 if form == "dogwood" else -ang * 0.05))
        if form == "plum":
            cloud(x, y - h * 0.65, h * 0.55, h * 0.35, int(h * 25), 0.16)
        elif form == "snowberry":
            cloud(x, y - h * 0.5, h * 0.8, h * 0.45, int(h * 30), 0.1)
            cloud(x, y - h * 0.5, h * 0.7, h * 0.4, int(h * 3), 0.22)       # white berries, larger
        elif form == "saskatoon":
            for tx, ty in tops:
                cloud(tx, ty + h * 0.1, h * 0.15, h * 0.2, int(h * 5), 0.12)
        else:                                                               # dogwood: sparse leaves on straight stems
            for tx, ty in tops:
                cloud(tx, ty + h * 0.15, h * 0.08, h * 0.25, int(h * 3), 0.12)
    elif form in ("cordgrass", "wildrye"):
        n_bl = 9 if form == "cordgrass" else 6
        for _ in range(n_bl):
            ang = rng.normal(0, 0.35)
            tx, ty = stem(x, y, ang, h * rng.uniform(0.6, 1.0), bend=np.sign(ang) * 0.12)
        if form == "wildrye":
            for _ in range(3):
                tx, ty = stem(x, y, rng.normal(0, 0.15), h, bend=0.1)
                # nodding bristly spike
                for k in range(6):
                    sx, sy = tx + k * 0.12, ty + k * 0.12
                    lines.append(f"M{sx:.2f},{sy:.2f}l0.25,-0.18")
    else:
        # Wildflowers: stems with their heads.
        n_st = dict(goldenrod=3, milkweed=2, bergamot=2, susan=2, alexander=2, anemone=2, trillium=1,
                    violet=2)[form]
        for _ in range(n_st):
            ang = rng.normal(0, 0.15)
            tx, ty = stem(x, y, ang, h * rng.uniform(0.75, 1.0), bend=0.02)
            if form == "goldenrod":                                         # arching plume
                for k in range(8):
                    a = -0.6 + k * 0.17
                    dots.append((np.array([tx + math.sin(a) * h * 0.18]), np.array([ty + abs(a) * h * 0.08]), 0.1))
            elif form in ("milkweed", "alexander"):                         # umbel: flat-topped cluster
                for k in range(5):
                    a = -0.7 + k * 0.35
                    ex, ey = tx + math.sin(a) * h * 0.08, ty - math.cos(a) * h * 0.06
                    lines.append(f"M{tx:.2f},{ty:.2f}L{ex:.2f},{ey:.2f}")
                    dots.append((np.array([ex]), np.array([ey]), 0.12))
            elif form == "bergamot":                                        # round head
                cloud(tx, ty, h * 0.05, h * 0.05, 7, 0.1)
            elif form == "susan":                                           # disc with rays
                dots.append((np.array([tx]), np.array([ty]), 0.25))
                for k in range(6):
                    a = k * math.pi / 3
                    lines.append(f"M{tx:.2f},{ty:.2f}l{math.cos(a) * 0.35:.2f},{math.sin(a) * 0.35:.2f}")
            elif form == "trillium":                                        # three leaves, nodding flower
                for side in (-1, 0, 1):
                    lines.append(f"M{tx:.2f},{ty:.2f}l{side * h * 0.25:.2f},{h * 0.05:.2f}")
                dots.append((np.array([tx + 0.15]), np.array([ty + 0.35]), 0.18))
            elif form == "anemone":
                dots.append((np.array([tx]), np.array([ty]), 0.22))
                lines.append(f"M{x:.2f},{y - h * 0.4:.2f}l{h * 0.3:.2f},{-h * 0.1:.2f}M{x:.2f},{y - h * 0.4:.2f}"
                             f"l{-h * 0.3:.2f},{-h * 0.1:.2f}")
            elif form == "violet":
                dots.append((np.array([tx]), np.array([ty]), 0.16))
    svg.layer(layers["stems"])
    svg.path("".join(lines), fill="none", stroke=ink, stroke_width=0.05, stroke_linecap="round", opacity=0.9)
    svg.layer(layers["leaves"])
    d = []
    widths = set()
    for xs, ys, size in dots:
        widths.add(size)
    for size in sorted(widths):
        d = "".join(f"M{a:.2f},{b:.2f}h0" for xs, ys, sz in dots if sz == size for a, b in zip(xs, ys))
        svg.path(d, fill="none", stroke=ink, stroke_width=size, stroke_linecap="round", opacity=0.85)


def plant_bank(svg, layers, sec, ch, ground, moisture, light, rng, spacing=0.6, shrub_every=4.0, ink="#111111",
               lo=0, hi=None):
    """Place and draw the ground layer along the section. Returns
    {latin: [chainages]} of what was placed."""
    placed = {}
    hi = len(ch) if hi is None else hi
    x = float(lo) + rng.uniform(0, spacing)
    next_shrub = float(lo) + rng.uniform(0, shrub_every)
    while x < hi - 0.5:
        i = int(x)
        m, l = moisture[i], light[i]
        ok = [k for k, v in FLORA.items() if m in v["moist"] and l in v["light"]]
        shrubs = [k for k in ok if FLORA[k]["form"] in SHRUBS]
        herbs = [k for k in ok if FLORA[k]["form"] not in SHRUBS]
        pick = None
        if shrubs and x >= next_shrub:
            pick = shrubs[rng.integers(len(shrubs))]
            next_shrub = x + shrub_every * rng.uniform(0.6, 1.6)
        elif herbs:
            pick = herbs[rng.integers(len(herbs))]
        if pick:
            v = FLORA[pick]
            h_m = rng.uniform(*v["h"])
            gz = float(np.interp(x, ch, ground))
            plant(svg, layers, v["form"], sec.X(x), sec.Y(gz), h_m * sec.sv, rng, ink=ink)
            placed.setdefault(pick, []).append(x)
        x += spacing * rng.uniform(0.6, 1.4)
    return placed
