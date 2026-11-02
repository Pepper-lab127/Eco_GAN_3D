"""Sheet 09a - How red-osier dogwood (Cornus sericea) spreads.

One shrub on the wet lower bank at the riverbank section's line (04b), grown
into a thicket over fifteen years, and the seed its fruit sends away:

  - plan, 1:50: the thicket's edge each year, every stem base (ramet) where
    it rooted, and how it got there: a STOLON (horizontal stem running just
    under the ground and rooting, the habit behind its old name stolonifera)
    or LAYERING (a low stem bending to the ground and rooting where it
    touches, at a node)
  - section, 1:20: the thicket cut through, showing the arching and layered
    stems, rooting nodes, stolons with new shoots, and the shallow fibrous
    roots above an assumed water table
  - plan, 1:2,500, on the lidar: birds eat the white fruit (mid-July to
    mid-October) and drop the seed where they perch next. Perches are the
    lidar canopy's tree tops; flights are weighted toward nearer trees

The growth is a diagram of the species' habit (layering, stolons, suckering,
per the UW propagation notes, Nature Manitoba and the Alberta species
profile), not a survey: rates and the flight-distance weighting are
ESTIMATES. Tree tops and the ground are lidar.

    python -m sheets.dogwood_spread
"""

import argparse
import math
import os

import contourpy
import numpy as np
from scipy import ndimage

from drawing import terrain
from drawing.svg import PAPER, Svg, points_to_path, scale_bar

INK = "#111111"


# --------------------------------------------------------------------------- growth
def stem_len(age):
    """Typical stem length (m) of a ramet `age` years old: to about 2.6 m."""
    return min(2.6, 0.5 + 0.32 * age)


def grow(years, rng, p_stolon=0.42, p_layer=0.3):
    """Ramets as dicts: x, y (m), born (year), parent index, how ('seed',
    'stolon', 'layer'). Crowding stops new ramets within 0.35 m of another,
    or where more than five stand within 1 m."""
    R = [dict(x=0.0, y=0.0, born=0, parent=None, how="seed")]
    for year in range(1, years + 1):
        cx = np.mean([r["x"] for r in R])
        cy = np.mean([r["y"] for r in R])
        for i, r in enumerate(list(R)):
            age = year - r["born"]
            if age < 3:
                continue
            for how, p, dist in (("stolon", p_stolon, rng.uniform(0.45, 1.1)),
                                 ("layer", p_layer, stem_len(age) * rng.uniform(0.6, 0.92))):
                if rng.random() > p:
                    continue
                out = math.atan2(r["y"] - cy, r["x"] - cx) if (r["x"], r["y"]) != (cx, cy) else rng.uniform(0, 6.28)
                a = out + rng.normal(0, 0.9)
                nx, ny = r["x"] + dist * math.cos(a), r["y"] + dist * math.sin(a)
                d = [math.hypot(nx - q["x"], ny - q["y"]) for q in R]
                if min(d) < 0.35 or sum(v < 1.0 for v in d) > 5:
                    continue
                R.append(dict(x=nx, y=ny, born=year, parent=i, how=how))
    return R


def edges_by_year(R, years, cell=0.02):
    """Closed outlines (m) of the thicket's crown edge at the end of each year."""
    xs = np.array([r["x"] for r in R])
    ys = np.array([r["y"] for r in R])
    pad = 3.0
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    gx = np.arange(x0, x1, cell)
    gy = np.arange(y0, y1, cell)
    X, Y = np.meshgrid(gx, gy)
    out = {}
    for year in range(1, years + 1):
        m = np.zeros(X.shape)
        for r in R:
            age = year - r["born"]
            if age < 0:
                continue
            rc = 0.25 + 0.42 * stem_len(age)
            m = np.maximum(m, (np.hypot(X - r["x"], Y - r["y"]) < rc).astype(float))
        m = ndimage.gaussian_filter(m, 4)
        lines = contourpy.contour_generator(gx, gy, m).lines(0.5)
        out[year] = [ln for ln in lines if len(ln) > 8]
    return out


# --------------------------------------------------------------------------- drawing helpers
def shrub_section(svg, L, x, y, h, rng, stems=7, lean=0.0, fruit=True):
    """Red-osier dogwood in elevation: straight-ish reddish stems from a crown,
    opposite side twigs, leaves near the tips, flat clusters of white fruit."""
    wood, twigs, leaves, fruit_d = [], [], [], []
    for k in range(stems):
        a = lean + rng.normal(0, 0.16)
        ln = h * rng.uniform(0.65, 1.0)
        n = 7
        px, py = x, y
        pts = [(px, py)]
        for j in range(n):
            a += a * 0.03 + rng.normal(0, 0.025)           # arch outward a little
            px += math.sin(a) * ln / n
            py -= math.cos(a) * ln / n
            pts.append((px, py))
            if j >= 2:                                     # opposite twigs at nodes
                for side in (-1, 1):
                    ta = a + side * rng.uniform(0.5, 0.9)
                    tl = ln * rng.uniform(0.07, 0.14)
                    tx, ty = px + math.sin(ta) * tl, py - math.cos(ta) * tl
                    twigs.append(f"M{px:.2f},{py:.2f}L{tx:.2f},{ty:.2f}")
                    if j >= n - 3:
                        leaves.append((tx, ty))
        wood.append("M" + "L".join(f"{a_:.2f},{b:.2f}" for a_, b in pts))
        leaves.append(pts[-1])
        if fruit and rng.random() < 0.6:
            fruit_d.append(pts[-1])
    svg.layer(L["stems"])
    svg.path("".join(wood), fill="none", stroke=INK, stroke_width=0.22, stroke_linecap="round",
             stroke_linejoin="round")
    svg.layer(L["twigs"])
    svg.path("".join(twigs), fill="none", stroke=INK, stroke_width=0.08, stroke_linecap="round")
    svg.layer(L["leaves"])
    d = []
    for lx, ly in leaves:
        for _ in range(6):
            d.append(f"M{lx + rng.normal(0, h * 0.04):.2f},{ly + rng.normal(0, h * 0.03):.2f}h0")
    svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.35, stroke_linecap="round", opacity=0.8)
    svg.layer(L["fruit"])
    for fx, fy in fruit_d:                                 # flat-topped cluster of white drupes
        for _ in range(7):
            svg.circle(fx + rng.normal(0, h * 0.025), fy - h * 0.02 + rng.normal(0, h * 0.008), 0.32,
                       fill="#ffffff", stroke=INK, stroke_width=0.06)


def roots(svg, L, x, y, depth, spread, rng, n=16, w=0.07):
    """Fibrous roots (mm), shallow and spreading."""
    d = []
    for _ in range(n):
        a = rng.uniform(-1.2, 1.2)
        px, py = x, y
        ln = rng.uniform(0.4, 1.0) * depth
        pts = [(px, py)]
        for _ in range(6):
            a += rng.normal(0, 0.25)
            px += math.sin(a) * ln / 6 * spread / depth
            py += math.cos(a) * ln / 6
            pts.append((px, py))
        d.append("M" + "L".join(f"{a_:.2f},{b:.2f}" for a_, b in pts))
    svg.layer(L["roots"])
    svg.path("".join(d), fill="none", stroke=INK, stroke_width=w, stroke_linecap="round", opacity=0.85)


# --------------------------------------------------------------------------- sheet
def build(args):
    rng = np.random.default_rng(args.seed)
    Wp, Hp = PAPER[args.paper]
    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(rings="THICKET EDGE BY YEAR · 0.05 mm, EVERY 5TH 0.15 mm", ramets="STEM BASES (RAMETS) · FILLED",
             stolon="STOLONS · DASHED 0.12 mm", layer="LAYERED STEMS · 0.12 mm",
             plan_stems="PLAN · STEMS, YEAR 15 · 0.05 mm",
             ground="SECTION · GROUND, WATER TABLE (ASSUMED, DASHED) · 0.1–0.25 mm",
             stems="SECTION · STEMS · 0.22 mm", twigs="SECTION · TWIGS · 0.08 mm",
             leaves="SECTION · LEAVES, STIPPLE", fruit="SECTION · WHITE FRUIT · 0.06 mm",
             roots="SECTION · ROOTS AND STOLONS · 0.07–0.15 mm", nodes="SECTION · ROOTING NODES",
             lidar="BIRDS · LIDAR CONTOURS, RIVER, CANOPY EDGE · 0.04–0.12 mm",
             perches="BIRDS · PERCHES (LIDAR TREE TOPS) · 0.08 mm", flights="BIRDS · FLIGHTS · 0.05 mm",
             seeds="BIRDS · SEED DROPPED", rings_b="BIRDS · DISTANCE RINGS, DOTTED",
             scale="SCALE BARS", text="TEXT")
    for v in L.values():
        svg.layer(v)

    years = args.years
    R = grow(years, rng)
    edges = edges_by_year(R, years)
    print(f"{len(R)} ramets after {years} years: "
          f"{sum(r['how'] == 'stolon' for r in R)} by stolon, {sum(r['how'] == 'layer' for r in R)} by layering")

    # ------------------------------------------------------------ plan 1:50
    s = 1000.0 / args.plan_scale
    allpts = np.vstack([np.vstack(v) for v in edges[years]])
    px0 = args.plan_x - (allpts[:, 0].min() + allpts[:, 0].max()) / 2 * s
    py0 = args.plan_y + (allpts[:, 1].min() + allpts[:, 1].max()) / 2 * s

    def PX(x, y):
        return px0 + x * s, py0 - y * s

    for year in range(1, years + 1):
        heavy = year % 5 == 0
        svg.layer(L["rings"])
        for ln in edges[year]:
            X, Y = PX(ln[:, 0], ln[:, 1])
            svg.path(points_to_path(X, Y) + "Z", fill="none", stroke=INK,
                     stroke_width=0.15 if heavy else 0.05, opacity=1.0 if heavy else 0.7)
        if heavy or year == 1:
            # Year number at the edge's northernmost point.
            ln = max(edges[year], key=len)
            k = int(np.argmax(ln[:, 1]))
            X, Y = PX(ln[k, 0], ln[k, 1])
            svg.layer(L["text"])
            svg.text(float(X), float(Y) - 1.2, f"{year}", size=1.8, fill=INK, anchor="middle")
    # How each ramet came to be.
    for r in R:
        if r["parent"] is None:
            continue
        q = R[r["parent"]]
        (ax, ay), (bx, by) = PX(q["x"], q["y"]), PX(r["x"], r["y"])
        if r["how"] == "stolon":
            svg.layer(L["stolon"])
            mx, my = (ax + bx) / 2 + rng.normal(0, 0.6), (ay + by) / 2 + rng.normal(0, 0.6)
            svg.path(f"M{ax:.2f},{ay:.2f}Q{mx:.2f},{my:.2f} {bx:.2f},{by:.2f}", fill="none", stroke=INK,
                     stroke_width=0.12, stroke_dasharray="0.8 0.6")
        else:
            svg.layer(L["layer"])
            svg.path(f"M{ax:.2f},{ay:.2f}L{bx:.2f},{by:.2f}", fill="none", stroke=INK, stroke_width=0.12)
            # The node where it rooted: a short bar across the stem.
            a = math.atan2(by - ay, bx - ax) + math.pi / 2
            svg.line(bx - math.cos(a) * 0.9, by - math.sin(a) * 0.9, bx + math.cos(a) * 0.9, by + math.sin(a) * 0.9,
                     stroke=INK, stroke_width=0.12)
    # Year-15 stems in plan, radiating from each ramet.
    d = []
    for r in R:
        age = years - r["born"]
        ln = stem_len(age) * 0.45 * s
        for _ in range(min(3 + 2 * age, 18)):
            a = rng.uniform(0, 2 * math.pi)
            l_ = ln * rng.uniform(0.4, 1.0)
            x, y = PX(r["x"], r["y"])
            d.append(f"M{x:.2f},{y:.2f}l{math.cos(a) * l_:.2f},{math.sin(a) * l_:.2f}")
    svg.layer(L["plan_stems"])
    svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.05, opacity=0.45)
    svg.layer(L["ramets"])
    for r in R:
        x, y = PX(r["x"], r["y"])
        rad = 0.35 + 0.06 * (years - r["born"])
        svg.circle(x, y, rad, fill=INK if r["how"] != "seed" else "#ffffff", stroke=INK, stroke_width=0.15)
    svg.layer(L["scale"])
    _, ytop = PX(0, allpts[:, 1].max())
    _, ybot = PX(0, allpts[:, 1].min())
    scale_bar(svg, args.plan_x - 2 * s, ybot + 8, 4 * s, 4, "m", color=INK)
    svg.layer(L["text"])
    svg.text(args.plan_x, ytop - 8, f"THE THICKET FROM ONE SHRUB, YEARS 1–{years} · 1:{args.plan_scale:,}",
             size=2.0, fill=INK, anchor="middle", spacing=0.2)

    # ------------------------------------------------------------ section 1:20
    ss = 1000.0 / args.section_scale
    # Cut along the line through the first shrub that crosses the most parent-child links.
    best = None
    for ang in np.linspace(0, math.pi, 36, endpoint=False):
        c, si = math.cos(ang), math.sin(ang)
        near = [i for i, r in enumerate(R) if abs(-r["x"] * si + r["y"] * c) < args.band]
        links = sum(1 for i in near if R[i]["parent"] in near)
        if best is None or links > best[0]:
            best = (links, ang, near)
    _, ang, near = best
    c, si = math.cos(ang), math.sin(ang)
    along = {i: R[i]["x"] * c + R[i]["y"] * si for i in near}
    lo, hi = min(along.values()) - 1.5, max(along.values()) + 1.5
    gx0 = args.section_x - (lo + hi) / 2 * ss
    gy = args.section_y

    def SX(u):
        return gx0 + u * ss

    # Ground: nearly level, a few centimetres of relief; assumed water table 0.6 m down.
    us = np.linspace(lo - 0.5, hi + 0.5, 200)
    gz = 0.04 * np.sin(us * 1.3) + 0.02 * np.sin(us * 3.1)

    def GY(u):
        return gy - float(np.interp(u, us, gz)) * ss

    svg.layer(L["ground"])
    svg.path(points_to_path(SX(us), gy - gz * ss), fill="none", stroke=INK, stroke_width=0.25)
    svg.path(points_to_path(SX(us), gy + 0.6 * ss + 0 * us), fill="none", stroke=INK, stroke_width=0.1,
             stroke_dasharray="1.2 0.8", opacity=0.7)
    # Soil stipple, thinning with depth.
    d = []
    for _ in range(int((hi - lo) * 260)):
        u = rng.uniform(lo - 0.5, hi + 0.5)
        z = rng.uniform(0, 0.9) ** 1.6
        d.append(f"M{SX(u):.2f},{GY(u) + z * ss:.2f}h0")
    svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.1, stroke_linecap="round", opacity=0.5)

    order = sorted(near, key=lambda i: R[i]["born"])
    for i in order:
        r = R[i]
        u = along[i]
        age = years - r["born"]
        h = stem_len(age) * ss * 1.05
        x, y = SX(u), GY(u)
        shrub_section(svg, L, x, y, h, rng, stems=min(3 + age, 14), lean=rng.normal(0, 0.06), fruit=age >= 3)
        roots(svg, L, x, y, depth=min(0.55, 0.15 + 0.04 * age) * ss, spread=min(0.9, 0.2 + 0.06 * age) * ss, rng=rng)
        p = r["parent"]
        if p is None or p not in along:
            continue
        pu = along[p]
        px_, py_ = SX(pu), GY(pu)
        if r["how"] == "layer":
            # A stem from the parent arching up and over, down to the ground, rooted at the node.
            dx = x - px_
            H = min(0.55 * abs(dx), stem_len(years - R[p]["born"]) * ss * 0.6)
            svg.layer(L["stems"])
            svg.path(f"M{px_:.2f},{py_:.2f}C{px_ + dx * 0.15:.2f},{py_ - H * 1.35:.2f} {x - dx * 0.2:.2f},{y - H * 1.2:.2f} "
                     f"{x:.2f},{y:.2f}", fill="none", stroke=INK, stroke_width=0.22, stroke_linecap="round")
            svg.layer(L["nodes"])
            svg.circle(x, y, 0.6, fill="#ffffff", stroke=INK, stroke_width=0.15)
            roots(svg, L, x, y, depth=0.25 * ss, spread=0.3 * ss, rng=rng, n=10, w=0.06)
        else:
            # A stolon just under the ground, rooting at nodes along the way.
            k = 24
            uu = np.linspace(pu, u, k)
            depth = 0.08 + 0.04 * np.sin(np.linspace(0, math.pi, k))
            X = SX(uu)
            Y = np.array([GY(v) for v in uu]) + depth * ss
            svg.layer(L["roots"])
            svg.path(points_to_path(X, Y), fill="none", stroke=INK, stroke_width=0.15, stroke_dasharray="1.0 0.5")
            for j in range(6, k - 3, 8):
                roots(svg, L, X[j], Y[j], depth=0.12 * ss, spread=0.1 * ss, rng=rng, n=4, w=0.05)
            svg.layer(L["nodes"])
            svg.circle(x, y + 0.08 * ss, 0.45, fill=INK, stroke="none")
    svg.layer(L["scale"])
    scale_bar(svg, SX(lo), gy + 1.1 * ss, 1 * ss, 1, "m", color=INK)
    svg.layer(L["text"])
    svg.text(args.section_x, gy + 1.45 * ss, f"SECTION THROUGH THE THICKET, YEAR {years} · 1:{args.section_scale}",
             size=2.0, fill=INK, anchor="middle", spacing=0.2)

    # ------------------------------------------------------------ birds, 1:2,500 on the lidar
    dtm, _, _ = terrain.load_dem(args.dem)
    dsm, _, _ = terrain.load_dem(args.dsm)
    r0, c0 = args.shrub_rc
    hw = args.bird_half
    g = dtm[r0 - hw:r0 + hw, c0 - hw:c0 + hw].astype(float)
    top = dsm[r0 - hw:r0 + hw, c0 - hw:c0 + hw].astype(float)
    chm = np.clip(top - g, 0, None)
    sb = 1000.0 / args.bird_scale
    bx0, by0 = args.bird_x - hw * sb, args.bird_y - hw * sb

    def BX(col):
        return bx0 + (np.asarray(col, float) + 0.5) * sb

    def BY(row):
        return by0 + (np.asarray(row, float) + 0.5) * sb

    clip = svg.clip_rect(bx0, by0, 2 * hw * sb, 2 * hw * sb)
    svg.layer(L["lidar"])
    svg.open_group(clip_path=clip)
    gen = contourpy.contour_generator(z=ndimage.gaussian_filter(g, 1.5))
    for z in np.arange(np.floor(np.nanmin(g)), np.nanmax(g), 1.0):
        for ln in gen.lines(z):
            if len(ln) > 6:
                svg.path(points_to_path(BX(ln[:, 0]), BY(ln[:, 1])), fill="none", stroke=INK, stroke_width=0.04,
                         opacity=0.4)
    for ln in contourpy.contour_generator(z=ndimage.gaussian_filter(g, 1.0)).lines(args.water):
        if len(ln) > 20:
            svg.path(points_to_path(BX(ln[:, 0]), BY(ln[:, 1])), fill="none", stroke=INK, stroke_width=0.12)
    for ln in contourpy.contour_generator(z=ndimage.gaussian_filter(chm, 1.5)).lines(3.0):
        if len(ln) > 12:
            svg.path(points_to_path(BX(ln[:, 0]), BY(ln[:, 1])), fill="none", stroke=INK, stroke_width=0.05,
                     stroke_dasharray="0.3 0.4", opacity=0.8)
    svg.close_group()
    # Perches: tree tops in the lidar canopy, at least 6 m tall, one per 9 m.
    peaks = (chm == ndimage.maximum_filter(chm, size=9)) & (chm > 6.0) & (g > args.water)
    pr, pc = np.nonzero(peaks)
    dist = np.hypot(pr - hw, pc - hw)
    keep = (dist > 4) & (dist < hw - 2)
    pr, pc, dist = pr[keep], pc[keep], dist[keep]
    svg.layer(L["perches"])
    for rr, cc in zip(pr, pc):
        svg.circle(float(BX(cc)), float(BY(rr)), 0.5, fill="none", stroke=INK, stroke_width=0.08, opacity=0.8)
    # Seed-carrying visits: each picks a perch, more likely the nearer it is.
    w = np.exp(-dist / args.bird_mean)
    w /= w.sum()
    visits = rng.choice(len(pr), size=args.visits, p=w)
    sx_, sy_ = float(BX(hw)), float(BY(hw))
    svg.layer(L["flights"])
    d = []
    for v in visits:
        ex, ey = float(BX(pc[v])), float(BY(pr[v]))
        mx, my = (sx_ + ex) / 2, (sy_ + ey) / 2
        nx, ny = -(ey - sy_), ex - sx_
        k = rng.normal(0, 0.18)
        d.append(f"M{sx_:.2f},{sy_:.2f}Q{mx + nx * k:.2f},{my + ny * k:.2f} {ex:.2f},{ey:.2f}")
    svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.05, opacity=0.45)
    svg.layer(L["seeds"])
    d = []
    for v in visits:
        for _ in range(args.seeds_per_visit):
            a, rr_ = rng.uniform(0, 6.28), abs(rng.normal(0, 2.5))
            d.append(f"M{float(BX(pc[v])) + math.cos(a) * rr_ * sb:.2f},{float(BY(pr[v])) + math.sin(a) * rr_ * sb:.2f}h0")
    svg.path("".join(d), fill="none", stroke=INK, stroke_width=0.28, stroke_linecap="round")
    svg.circle(sx_, sy_, 1.2, fill="#ffffff", stroke=INK, stroke_width=0.25)
    svg.circle(sx_, sy_, 0.5, fill=INK, stroke="none")
    svg.layer(L["rings_b"])
    for rm in (50, 100, 200):
        svg.circle(sx_, sy_, rm * sb, fill="none", stroke=INK, stroke_width=0.06, stroke_dasharray="0.2 0.8")
        svg.layer(L["text"])
        svg.text(sx_ + rm * sb * 0.707 + 0.8, sy_ - rm * sb * 0.707 - 0.8, f"{rm} m", size=1.6, fill=INK)
        svg.layer(L["rings_b"])
    svg.rect(bx0, by0, 2 * hw * sb, 2 * hw * sb, fill="none", stroke=INK, stroke_width=0.08, opacity=0.6)
    svg.layer(L["scale"])
    scale_bar(svg, bx0, by0 + 2 * hw * sb + 5, 100 * sb, 100, "m", color=INK)
    svg.layer(L["text"])
    svg.text(args.bird_x, by0 - 6, f"WHERE BIRDS CARRY THE SEED · 1:{args.bird_scale:,}", size=2.0, fill=INK,
             anchor="middle", spacing=0.2)
    print(f"{len(pr)} perches; {args.visits} visits; median flight {np.median(dist[visits]):.0f} m")

    annot(20, 20, "HOW RED-OSIER DOGWOOD SPREADS", size=5.2, fill=INK, spacing=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default="data/lockport/dem/lockport_dtm_1m.tif")
    ap.add_argument("--dsm", default="data/lockport/dem/lockport_dsm_1m.tif")
    ap.add_argument("--shrub-rc", type=int, nargs=2, default=[1840, 1562], help="DEM row, col of the shrub (04b line)")
    ap.add_argument("--water", type=float, default=218.4, help="lidar ground below this is river")
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--plan-scale", type=int, default=50)
    ap.add_argument("--plan-x", type=float, default=270.0)
    ap.add_argument("--plan-y", type=float, default=180.0)
    ap.add_argument("--section-scale", type=int, default=20)
    ap.add_argument("--section-x", type=float, default=457.2)
    ap.add_argument("--section-y", type=float, default=470.0, help="mm: ground line of the section")
    ap.add_argument("--band", type=float, default=0.7, help="m either side of the cut a ramet is shown")
    ap.add_argument("--bird-scale", type=int, default=2500)
    ap.add_argument("--bird-half", type=int, default=260, help="m, half the bird plan's width")
    ap.add_argument("--bird-x", type=float, default=640.0)
    ap.add_argument("--bird-y", type=float, default=180.0)
    ap.add_argument("--bird-mean", type=float, default=80.0, help="m, falloff of perch choice with distance (estimate)")
    ap.add_argument("--visits", type=int, default=160)
    ap.add_argument("--seeds-per-visit", type=int, default=2)
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="outputs/sheets/09a_dogwood_spread.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
