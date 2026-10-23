"""Subsurface layers from well logs, interpolated along a section and drawn
with organic textures (varved clay, clast-filled till, fissile shale,
bedded and jointed carbonate with vugs)."""

import numpy as np

from drawing.svg import points_to_path


# Order matters: top to bottom.
LAYERS = ["clay", "till", "shale", "carbonate"]
NAMES = {
    "clay": "LAKE AGASSIZ CLAY (MUD / FINES)",
    "till": "TILL (DIAMICTON)",
    "shale": "SHALE",
    "carbonate": "CARBONATE BEDROCK · LIMESTONE AQUIFER",
}

# Logs from the Groundwater Information Network (depth from surface, m, to the
# TOP of each layer). A layer absent from a well has zero thickness there.
LOCKPORT_WELLS = {
    "west": dict(col=1150, tops={"clay": 0.0, "till": 3.96, "shale": 10.06, "carbonate": 10.06},
                 bottom=30.48),
    "east": dict(col=1850, tops={"clay": 0.0, "till": 1.52, "shale": 11.28, "carbonate": 17.98},
                 bottom=63.09),
}


def interfaces(cols, ground_at, wells=LOCKPORT_WELLS):
    """Elevation of the top of each layer at each DEM column, interpolated
    linearly between wells (held flat beyond them). Each well's depths hang
    from the lidar ground at its column (well collar elevations not given)."""
    ws = sorted(wells.values(), key=lambda w: w["col"])
    xs = [w["col"] for w in ws]
    out = {}
    for layer in LAYERS:
        zs = [ground_at(w["col"]) - w["tops"][layer] for w in ws]
        out[layer] = np.interp(cols, xs, zs)
    return out


def _wave(rng, n_terms=4, amp=0.35, wl=(6, 40)):
    """A smooth random 1-D wave y(x) in mm (a sum of sines)."""
    ks = 2 * np.pi / rng.uniform(*wl, n_terms)
    ph = rng.uniform(0, 2 * np.pi, n_terms)
    am = amp * rng.uniform(0.3, 1.0, n_terms) / np.sqrt(n_terms)
    return lambda x: sum(a * np.sin(k * x + p) for a, k, p in zip(am, ks, ph))


def _blob(rng, cx, cy, r, n=9, rough=0.35):
    """An irregular rounded clast outline."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + rough * rng.uniform(-1, 1, n))
    xs, ys = cx + rr * np.cos(t), cy + rr * np.sin(t) * rng.uniform(0.55, 0.9)
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys)) + " Z"


def _texture(svg, kind, x0, x1, y0, y1, ink, rng, w):
    """Organic pattern over a box; the caller clips it to the layer.

    clay: varves, fine wavering laminae that pinch and part;
    till: unsorted clasts of every size in a dusted matrix;
    shale: fissile lenses, short overlapping wavy partings;
    carbonate: wavy bedding planes, irregular joints and small vugs.
    """
    xs = np.arange(x0 - 2, x1 + 2, 0.6)
    if kind == "clay":
        y = y0
        while y < y1:
            y += rng.uniform(0.35, 0.9)
            f = _wave(rng, amp=0.25)
            yy = y + f(xs)
            # Break each lamina into runs so the varves pinch and part.
            keep = _wave(rng, amp=1.0, wl=(8, 30))(xs) > rng.uniform(-0.6, 0.2)
            run, parts = [], []
            for x, v, k in zip(xs, yy, keep):
                if k:
                    run.append(f"{x:.2f},{v:.2f}")
                elif len(run) > 1:
                    parts.append("M" + " L".join(run))
                    run = []
                else:
                    run = []
            if len(run) > 1:
                parts.append("M" + " L".join(run))
            svg.path(" ".join(parts), fill="none", stroke=ink, stroke_width=w * rng.uniform(0.6, 1.2),
                     opacity=float(rng.uniform(0.45, 0.85)))
    elif kind == "till":
        area = (x1 - x0) * (y1 - y0)
        n = int(area / 9)
        sizes = np.clip(rng.lognormal(-1.0, 0.7, n), 0.12, 2.6)
        d = " ".join(_blob(rng, rng.uniform(x0, x1), rng.uniform(y0, y1), r) for r in sizes)
        svg.path(d, fill="none", stroke=ink, stroke_width=w, opacity=0.75)
        m = int(area * 1.2)
        d = "".join(f"M{x:.2f},{y:.2f}h0" for x, y in zip(rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)))
        svg.path(d, fill="none", stroke=ink, stroke_width=w * 1.6, stroke_linecap="round", opacity=0.45)
    elif kind == "shale":
        n = int((x1 - x0) * (y1 - y0) / 1.6)
        parts = []
        for _ in range(n):
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            half = rng.uniform(1.0, 4.5)
            sag = rng.uniform(-0.12, 0.12)
            parts.append(f"M{cx - half:.2f},{cy:.2f} Q{cx:.2f},{cy + sag:.2f} {cx + half:.2f},{cy:.2f}")
        svg.path(" ".join(parts), fill="none", stroke=ink, stroke_width=w, opacity=0.7)
    elif kind == "carbonate":
        # Bedding planes that wander, thin out and break off.
        beds, y, parts = [], y0, []
        while y < y1:
            y += rng.uniform(0.8, 3.2)
            b = y + _wave(rng, amp=0.45, wl=(12, 70))(xs)
            beds.append(b)
            keep = _wave(rng, amp=1.0, wl=(10, 45))(xs) > rng.uniform(-0.7, 0.3)
            run = []
            for x, v, k in zip(xs, b, keep):
                if k:
                    run.append(f"{x:.2f},{v:.2f}")
                else:
                    if len(run) > 1:
                        parts.append("M" + " L".join(run))
                    run = []
            if len(run) > 1:
                parts.append("M" + " L".join(run))
        svg.path(" ".join(parts), fill="none", stroke=ink, stroke_width=w, opacity=0.7)
        # Joints: sparse, leaning, ragged, only in some beds.
        parts = []
        for top, bot in zip(beds[:-1], beds[1:]):
            if rng.random() < 0.45:
                continue
            x = x0 + rng.uniform(0, 10)
            while x < x1:
                i = min(max(int((x - xs[0]) / 0.6), 0), len(xs) - 1)
                mid = (top[i] + bot[i]) / 2 + rng.uniform(-0.2, 0.2)
                parts.append(f"M{x:.2f},{top[i]:.2f}L{x + rng.uniform(-0.3, 0.3):.2f},{mid:.2f}"
                             f"L{x + rng.uniform(-0.5, 0.5):.2f},{bot[i]:.2f}")
                x += rng.uniform(3.0, 16.0)
        svg.path(" ".join(parts), fill="none", stroke=ink, stroke_width=w * 0.8, opacity=0.55)
        # Vugs and fossil fragments: the porous, dissolved rock of the aquifer.
        n = int((x1 - x0) * (y1 - y0) / 7)
        d = " ".join(_blob(rng, rng.uniform(x0, x1), rng.uniform(y0, y1), r, n=7, rough=0.5)
                     for r in np.clip(rng.lognormal(-1.8, 0.6, n), 0.06, 0.7))
        svg.path(d, fill="none", stroke=ink, stroke_width=w * 0.8, opacity=0.55)
        m = int((x1 - x0) * (y1 - y0) * 0.6)
        d = "".join(f"M{x:.2f},{y:.2f}h0" for x, y in zip(rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)))
        svg.path(d, fill="none", stroke=ink, stroke_width=w * 1.4, stroke_linecap="round", opacity=0.35)


def draw_layers(svg, sec, ch, cols, ground, ground_at, z_bottom, ink, seed=7, weight=0.08, top_cap=None,
                layer_names=None, label=True, wells=LOCKPORT_WELLS):
    """Draw each layer, patterned, between its top (capped by the ground and
    by top_cap, e.g. to leave an unsurveyed riverbed blank) and the next
    layer's top. ground_at(col): lidar ground at a DEM column, for hanging the
    well logs. layer_names: optional {kind: svg layer name}."""
    rng = np.random.default_rng(seed)
    tops = interfaces(cols, ground_at, wells)
    cap = ground if top_cap is None else np.minimum(ground, top_cap)
    X = sec.X(ch)
    for i, kind in enumerate(LAYERS):
        upper = np.minimum(cap, tops[kind])
        lower = tops[LAYERS[i + 1]] if i + 1 < len(LAYERS) else np.full_like(upper, z_bottom)
        lower = np.maximum(lower, z_bottom)
        if not (upper > lower + 0.01).any():
            continue
        poly = (f"M{X[0]:.2f},{sec.Y(upper[0]):.2f}"
                + "".join(f"L{x:.2f},{sec.Y(z):.2f}" for x, z in zip(X[1:], upper[1:]))
                + "".join(f"L{x:.2f},{sec.Y(z):.2f}" for x, z in zip(X[::-1], np.minimum(lower, upper)[::-1]))
                + "Z")
        if layer_names:
            svg.layer(layer_names[kind])
        cid = svg.clip_path(poly)
        svg.open_group(clip_path=cid)
        _texture(svg, kind, float(X[0]), float(X[-1]), float(sec.Y(np.nanmax(upper))), float(sec.Y(np.nanmin(lower))),
               ink, rng, weight)
        svg.close_group()
        # Boundary line along the layer's top where it lies below ground.
        buried = np.where(tops[kind] < cap - 0.05, tops[kind], np.nan)
        svg.path(points_to_path(X, sec.Y(buried)), fill="none", stroke=ink, stroke_width=weight * 1.5, opacity=0.9)
        if label:
            thick = upper - np.minimum(lower, upper)
            k = int(np.nanargmax(thick))
            if thick[k] > 1.0:
                mid = (upper[k] + max(lower[k], z_bottom)) / 2
                lx = float(np.clip(X[k], X[0] + 45, X[-1] - 45))
                svg.text(lx, sec.Y(mid) + 0.7, NAMES[kind], size=1.8, fill=ink, anchor="middle", spacing=0.2)
    return tops

