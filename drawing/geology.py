"""Subsurface layers from well logs, interpolated along a section and drawn
with conventional patterns (clay dashes, till pebbles, shale lines,
carbonate brickwork)."""

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


def _hatch(svg, kind, x0, x1, y0, y1, ink, rng, w):
    """Pattern linework over a box; the caller clips it to the layer."""
    if kind == "clay":
        step, dash, gap = 1.1, 1.4, 1.0
        for k, y in enumerate(np.arange(y0, y1, step)):
            off = (k % 2) * (dash + gap) / 2
            d = " ".join(f"M{x:.2f},{y:.2f}h{dash}" for x in np.arange(x0 - off, x1, dash + gap))
            svg.path(d, fill="none", stroke=ink, stroke_width=w, opacity=0.8)
    elif kind == "till":
        n = int((x1 - x0) * (y1 - y0) / 2.2)
        xs, ys = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
        rs = rng.uniform(0.15, 0.55, n)
        d = " ".join(f"M{x + r:.2f},{y:.2f}a{r:.2f},{r * 0.7:.2f} 0 1 0 {-2 * r:.2f},0a{r:.2f},{r * 0.7:.2f} 0 1 0 {2 * r:.2f},0"
                     for x, y, r in zip(xs[: n // 3], ys[: n // 3], rs[: n // 3]))
        svg.path(d, fill="none", stroke=ink, stroke_width=w, opacity=0.8)
        d = "".join(f"M{x:.2f},{y:.2f}h0" for x, y in zip(xs[n // 3:], ys[n // 3:]))
        svg.path(d, fill="none", stroke=ink, stroke_width=w * 2.5, stroke_linecap="round", opacity=0.7)
    elif kind == "shale":
        d = " ".join(f"M{x0:.2f},{y:.2f}H{x1:.2f}" for y in np.arange(y0, y1, 0.45))
        svg.path(d, fill="none", stroke=ink, stroke_width=w, opacity=0.75)
    elif kind == "carbonate":
        course, brick = 1.6, 3.4
        parts = [f"M{x0:.2f},{y:.2f}H{x1:.2f}" for y in np.arange(y0, y1, course)]
        for k, y in enumerate(np.arange(y0, y1, course)):
            off = (k % 2) * brick / 2
            parts += [f"M{x:.2f},{y:.2f}v{course:.2f}" for x in np.arange(x0 - off, x1, brick)]
        svg.path(" ".join(parts), fill="none", stroke=ink, stroke_width=w, opacity=0.8)


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
        _hatch(svg, kind, float(X[0]), float(X[-1]), float(sec.Y(np.nanmax(upper))), float(sec.Y(np.nanmin(lower))),
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

