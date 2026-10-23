"""Key plan: a small, simplified plan placed on a section sheet to show where
the section is cut."""

from drawing import terrain
from drawing.svg import north_arrow, points_to_path, scale_bar


def key_plan(svg, dem, x, y, window, cut_row, cut_cols, dam_row, dam_cols, ink, scale=20000,
             step=4, valley_z=228.0, angle=0.0, other_cuts=(), layers=None, label=("A", "A′"),
             water_below=223.2, names=()):
    """Draw a key plan with its top-left corner at (x, y) mm.

    window: DEM (row0, row1, col0, col1) shown. The river (lidar water above and
    below the dam), the valley edge (a smoothed contour at valley_z), the dam and
    the cut line are drawn; the cut gets end ticks, letters and view arrows
    (looking north, the direction the section is seen in). other_cuts: further
    (row, (c0, c1)) lines drawn thin and dashed for context.
    layers: optional dict with "lines", "cut" and "text" layer names.
    names: (text, row, col, angle) place names.
    Returns the plan's (width, height) in mm.
    """
    def lay(key):
        if layers:
            svg.layer(layers[key])

    r0, r1, c0, c1 = window
    a = dem[r0:r1, c0:c1]
    s = 1000.0 / scale                         # mm per m (DEM cells are 1 m)
    w, h = a.shape[1] * s, a.shape[0] * s

    _, wu, wd = terrain.river_reaches(a, dam_row - r0, water_below)
    water = (wu | wd)[::step, ::step]

    def P(col, row):
        return x + col * s, y + row * s

    lay("lines")
    clip = svg.clip_rect(x, y, w, h)
    svg.open_group(clip_path=clip)
    svg.rect(x, y, w, h, fill="none", stroke=ink, stroke_width=0.1)
    for ln in terrain.smooth_contours(a[::step, ::step], [valley_z], sigma=1.5)[valley_z]:
        if len(ln) > 10:
            svg.path(points_to_path(x + ln[:, 0] * step * s, y + ln[:, 1] * step * s), fill="none",
                     stroke=ink, stroke_width=0.035, opacity=0.45)
    for ln in terrain.mask_lines(water):
        svg.path(points_to_path(x + ln[:, 0] * step * s, y + ln[:, 1] * step * s), fill="none",
                 stroke=ink, stroke_width=0.12)
    xa, ya = P(dam_cols[0] - c0, dam_row - r0)
    xb, _ = P(dam_cols[1] - c0, dam_row - r0)
    svg.line(xa, ya, xb, ya, stroke=ink, stroke_width=0.35)
    for row, (ca, cb) in other_cuts:
        xa, ya = P(ca - c0, row - r0)
        xb, _ = P(cb - c0, row - r0)
        svg.line(xa, ya, xb, ya, stroke=ink, stroke_width=0.08, stroke_dasharray="0.6 0.6", opacity=0.7)
    svg.close_group()

    # The cut: heavy line, end ticks, view arrows (looking north) and letters.
    lay("cut")
    xa, ya = P(cut_cols[0] - c0, cut_row - r0)
    xb, _ = P(cut_cols[1] - c0, cut_row - r0)
    svg.line(xa, ya, xb, ya, stroke=ink, stroke_width=0.3, stroke_dasharray="2 0.6 0.4 0.6")
    for xe in (xa, xb):
        svg.line(xe, ya - 2.2, xe, ya + 2.2, stroke=ink, stroke_width=0.3)
        svg.path(f"M{xe - 1.0:.2f},{ya - 1.6:.2f} L{xe:.2f},{ya - 3.2:.2f} L{xe + 1.0:.2f},{ya - 1.6:.2f} Z",
                 fill=ink, stroke="none")
    lay("text")
    svg.text(xa - 1.5, ya + 0.8, label[0], size=2.4, fill=ink, anchor="end")
    svg.text(xb + 1.5, ya + 0.8, label[1], size=2.4, fill=ink)

    for text, row, col, ang in names:
        px, py = P(col - c0, row - r0)
        svg.text(px, py, text, size=1.5, fill=ink, anchor="middle", spacing=0.15, opacity=0.8,
                 rotate=ang or None)
    svg.text(x, y - 2.5, f"KEY PLAN · 1:{scale:,}", size=2.0, fill=ink, spacing=0.25)
    svg.text(x + w, y - 2.5, "DAM · RIVER · VALLEY EDGE", size=1.6, fill=ink, anchor="end", opacity=0.7)
    lay("lines")
    scale_bar(svg, x, y + h + 3, 500 * s, 500, "m", color=ink)
    north_arrow(svg, x + w - 4, y + h + 5, r=2.5, color=ink, angle=angle)
    return w, h
