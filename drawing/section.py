"""Vertical sections: a terrain profile along a DEM row, and the datum grid."""

import numpy as np

from drawing.svg import points_to_path


class Section:
    """Maps chainage (m along the line) and elevation (m) to sheet mm."""

    def __init__(self, x0, y0, h_scale, v_scale, z_top):
        self.x0, self.y0 = x0, y0
        self.sh = 1000.0 / h_scale           # mm per metre, horizontal
        self.sv = 1000.0 / v_scale           # mm per metre, vertical
        self.z_top = z_top

    def X(self, ch):
        return self.x0 + np.asarray(ch) * self.sh

    def Y(self, z):
        return self.y0 + (self.z_top - np.asarray(z)) * self.sv

    def path(self, ch, z):
        return points_to_path(self.X(ch), self.Y(z))


def profile(dem, row, c0, c1):
    """Ground along a DEM row from column c0 to c1: chainage (m) and z."""
    z = dem[row, c0:c1].astype(float)
    return np.arange(len(z), dtype=float), z


def water_edges(ground, level, near):
    """Chainages where a water level meets the banks either side of `near`
    (an index inside the channel): the first ground above the level outward."""
    left = near
    while left > 0 and ground[left - 1] <= level:
        left -= 1
    right = near
    while right < len(ground) - 1 and ground[right + 1] <= level:
        right += 1
    return left, right


def datum_grid(svg, sec, width_m, z_bottom, z_top, ink, minor=1.0, major=5.0, chain=50.0,
               weight=0.03, label_size=1.9, side_labels=True):
    """Dotted elevation lines every `minor` m (labelled every `major` m, as
    +230 etc.), and chainage lines every `chain` m."""
    x0, x1 = sec.X(0), sec.X(width_m)
    for z in np.arange(np.ceil(z_bottom), np.floor(z_top) + 0.001, minor):
        is_major = abs(z / major - round(z / major)) < 1e-6
        y = sec.Y(z)
        svg.line(x0, y, x1, y, stroke=ink, stroke_width=weight * (1.6 if is_major else 1.0),
                 stroke_dasharray="0.15 0.85" if not is_major else "0.15 0.45", opacity=0.45 if is_major else 0.25)
        if is_major and side_labels:
            svg.text(x0 - 2, y + label_size * 0.35, f"+{z:.0f}", size=label_size, fill=ink, anchor="end")
            svg.text(x1 + 2, y + label_size * 0.35, f"+{z:.0f}", size=label_size, fill=ink)
    for ch in np.arange(0, width_m + 0.001, chain):
        x = sec.X(ch)
        svg.line(x, sec.Y(z_top), x, sec.Y(z_bottom), stroke=ink, stroke_width=weight,
                 stroke_dasharray="0.15 0.85", opacity=0.2)
        svg.text(x, sec.Y(z_bottom) + label_size + 1.5, f"{ch:.0f}", size=label_size * 0.85, fill=ink,
                 anchor="middle", opacity=0.7)
