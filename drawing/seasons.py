"""Season-based time axis.

Instead of equal months, each phase of the river year gets horizontal space
by how much is happening: the long ice-covered winter is compressed and the
few weeks of break-up and flood are stretched.
"""

import datetime as dt

import numpy as np


# (name, start month, start day, weight). Phases run in order from the start
# of the water year; each ends where the next begins.
RED_RIVER_PHASES = [
    ("FREEZE-UP", 11, 1, 1.0),
    ("ICE COVER", 12, 1, 0.3),
    ("BREAK-UP", 3, 25, 2.6),
    ("FLOOD CREST", 4, 20, 2.0),
    ("RECESSION", 5, 25, 0.8),
    ("SUMMER LOW", 8, 1, 0.5),
]


def _wy_index(month, day, start_month=11, start_day=1):
    """Day index (0..364) of month/day within a water year that starts on
    start_month/start_day, using a non-leap calendar."""
    ref = dt.date(2001, start_month, start_day)
    d = dt.date(2001 if (month, day) >= (start_month, start_day) else 2002, month, day)
    return (d - ref).days


def doy_to_month_day(doy):
    d = dt.date(2001, 1, 1) + dt.timedelta(days=doy - 1)
    return d.month, d.day


class PhaseAxis:
    def __init__(self, phases=RED_RIVER_PHASES, x0=0.0, x1=1.0):
        self.phases = phases
        starts = [_wy_index(m, d) for _, m, d, _ in phases] + [365]
        if starts[0] != 0:
            raise ValueError("First phase must start the water year")
        widths = np.array([(starts[i + 1] - starts[i]) * phases[i][3] for i in range(len(phases))])
        edges_x = np.concatenate([[0], np.cumsum(widths)]) / widths.sum()
        self.day_edges = np.array(starts, dtype=float)
        self.x_edges = x0 + edges_x * (x1 - x0)

    def x(self, day_index):
        return np.interp(day_index, self.day_edges, self.x_edges)

    def boundaries(self):
        """[(name, x_start, x_end, day_start)] for labelling."""
        return [
            (self.phases[i][0], self.x_edges[i], self.x_edges[i + 1], int(self.day_edges[i]))
            for i in range(len(self.phases))
        ]

    def phase_x(self, name):
        for n, xs, _, _ in self.boundaries():
            if n == name:
                return xs
        raise KeyError(name)


def phases_with_ice_dates(breakup_doy=None, freezeup_doy=None, phases=RED_RIVER_PHASES):
    """Shift BREAK-UP start to the observed median break-up (minus a week of
    ice movement) when the record carries ice flags."""
    out = list(phases)
    if breakup_doy:
        m, d = doy_to_month_day(max(breakup_doy - 7, 60))
        names = [p[0] for p in out]
        i = names.index("BREAK-UP")
        # Only accept a date that keeps the phases in order.
        if _wy_index(*out[i - 1][1:3]) < _wy_index(m, d) < _wy_index(*out[i + 1][1:3]):
            out[i] = ("BREAK-UP", m, d, out[i][3])
    return out


def day_label(day_index, start_month=11, start_day=1):
    d = dt.date(2001, start_month, start_day) + dt.timedelta(days=int(day_index))
    return d.strftime("%b %d").upper()
