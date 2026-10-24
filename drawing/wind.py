"""Wind at the riverbank: how measured airport wind is slowed by the trees
and the valley, from the lidar surface.

A screening model, not a flow simulation. For each hour of record (speed and
direction at Winnipeg airport, 10 m over open grass), the wind at a point is

  - scaled to the local surface by a logarithmic profile: open grass, water,
    or canopy (zero-plane displacement 0.7 h, roughness 0.1 h; inside the
    crowns the speed decays exponentially toward the ground), and
  - reduced by shelter from whatever stands upwind along that hour's
    direction: an obstacle of effective height H (lidar surface upwind minus
    the point's own height) at distance d slows the wind to g(d / H), the
    typical recovery behind a tree belt (strongest within about 5 H, back to
    full speed by about 25 H; after Heisler and DeWalle 1988).

Sixteen direction sectors are computed once each; every hour takes its
sector's field.
"""

import math

import numpy as np
from scipy import ndimage

SECTORS = 16
NAMES = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()


def sector_of(deg):
    return (np.asarray(deg) + 360 / SECTORS / 2) % 360 // (360 / SECTORS)


def shelter_curve(t):
    """Relative wind speed at t obstacle-heights downwind of a porous tree belt."""
    t = np.asarray(t, float)
    return np.clip(np.where(t < 4, 0.35, 0.35 + 0.65 * (t - 4) / 21), 0.35, 1.0)


def profile_factor(z_above, h, z0_open=0.03, z_ref=10.0, blend=60.0):
    """Mean speed at z_above m over the ground, relative to the airport's
    10 m reading, over canopy height h (0 = open ground)."""
    z_above = np.asarray(z_above, float)
    h = np.asarray(h, float)
    tree = h > 2.0
    d = np.where(tree, 0.7 * h, 0.0)
    z0 = np.where(tree, 0.1 * h, z0_open)
    # Speed at the blending height is the same everywhere; scale down to the surface.
    u_blend = math.log(blend / 0.03) / math.log(z_ref / 0.03)
    z_eff = np.maximum(z_above, d + z0 * 1.5)
    above = u_blend * np.log(np.maximum(z_eff - d, z0 * 1.01) / z0) / np.log(np.maximum(blend - d, z0 * 2) / z0)
    u_top = u_blend * np.log(np.maximum(h - d, z0 * 1.01) / z0) / np.log(np.maximum(blend - d, z0 * 2) / z0)
    inside = u_top * np.exp(-2.5 * (1 - np.clip(z_above / np.maximum(h, 1e-6), 0, 1)))
    return np.clip(np.where(tree & (z_above < h), inside, above), 0, None)


def shelter(dsm, z_abs, rows, cols, direction_from, cell=1.0, reach=300.0, step=2.0):
    """Shelter factor (0.35-1) at points (rows, cols) in DSM cells, at absolute
    heights z_abs, for wind from `direction_from` degrees (north = up = -row)."""
    toward = math.radians(direction_from)
    # Upwind lies in the direction the wind comes from.
    dr, dc = -math.cos(toward), math.sin(toward)
    r = np.ones(np.broadcast(rows, z_abs).shape)
    for dist in np.arange(step, reach + step, step):
        rr = rows + dr * dist / cell
        cc = cols + dc * dist / cell
        up = ndimage.map_coordinates(dsm, [np.ravel(rr * np.ones_like(r)), np.ravel(cc * np.ones_like(r))],
                                     order=1, mode="nearest").reshape(r.shape)
        h_eff = up - z_abs
        t = np.where(h_eff > 0.5, dist / np.maximum(h_eff, 0.5), np.inf)
        r = np.minimum(r, shelter_curve(t))
    return r


def sector_fields(dsm, dtm_at, canopy_at, rows, cols, z_abs, hourly):
    """Relative speed field per sector (dict sector -> array like z_abs), and
    the hours' sectors. dtm_at / canopy_at: ground height and canopy height at
    each point (arrays broadcastable to z_abs)."""
    deg = np.array([h[1] for h in hourly])
    sec = sector_of(deg).astype(int)
    prof = profile_factor(z_abs - dtm_at, canopy_at)
    fields = {}
    for k in np.unique(sec):
        fields[int(k)] = prof * shelter(dsm, z_abs, rows, cols, k * 360 / SECTORS)
    return fields, sec


def rose(hourly):
    """Share of hours, mean speed and share of wind energy per sector."""
    deg = np.array([h[1] for h in hourly])
    spd = np.array([h[2] for h in hourly])
    sec = sector_of(deg).astype(int)
    out = []
    for k in range(SECTORS):
        m = sec == k
        out.append(dict(name=NAMES[k], share=m.mean(), mean=float(spd[m].mean()) if m.any() else 0.0,
                        energy=float((spd[m] ** 3).sum() / (spd ** 3).sum())))
    return out
