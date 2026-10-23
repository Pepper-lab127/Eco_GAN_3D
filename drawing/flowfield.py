"""Depth-averaged flow estimated from conveyance (an ESTIMATE, not a hydraulic
model).

Water spreads over a wetted area from an inflow edge to an outflow edge. Each
cell carries flow in proportion to its conveyance, T = depth^(5/3) (Manning's
law at a common slope), so the head phi satisfies div(T grad phi) = 0 with
phi = 1 on the inflow cells and 0 on the outflow cells, banks closed. Unit
discharge is q = -c T grad phi, scaled so the total equals the measured flow;
velocity is q / depth. Constrictions speed water up and deep water carries
more, but eddies, jets and momentum are not modelled.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve(domain, depth, inflow, outflow, discharge, cell):
    """domain, inflow, outflow: boolean grids; depth in m; discharge m3/s;
    cell: grid spacing in m. Returns (u, v, fn): velocity in m/s along
    columns (east) and rows (south), zero outside the domain, and fn, the
    discharge (m3/s) crossing each north cell face, from (r, c) to (r-1, c)."""
    H, W = domain.shape
    T = np.where(domain, np.maximum(depth, 0.05) ** (5 / 3), 0.0)
    idx = -np.ones(domain.shape, int)
    free = domain & ~inflow & ~outflow
    idx[free] = np.arange(free.sum())
    n = int(free.sum())
    phi_fixed = np.where(inflow, 1.0, 0.0)

    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    diag = np.zeros(n)
    fr, fc = np.nonzero(free)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = fr + dr, fc + dc
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        r_, c_, nr_, nc_ = fr[ok], fc[ok], nr[ok], nc[ok]
        inside = domain[nr_, nc_]
        r_, c_, nr_, nc_ = r_[inside], c_[inside], nr_[inside], nc_[inside]
        t = 2 * T[r_, c_] * T[nr_, nc_] / (T[r_, c_] + T[nr_, nc_])      # harmonic mean at the face
        me = idx[r_, c_]
        diag_add = np.bincount(me, weights=t, minlength=n)
        diag += diag_add
        nb = idx[nr_, nc_]
        unknown = nb >= 0
        rows.append(me[unknown]); cols.append(nb[unknown]); vals.append(-t[unknown])
        rhs += np.bincount(me[~unknown], weights=t[~unknown] * phi_fixed[nr_[~unknown], nc_[~unknown]], minlength=n)
    A = sparse.csr_matrix((np.concatenate(vals + [diag]),
                           (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                          shape=(n, n))
    phi = phi_fixed.copy()
    phi[free] = spsolve(A.tocsc(), rhs)
    phi[~domain] = np.nan

    # Flux leaving the inflow cells sets the scale.
    flux = 0.0
    ir, ic = np.nonzero(inflow)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = ir + dr, ic + dc
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        r_, c_, nr_, nc_ = ir[ok], ic[ok], nr[ok], nc[ok]
        go = domain[nr_, nc_] & ~inflow[nr_, nc_]
        r_, c_, nr_, nc_ = r_[go], c_[go], nr_[go], nc_[go]
        t = 2 * T[r_, c_] * T[nr_, nc_] / (T[r_, c_] + T[nr_, nc_])
        flux += float(np.sum(t * (1.0 - phi[nr_, nc_])))
    c = discharge / flux if flux > 0 else 0.0

    # Gradient by central differences inside the domain (one-sided at banks).
    p = np.where(domain, phi, np.nan)
    gy, gx = np.full(p.shape, np.nan), np.full(p.shape, np.nan)
    gx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / 2
    gy[1:-1, :] = (p[2:, :] - p[:-2, :]) / 2
    for g, fwd, bwd in ((gx, (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                        (gy, (slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        one = np.full(p.shape, np.nan)
        one[bwd] = p[fwd] - p[bwd]
        g[np.isnan(g)] = one[np.isnan(g)]
        one2 = np.full(p.shape, np.nan)
        one2[fwd] = p[fwd] - p[bwd]
        g[np.isnan(g)] = one2[np.isnan(g)]
    gx, gy = np.nan_to_num(gx) / cell, np.nan_to_num(gy) / cell
    d = np.maximum(depth, 0.05)
    # A face carries c T dphi; per metre of width that is q = -c T grad(phi).
    u = np.where(domain, -c * T * gx / d, 0.0)
    v = np.where(domain, -c * T * gy / d, 0.0)
    # Exact face discharges (conservative), for the stream function.
    fn = np.zeros(domain.shape)
    both = domain[1:] & domain[:-1]
    t = np.where(both, 2 * T[1:] * T[:-1] / np.maximum(T[1:] + T[:-1], 1e-12), 0.0)
    fn[1:] = np.where(both, c * t * (np.nan_to_num(phi[1:]) - np.nan_to_num(phi[:-1])), 0.0)
    return u, v, fn


def _sample(f, r, c):
    """Bilinear sample of grid f at fractional (row, col)."""
    H, W = f.shape
    r0, c0 = int(np.floor(r)), int(np.floor(c))
    if r0 < 0 or c0 < 0 or r0 >= H - 1 or c0 >= W - 1:
        return np.nan
    fr, fc = r - r0, c - c0
    return ((1 - fr) * (1 - fc) * f[r0, c0] + (1 - fr) * fc * f[r0, c0 + 1]
            + fr * (1 - fc) * f[r0 + 1, c0] + fr * fc * f[r0 + 1, c0 + 1])


def seeds_equal_discharge(u, v, depth, domain, row, per_line, cell):
    """Seed points along a row so each streamline carries `per_line` m3/s:
    (row, col) positions where the cumulative discharge across the row
    reaches each multiple of per_line (plus half, to centre them)."""
    q = np.where(domain[row], np.abs(v[row]) * depth[row] * cell, 0.0)
    cum = np.cumsum(q)
    total = cum[-1]
    targets = np.arange(per_line / 2, total, per_line)
    cols = np.interp(targets, cum, np.arange(len(cum)))
    return [(row - 0.5, c) for c in cols], total


def trace(u, v, domain, seed, step=0.5, max_steps=20000, min_speed=0.003):
    """Follow the flow from seed (row, col) by midpoint steps of `step`
    cells. Returns arrays rows, cols and travel time (s per cell length)."""
    inside = domain.astype(float)
    r, c = seed
    rows, cols, times = [r], [c], [0.0]
    t = 0.0
    for _ in range(max_steps):
        uu, vv = _sample(u, r, c), _sample(v, r, c)
        sp = np.hypot(uu, vv) if uu == uu else np.nan
        if not sp > min_speed:
            break
        rm, cm = r + 0.5 * step * vv / sp, c + 0.5 * step * uu / sp
        um, vm = _sample(u, rm, cm), _sample(v, rm, cm)
        spm = np.hypot(um, vm) if um == um else np.nan
        if not spm > min_speed or _sample(inside, rm, cm) < 0.5:
            break
        r, c = r + step * vm / spm, c + step * um / spm
        t += step / spm
        rows.append(r); cols.append(c); times.append(t)
    return np.array(rows), np.array(cols), np.array(times)


def stream_lines(u, v, fn, domain, per_line, cell, edge=2, levels=None):
    """Streamlines as contours of the stream function: psi is the discharge
    (m3/s) crossing each row of north faces west of a point, summed from the
    exact face discharges fn (see solve), so it is conserved from row to row;
    lines at every per_line carry equal flow and cannot merge. `edge` rows at
    the inflow and outflow edges, where the flow is held fixed, are left out.
    `levels` (m3/s) picks the lines to draw, e.g. one stream tube's band.

    Returns [(rows, cols, seconds)] in cell units along each line, oriented
    downstream, with cumulative travel time from its upstream end.
    """
    import contourpy

    psi = np.cumsum(fn, axis=1)                          # at the north face of each cell
    face = domain.copy()
    face[1:] &= domain[:-1]
    face[0] = False
    psi = np.where(face, psi, np.nan)
    psi[:edge + 1] = np.nan
    psi[-edge:] = np.nan
    if levels is None:
        total = np.nanpercentile(psi[np.isfinite(psi)], 99.5)
        levels = np.arange(per_line / 2, total, per_line)
    gen = contourpy.contour_generator(z=psi, name="serial")
    speed = np.hypot(u, v)
    out = []
    for lv in levels:
        for ln in gen.lines(lv):
            if len(ln) < 6:
                continue
            cols, rows = ln[:, 0], ln[:, 1] - 0.5             # faces lie half a cell north
            # Orient downstream: compare the line's direction with the flow.
            mid = len(ln) // 2
            r_, c_ = int(round(rows[mid])), int(round(cols[mid]))
            dr, dc = rows[mid + 1] - rows[mid - 1], cols[mid + 1] - cols[mid - 1]
            if dr * v[r_, c_] + dc * u[r_, c_] < 0:
                rows, cols = rows[::-1], cols[::-1]
            seg = np.hypot(np.diff(rows), np.diff(cols))
            sp = np.array([_sample(speed, r, c) for r, c in zip(rows[1:], cols[1:])])
            sp = np.where(sp > 0.003, sp, np.nan)
            dt = np.nan_to_num(seg / sp, nan=0.0)
            out.append((rows, cols, np.concatenate([[0.0], np.cumsum(dt)])))
    return out
