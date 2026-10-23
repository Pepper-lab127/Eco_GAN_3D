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
    cell: grid spacing in m. Returns (u, v) velocity in m/s along columns
    (east) and rows (south), zero outside the domain."""
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
    return u, v
