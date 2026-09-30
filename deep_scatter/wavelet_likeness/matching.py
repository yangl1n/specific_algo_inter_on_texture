"""Gauge-invariant comparison and orientation matching.

For real inputs a filter is only defined up to a complex gain alpha (the modulus
|alpha g * x| differs from |g * x| by |alpha|, which is not a shape change) and up to
conjugation (|conj(g) * x| = |g * x|).  So the distance between a filter g and a
reference h is

    r(g, h) = min_{alpha, conj} ||g~ - alpha h|| / ||g||
            = sqrt(1 - max(|<h, g>|, |<h, conj g>|)^2 / (||g||^2 ||h||^2)).

Spatial shift is *not* quotiented here; the Morlet fit carries u0 for that.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

_EPS = 1e-30


def gauge_residuals(g, ref):
    """Pairwise gauge-invariant residuals.

    g (N, L, n, n), ref (L, n, n) or (N, L, n, n) ->
        R (N, L, L)   R[n, k, j] = r(g[n, k], ref[j])
        conj (N, L, L) bool, True where conj(g) matched better
    """
    N, L = g.shape[:2]
    v = g.reshape(N, L, -1)
    h = ref.reshape((-1, L, v.shape[-1]) if ref.ndim == 4 else (1, L, v.shape[-1]))
    h = np.broadcast_to(h, (N, L, v.shape[-1]))
    ip = np.abs(np.einsum("nji,nki->nkj", h.conj(), v))
    ipc = np.abs(np.einsum("nji,nki->nkj", h.conj(), v.conj()))
    nv = np.linalg.norm(v, axis=2)[:, :, None]
    nh = np.linalg.norm(h, axis=2)[:, None, :]
    den = np.maximum(nv * nh, _EPS)
    best = np.maximum(ip, ipc) / den
    R = np.sqrt(np.clip(1.0 - best ** 2, 0.0, 1.0))
    R[np.broadcast_to(nv, R.shape) < _EPS] = 1.0
    return R, ipc > ip


def match(g, ref):
    """Assign each filter of each bank to a reference orientation.

    Returns (matched_index (N, L) int, used_conjugate (N, L) bool,
    match_residual (N, L)).  Linear assignment on R, so a bank whose orientations
    were permuted by training is matched back rather than scored as destroyed.
    """
    R, conj = gauge_residuals(g, ref)
    N, L = R.shape[:2]
    idx = np.empty((N, L), int)
    for i in range(N):
        rows, cols = linear_sum_assignment(R[i])
        idx[i, rows] = cols
    ar = np.arange(L)[None]
    ni = np.arange(N)[:, None]
    return idx, conj[ni, ar, idx], R[ni, ar, idx]
