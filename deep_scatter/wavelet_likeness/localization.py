"""Non-parametric per-filter quantities (M2): norm, spectral centroid, DC leakage and
spatial moments.

Each filter is scaled to unit l2 norm first and its norm is reported separately.  Every
spectrum goes through fourier.spectrum.
"""

import numpy as np

from .fourier import dc_index, freq_grid, spatial_grid, spectrum

M2_COLUMNS = ("filter_norm", "omega_c_norm", "omega_c_angle_deg", "dc_leakage",
              "centroid_x", "centroid_y", "spatial_spread")

_EPS = 1e-30


def m2_metrics(g, P, chunk=None):
    """M2 for a batch ``g`` (N, n, n) -> dict of (N,) arrays, keys M2_COLUMNS.

        filter_norm        ||g||
        omega_c_norm       |omega_c|, the energy-weighted spectral centroid (rad/pixel)
        omega_c_angle_deg  direction of omega_c mod 180 (NaN for |omega_c| < 0.01)
        dc_leakage         |g_hat(0)| / max |g_hat|
        centroid_x/y, spatial_spread
                           mean and spread of |g|^2 on the centred grid

    The spectral centroid is measured in coordinates unwrapped around each filter's
    spectral peak: a carrier near Nyquist (xi = 3 pi / 4 on a 5x5 grid) spills past
    +-pi on the periodic DFT.  omega_c and the spatial moments seed the Morlet fit.
    """
    g = np.asarray(g)
    N, n = g.shape[0], g.shape[-1]
    if chunk is None:                      # bound the (chunk, P*P) working arrays
        chunk = max(1, min(2048, 2 ** 21 // (P * P)))
    om = freq_grid(P)
    wx0, wy0 = om[..., 0].ravel(), om[..., 1].ravel()
    grid = spatial_grid(n)
    gx, gy = grid[..., 0].ravel(), grid[..., 1].ravel()
    dc = dc_index(P) * P + dc_index(P)
    out = {k: np.empty(N) for k in M2_COLUMNS}

    for s in range(0, N, chunk):
        blk = g[s:s + chunk].reshape(-1, n, n)
        norms = np.sqrt((np.abs(blk) ** 2).sum((1, 2)))
        u = blk / np.maximum(norms, _EPS)[:, None, None]
        F = spectrum(u, P).reshape(len(u), -1)
        E = np.abs(F) ** 2
        E /= np.maximum(E.sum(1, keepdims=True), _EPS)

        # frequencies unwrapped around the spectral peak
        pk = E.argmax(1)
        wx = wx0[pk][:, None] + (wx0[None] - wx0[pk][:, None] + np.pi) % (2 * np.pi) - np.pi
        wy = wy0[pk][:, None] + (wy0[None] - wy0[pk][:, None] + np.pi) % (2 * np.pi) - np.pi
        cx, cy = (E * wx).sum(1), (E * wy).sum(1)
        wc = np.hypot(cx, cy)
        # carrier direction mod 180, so conjugate-equivalent filters agree
        ang_deg = np.degrees(np.arctan2(cy, cx)) % 180.0
        ang_deg[wc < 1e-2] = np.nan

        a2 = np.abs(u.reshape(len(u), -1)) ** 2
        mx, my = a2 @ gx, a2 @ gy
        su = np.sqrt(np.maximum(a2 @ (gx ** 2 + gy ** 2) - mx ** 2 - my ** 2, 0))

        sl = slice(s, s + len(u))
        out["filter_norm"][sl] = norms
        out["omega_c_norm"][sl] = wc
        out["omega_c_angle_deg"][sl] = ang_deg
        out["dc_leakage"][sl] = np.abs(F[:, dc]) / np.maximum(np.abs(F).max(1), _EPS)
        out["centroid_x"][sl] = mx
        out["centroid_y"][sl] = my
        out["spatial_spread"][sl] = su
        dead = norms < _EPS
        for k in M2_COLUMNS:
            if k != "filter_norm":
                out[k][sl][dead] = np.nan
    return out
