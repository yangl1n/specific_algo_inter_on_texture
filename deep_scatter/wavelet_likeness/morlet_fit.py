"""Parametric Morlet fit of a band-pass filter, by variable projection.

Model (same parameterisation as ``wavelet_kernels.gabor_2d`` / ``morlet_2d``, so the
init filters are reproduced exactly with u0 = 0 and slant = s)::

    x' =  cos(t)(x - x0) + sin(t)(y - y0)
    y' = -sin(t)(x - x0) + cos(t)(y - y0)
    env = exp(-(x'^2 + s^2 y'^2) / (2 sigma^2))
    psi = alpha * (exp(i xi x') - beta) * env,   beta = sum(exp(i xi x') env) / sum(env)

beta makes psi sum to zero on the sampling grid, as morlet_2d does.  alpha (complex) is
solved in closed form for every trial of the nonlinear parameters; only those go to
``scipy.optimize.least_squares`` (TRF, bounded).

Gauge: exp(i phi) g and conj(g) give the same modulus |g * x| for real x, so alpha's
phase is free and conjugation is handled upstream (``matching.match``).  Because
psi(t + pi) = conj(psi(t)) in this family, the fitted t is canonicalised to [0, 2 pi)
and only reported mod pi.
"""

import warnings

import numpy as np
from scipy.optimize import least_squares

from .fourier import spatial_grid

PARAMS = ("theta", "xi", "sigma", "s", "x0", "y0")


def _basis(p, grid):
    t, xi, sig, s, x0, y0 = p
    ux, uy = grid[..., 0] - x0, grid[..., 1] - y0
    c, sn = np.cos(t), np.sin(t)
    xr = c * ux + sn * uy
    yr = -sn * ux + c * uy
    env = np.exp(-(xr * xr + s * s * yr * yr) / (2.0 * sig * sig))
    car = np.exp(1j * xi * xr)
    beta = (car * env).sum() / env.sum()
    return (car - beta) * env


def _project(b, g):
    """Closed-form complex gain of basis ``b`` for ``g``, and the residual."""
    bb = np.vdot(b, b).real
    if not np.isfinite(bb) or bb < 1e-300:
        return 0.0, g
    a = np.vdot(b, g) / bb
    return a, g - a * b


def _bounds(n):
    """Bounds on (theta, xi, sigma, s, x0, y0) for an n x n filter."""
    shift = max(1.5, n / 8.0)
    lo = [-np.pi, 0.02, 0.25, 0.1, -shift, -shift]
    hi = [3 * np.pi, np.pi, float(n), 10.0, shift, shift]
    return np.array(lo), np.array(hi)


def _clip_start(p, lo, hi):
    p = np.array(p, float)
    p[0] = np.mod(p[0], 2 * np.pi)
    eps = 1e-6 * (hi - lo)
    return np.clip(p, lo + eps, hi - eps)


def fit(g, starts, max_nfev=400):
    """Fit one filter ``g`` (n x n, complex) from each start and keep the best.

    ``starts`` is a list of parameter vectors in ``PARAMS`` order.  Returns a dict with
    the fitted parameters and ``rho``, the residual relative to ``||g||``.
    """
    g = np.asarray(g, complex)
    n = g.shape[-1]
    gn = np.linalg.norm(g)
    if gn == 0 or not np.isfinite(gn):
        return _empty()
    grid = spatial_grid(n)
    gu = g / gn
    lo, hi = _bounds(n)

    def resid(p):
        _, r = _project(_basis(p, grid), gu)
        return np.concatenate([r.real.ravel(), r.imag.ravel()])

    results = []
    for p0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = least_squares(resid, _clip_start(p0, lo, hi), bounds=(lo, hi),
                                  method="trf", x_scale="jac", max_nfev=max_nfev)
            results.append(r)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not results:
        return _empty()
    best = results[int(np.argmin([r.cost for r in results]))]
    p = best.x.copy()
    p[0] = np.mod(p[0], 2 * np.pi)
    _, r = _project(_basis(p, grid), gu)
    out = dict(zip(PARAMS, p))
    out["rho"] = float(np.linalg.norm(r))
    return out


def _empty():
    out = {k: np.nan for k in PARAMS}
    out["rho"] = np.nan
    return out


def moment_start(m2):
    """Start vector from the M2 moments of one filter (see localization.m2_metrics)."""
    t = np.deg2rad(m2["omega_c_angle_deg"])
    sig = max(m2["spatial_spread"] / np.sqrt(2.0), 0.3)
    return [t, max(m2["omega_c_norm"], 0.05), sig, 1.0, m2["centroid_x"], m2["centroid_y"]]


def start_set(ref_params, m2=None):
    """The multi-start set: the reference fit, a widened reference, and the M2 moments
    (both carrier directions)."""
    starts = [[ref_params[k] for k in PARAMS]]
    wide = dict(ref_params, sigma=ref_params["sigma"] * 1.4, xi=ref_params["xi"] * 0.7)
    starts.append([wide[k] for k in PARAMS])
    if m2 is not None and np.isfinite(m2["omega_c_angle_deg"]):
        ms = moment_start(m2)
        starts.append(ms)
        flip = list(ms)
        flip[0] += np.pi
        starts.append(flip)
    return starts


def wrap_deg_pi(d):
    """Wrap an angle difference in degrees to [-90, 90) (orientations are mod 180)."""
    return (np.asarray(d) + 90.0) % 180.0 - 90.0
