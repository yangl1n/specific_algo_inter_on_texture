"""The one FFT convention used by every M2 quantity.

Filters are arrays indexed ``g[x, y]`` on a centred grid (axis 0 = x, axis 1 = y),
exactly as ``wavelet_kernels.gabor_2d`` lays them out.  Frequencies are in
radians/pixel in ``[-pi, pi)``: ``omega[..., 0]`` pairs with axis 0 and
``omega[..., 1]`` with axis 1, so a carrier ``exp(i xi (x cos t + y sin t))`` puts its
energy at angle ``atan2(omega_1, omega_0) = t``.

The spatial origin is the array centre (``n // 2``), not index 0.  That changes only
the phase of the DFT, never ``|g_hat|``, so every quantity here is unaffected by it.
"""

import numpy as np

PAD_STORED = 64          # zero-padded FFT size for the stored 5x5 banks


def freq_grid(P):
    """(P, P, 2) angular-frequency grid, fftshift-ed, radians/pixel in [-pi, pi)."""
    w = np.fft.fftshift(np.fft.fftfreq(P)) * 2.0 * np.pi
    return np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)


def spatial_grid(n):
    """(n, n, 2) centred integer coordinates, origin at index n // 2."""
    u = np.arange(n) - n // 2
    return np.stack(np.meshgrid(u, u, indexing="ij"), axis=-1).astype(np.float64)


def spectrum(g, P):
    """fftshift-ed zero-padded DFT of a batch ``g`` of shape (..., n, n) -> (..., P, P)."""
    return np.fft.fftshift(np.fft.fft2(g, s=(P, P), axes=(-2, -1)), axes=(-2, -1))


def dc_index(P):
    """Index of omega = 0 in an fftshift-ed length-P axis."""
    return P // 2
