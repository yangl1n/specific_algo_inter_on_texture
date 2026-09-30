"""Standalone 2D Gabor / Morlet wavelet kernel generation.

Produces small (K x K) spatial-domain kernels suitable for nn.Conv2d
initialization.  No PyTorch or kymatio dependency — pure NumPy.
"""

import numpy as np


def gabor_2d(kernel_size, sigma, theta, xi, slant=1.0):
    """2D Gabor filter on a centered spatial grid.

    psi(x, y) = g_sigma(x, y) * exp(i * xi * (x*cos(theta) + y*sin(theta)))

    Parameters
    ----------
    kernel_size : int
        Side length of the square kernel (must be odd).
    sigma : float
        Bandwidth (standard deviation of the Gaussian envelope).
    theta : float
        Orientation angle in radians.
    xi : float
        Central frequency of the complex carrier.
    slant : float
        Controls the eccentricity of the Gaussian envelope.

    Returns
    -------
    ndarray, complex64, shape (kernel_size, kernel_size)
    """
    assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
    half = kernel_size // 2
    xx, yy = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

    cos_t, sin_t = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)
    R_inv = np.array([[cos_t,  sin_t],
                      [-sin_t, cos_t]], dtype=np.float32)
    D = np.array([[1.0, 0.0],
                  [0.0, slant * slant]], dtype=np.float32)
    curv = R @ D @ R_inv / (2.0 * sigma * sigma)

    envelope = np.exp(
        -(curv[0, 0] * xx * xx
          + (curv[0, 1] + curv[1, 0]) * xx * yy
          + curv[1, 1] * yy * yy)
    )
    carrier = np.exp(
        1j * xi * (xx * cos_t + yy * sin_t)
    )

    gab = (envelope * carrier).astype(np.complex64)
    gab /= (2.0 * np.pi * sigma * sigma / slant)
    return gab


def morlet_2d(kernel_size, sigma, theta, xi, slant=0.5):
    """2D Morlet wavelet (zero-mean corrected Gabor).

    morlet = gabor(xi) - K * gabor(0),  where K ensures zero DC.

    Parameters
    ----------
    kernel_size : int
        Side length of the square kernel (must be odd).
    sigma : float
        Bandwidth parameter.
    theta : float
        Orientation angle in radians.
    xi : float
        Central frequency.
    slant : float
        Eccentricity of the Gaussian envelope.

    Returns
    -------
    ndarray, complex64, shape (kernel_size, kernel_size)
    """
    wv = gabor_2d(kernel_size, sigma, theta, xi, slant)
    wv_dc = gabor_2d(kernel_size, sigma, theta, 0.0, slant)
    K = wv.sum() / wv_dc.sum()
    return (wv - K * wv_dc).astype(np.complex64)
