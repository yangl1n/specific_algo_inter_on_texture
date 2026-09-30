"""Checkpoint -> filter arrays.

Storage (as written by spatial_scattering.WaveletConv):

* block b's band-pass bank is two *independent* real Conv2d weights
  ``psi_real``/``psi_imag`` of shape (C*L, h, k, k); the complex filter is
  ``W_r + i W_i``.  Output channel ``i*L + l`` is orientation ``l`` of input channel
  ``i``, and its diagonal ("own channel") kernel is input slot ``i % h``.  The other
  ``h - 1`` slots are the cross-channel mixing kernels (zero at init, out of scope).
* the low-pass is a separate real Conv2d ``phi_conv`` of shape (C, h, k, k), diagonal
  slot ``i % h``.  It is not part of the band-pass bank.
* weights are cross-correlation kernels (PyTorch conv), zero padding ``k // 2``.

Everything returned is numpy float64 / complex128.
"""

import numpy as np
import torch

from . import DS_ROOT  # noqa: F401  (puts deep_scatter/ on sys.path)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_block_state(path):
    """Extractor weights from any checkpoint flavour, keyed ``"<b>.wavelets.…"``.

    Accepts best.pth / curr.pth / phase_a.pth (``{"model": state}`` with ``blocks.``
    prefixes) and param_snapshot.data / scattering_anchor.data (already block-keyed).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    out = {}
    for k, v in obj.items():
        if k.startswith("blocks."):
            k = k[len("blocks."):]
        elif not k.split(".", 1)[0].isdigit():
            continue
        out[k] = v.detach().double().numpy()
    if not out:
        raise ValueError(f"{path}: no extractor weights found")
    return out


def n_blocks(state):
    return 1 + max(int(k.split(".", 1)[0]) for k in state)


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------

def nominal_theta(L):
    """Init orientation of band-pass index l, exactly as WaveletConv._init_weights."""
    return np.array([(int(L - L / 2 - 1) - l) * np.pi / L for l in range(L)])


# ---------------------------------------------------------------------------
# stored (per-block) filters
# ---------------------------------------------------------------------------

def extract_filters(state):
    """Diagonal filters of every block.

    Returns a list, one dict per block ``b``::

        lowpass   (C, k, k)        real      phi diagonal, input channel c
        bandpass  (C, L, k, k)     complex   psi diagonal, or None if lowpass-only
        pool      (2, 2)           real      learnable pooling kernel
        C, L, h, k                 ints      channels, orientations, group size, kernel
    """
    blocks = []
    for b in range(n_blocks(state)):
        phi = state[f"{b}.wavelets.phi_conv.weight"]
        C, h, k, _ = phi.shape
        rows = np.arange(C)
        entry = dict(C=C, h=h, k=k, L=None,
                     lowpass=phi[rows, rows % h],
                     pool=state[f"{b}.pool.conv.weight"][0, 0],
                     bandpass=None)
        key_r = f"{b}.wavelets.psi_real.weight"
        if key_r in state:
            wr, wi = state[key_r], state[f"{b}.wavelets.psi_imag.weight"]
            L = wr.shape[0] // C
            w = (wr + 1j * wi).reshape(C, L, h, k, k)
            entry["bandpass"] = w[rows, :, rows % h]          # (C, L, k, k)
            entry["L"] = L
        blocks.append(entry)
    return blocks



def reference_bank(L, kernel_size):
    """The init Morlets and Gaussian, taken from WaveletConv itself (not re-derived).

    Returns ``(lowpass (k, k) real, bandpass (L, k, k) complex)``.  At init every
    channel of every block carries exactly these kernels.
    """
    from spatial_scattering import WaveletConv
    wc = WaveletConv(1, L, kernel_size=kernel_size)
    lp = wc.phi_conv.weight.detach().double().numpy()[0, 0]
    bp = (wc.psi_real.weight + 1j * wc.psi_imag.weight).detach().numpy()[:, 0]
    return lp, bp.astype(np.complex128)

