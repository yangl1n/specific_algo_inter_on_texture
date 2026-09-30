import torch
import torch.nn as nn
import os
import sys
import math


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_THIS_DIR)
_DEEP_SCATTER_DIR = os.path.join(_WORKSPACE_ROOT, 'deep_scatter')
for _path in [_WORKSPACE_ROOT, _DEEP_SCATTER_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


try:
    from deep_scatter.spatial_scattering import AvgPoolConv, PhaseReLU, WaveletConv
except Exception:
    from spatial_scattering import AvgPoolConv, PhaseReLU, WaveletConv


class DeepScatterParity2D(nn.Module):
    """deep_scatter-backed scattering block (J=1).

    With ``backend='deep_scatter_4L'`` the output has C * (1 + 4L) channels, matching
    ``phase_scattering2d_torch``; with ``backend='deep_scatter'`` it has C * (1 + L).
    """

    def __init__(self, J, shape, in_channels, L=8, learnable_filters=False, random_init=False, modulus_type='phase_relu', lowpass_only=False, backend='deep_scatter'):
        super().__init__()
        if J != 1:
            raise ValueError("DeepScatterParity2D currently supports J=1 only")
        self.J = J
        self.shape = shape
        self.L = L
        self.learnable_filters = bool(learnable_filters)

        self.wavelets = WaveletConv(
            in_channels=in_channels,
            L=self.L,
            kernel_size=7,
            mixing_horizon=None,
            lowpass_only=False,
            random_init=bool(random_init),
        )
        self.pool = AvgPoolConv()
        for p in self.pool.parameters():
            p.requires_grad = False

        if modulus_type == 'phase_relu':
            self.phase_relu = PhaseReLU()
            self.phase_collapse = nn.Conv2d(
                in_channels * 4 * L, in_channels * L,
                kernel_size=1, groups=in_channels * L, bias=False)
            if not random_init:
                self.phase_collapse.weight.data.fill_(1.0 / math.sqrt(2))

        self.modulus_type = modulus_type
        self.backend = backend
        
        self.set_filters_trainable(self.learnable_filters)

    def set_filters_trainable(self, trainable):
        self.learnable_filters = bool(trainable)
        if self.wavelets is None:
            return
        for p in self.wavelets.parameters():
            p.requires_grad = self.learnable_filters

    def filter_named_parameters(self, prefix=""):
        for name, param in self.wavelets.named_parameters(prefix=prefix):
            yield name, param

    def forward(self, x):
        s0 = self.pool(self.wavelets.phi_conv(x))
        ur = self.pool(self.wavelets.psi_real(x))
        ui = self.pool(self.wavelets.psi_imag(x))

        if self.backend == 'deep_scatter_4L':
            u = torch.stack([ur, ui], dim=-1)
            s1 = self.phase_relu(u)
            s1 = s1.permute(0, 1, 4, 2, 3).contiguous().flatten(1, 2)
        elif self.backend == 'deep_scatter':
            if self.modulus_type == 'phase_relu':
                u_complex = torch.stack([ur, ui], dim=-1)       # [B, C*L, M/2, N/2, 2]
                s1 = self.phase_relu(u_complex)                 # [B, C*L, M/2, N/2, 4]
                s1 = s1.permute(0, 1, 4, 2, 3).flatten(1, 2)   # [B, 4*C*L, M/2, N/2]
                s1 = self.phase_collapse(s1)                    # [B, C*L, M/2, N/2]
            else:
                s1 = torch.sqrt(ur * ur + ui * ui + 1e-8)      # [B, C*L, M/2, N/2]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        return torch.cat([s0, s1], dim=1)
