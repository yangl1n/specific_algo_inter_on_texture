import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kymatio.scattering2d.frontend.base_frontend import ScatteringBase2D
from kymatio.frontend.torch_frontend import ScatteringTorch

import matplotlib.pyplot as plt

def imshow_complex(tensor, name = 'tst', channel = None, title=None):
    """Utility function to visualize complex tensors as RGB images."""
    # tensor is expected to have shape (..., 2) where the last dimension is (real, imag)
    
    if channel is not None:
        tensor = tensor[channel, ...]  # Select the specified channel
    real = tensor[..., 0]
    imag = tensor[..., 1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(real.cpu().numpy(), cmap='gray')
    ax[0].set_title('Real Part')
    ax[0].axis('off')
    ax[1].imshow(imag.cpu().numpy(), cmap='gray')
    ax[1].set_title('Imaginary Part')
    ax[1].axis('off')
    plt.suptitle(title)
    plt.savefig(f'{name}_channel{channel}.png')
    plt.close(fig)

def imshow_real(tensor, name = 'tst', channel = None, title=None):
    """Utility function to visualize real tensors as grayscale images."""
    if channel is not None:
        tensor = tensor[channel, ...]  # Select the specified channel

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tensor.cpu().numpy(), cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.savefig(f'{name}_channel{channel}.png')
    plt.close(fig)

class ScatteringTorch2D_wph(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=1, pre_pad=False, backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters - convert complex numpy arrays to torch tensors with (..., 2) representation
        # Manually stack real and imaginary parts to get (..., 2) shape
        phi_np = self.phi['levels'][0]
        phi_real_imag = np.stack([phi_np.real, phi_np.imag], axis=-1).astype(np.float32)
        self.register_buffer('phis', torch.from_numpy(phi_real_imag))

        psis = []
        for j in range(len(self.psi)):
            psi_np = self.psi[j]['levels'][0]
            psi_real_imag = np.stack([psi_np.real, psi_np.imag], axis=-1).astype(np.float32)
            psis.append(torch.from_numpy(psi_real_imag))

        # Stack psis: each psi is (M, N, 2), stack to get (L, M, N, 2)
        self.register_buffer('psis', torch.stack(psis, dim=0))

    def set_filters_trainable(self, trainable):
        """Set whether filters are trainable (no-op for base class)."""
        pass

    def filter_named_parameters(self, prefix=""):
        """Iterate over filter parameters (yields nothing for base class)."""
        return
        yield  # Make this a generator

    def scattering(self, input_x):
        """Forward pass of the scattering.

            Parameters
            ----------
            input_x : tensor
                Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
                arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
                :math:`n_2` is the number of input channels.

            Raises
            ------
            RuntimeError
                In the event that the input does not have at least two
                dimensions, or the tensor is not contiguous, or the tensor is
                not of the correct spatial size, padded or not.
            TypeError
                In the event that the input is not a Torch tensor.

            Returns
            -------
            S : tensor
                Scattering of the input, a tensor with k+3 dimensions :math:`(n_1, ...,n_k, D, Md, Nd)`
                where :math:`D` corresponds to a new channel dimension and :math:`(Md, Nd)` are
                downsampled sizes by a factor :math:`2^J`. Currently, k=2 is hardcoded.

        """
        if not torch.is_tensor(input_x):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input_x.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input_x.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')
        if (input_x.shape[-1] != self.shape[0] or input_x.shape[-2] != self.shape[1]) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.shape[1], self.shape[0]))

        if (input_x.shape[-1] != self._N_padded or input_x.shape[-2] != self._M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self._M_padded, self._N_padded))

        batch_shape = input_x.shape[:-2]
        signal_shape = input_x.shape[-2:]
        input_x = input_x.reshape((-1,) + signal_shape)

        S = phase_scattering2d(input_x, self.pad, self.unpad, self.backend, self.J, self.phis, self.psis)
        scattering_shape = S.shape[-3:]
        S = S.reshape(batch_shape + scattering_shape)

        return S
    
    def _sync_filters_to_parent_attributes(self):
        """Sync trainable filter parameters back to parent's self.phi and self.psi.
        
        This ensures that self.phi and self.psi reflect the current trained filter values,
        preventing stale non-trainable copies from being used elsewhere.
        """
        # Update phi
        if hasattr(self, 'phis') and 'levels' in self.phi:
            self.phi['levels'][0] = self.phis
        
        # Update psi
        if hasattr(self, 'psis') and 'levels' in self.psi:
            for j in range(len(self.psi)):
                if j < len(self.psis):
                    self.psi[j]['levels'][0] = self.psis[j]
    
    def __repr__(self):
        return (f"ScatteringTorch2D_wph(J={self.J}, shape={self.shape}, L={self.L}, "
                f"max_order={self.max_order}, pre_pad={self.pre_pad})")


class ScatteringTorch2D_wphTrainable(ScatteringTorch2D_wph):
    """Trainable variant of ScatteringTorch2D_wph with learnable filters.
    
    This class extends ScatteringTorch2D_wph to support trainable filter parameters.
    Filters (phis and psis) are registered as nn.Parameter instead of buffers,
    allowing them to be updated during training via gradient descent.
    
    Important: self.phi and self.psi are kept synchronized with the trained
    self.phis and self.psis to prevent stale non-trainable copies.
    """

    def __init__(self, J, shape, L=8, max_order=1, pre_pad=False, backend='torch',
                 learnable_filters=False):
        """Initialize trainable scattering transform.
        
        Parameters
        ----------
        J : int
            Number of scales
        shape : tuple
            Spatial shape (M, N)
        L : int, optional
            Number of orientations (default: 8)
        max_order : int, optional
            Maximum scattering order (default: 1)
        pre_pad : bool, optional
            Whether to pre-pad (default: False)
        backend : str, optional
            Backend type (default: 'torch')
        learnable_filters : bool, optional
            Whether filters are trainable (default: False - frozen during phase_a)
        """
        self.learnable_filters = bool(learnable_filters)
        super().__init__(J, shape, L, max_order, pre_pad, backend)

    def register_filters(self):
        """Register filters directly as nn.Parameter for gradient tracking."""
        # Create the filters - convert complex numpy arrays to torch tensors with (..., 2) representation
        # Manually stack real and imaginary parts to get (..., 2) shape
        phi_np = self.phi['levels'][0]
        phi_real_imag = np.stack([phi_np.real, phi_np.imag], axis=-1).astype(np.float32)
        # Parameters (not buffers) so the filters can be trained.
        self.phis = nn.Parameter(torch.from_numpy(phi_real_imag))
        self.phis.requires_grad_(self.learnable_filters)

        psis = []
        for j in range(len(self.psi)):
            psi_np = self.psi[j]['levels'][0]
            psi_real_imag = np.stack([psi_np.real, psi_np.imag], axis=-1).astype(np.float32)
            psis.append(torch.from_numpy(psi_real_imag))

        # (L, M, N, 2)
        self.psis = nn.Parameter(torch.stack(psis, dim=0))
        self.psis.requires_grad_(self.learnable_filters)



    def set_filters_trainable(self, trainable):
        """Set whether filters are trainable.
        
        Parameters
        ----------
        trainable : bool
            If True, filters will have gradients computed. If False, frozen.
        """
        self.learnable_filters = bool(trainable)
        if hasattr(self, 'phis'):
            self.phis.requires_grad = self.learnable_filters
        if hasattr(self, 'psis'):
            self.psis.requires_grad = self.learnable_filters

    def filter_named_parameters(self, prefix=""):
        """Iterate over filter parameters.
        
        Yields
        ------
        tuple
            (name, parameter) for each filter parameter
        """
        if hasattr(self, 'phis'):
            yield (prefix + "phis" if prefix else "phis", self.phis)
        if hasattr(self, 'psis'):
            yield (prefix + "psis" if prefix else "psis", self.psis)

    def scattering(self, input_x):
        """Forward pass with filter synchronization.
        
        Ensures that self.phi and self.psi stay synchronized with trained
        self.phis and self.psis to prevent stale copies from being used.
        """
        # Sync trained parameters to parent attributes before forward pass
        self._sync_filters_to_parent_attributes()
        # Call parent's scattering method (which uses self.phis and self.psis)
        return super().scattering(input_x)

    def __repr__(self):
        trainable_str = "trainable" if self.learnable_filters else "frozen"
        return (f"ScatteringTorch2D_wphTrainable(J={self.J}, shape={self.shape}, L={self.L}, "
                f"max_order={self.max_order}, pre_pad={self.pre_pad}, filters={trainable_str})")


def phase_scattering2d(x, pad, unpad, backend, J, phi, psi):
    subsample_fourier = backend.subsample_fourier
    rfft = backend.rfft
    irfft = backend.irfft
    ifft = backend.ifft
    cdgmm = backend.cdgmm

    def unpad_real(tensor):
        # backend.unpad expects a trailing complex dimension; irfft already adds it
        if tensor.shape[-1] == 1:
            return unpad(tensor)
        return unpad(tensor.unsqueeze(-1))

    out_S = []

    # x is of size (B, C, M_ori, N_ori)

    U_r = pad(x).unsqueeze(1)  # size (BC, 1, M, N)
    U_0_c = rfft(U_r)  # (BC, 1, M, N, 2) - old-style complex with trailing 2

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi)  # (BC, 1, M, N, 2) - old-style complex
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)  # (BC, 1, M//2, N//2, 2)
    S_0 = irfft(U_1_c)  # (BC, 1, M//2, N//2) - real
    S_0 = unpad_real(S_0)  # (BC, 1, M//2-2, N//2-2)
    out_S.append(S_0)
    
    # Apply each psi filter separately since cdgmm expects B to be (M, N, 2)
    U_1_list = []
    for l in range(psi.shape[0]):  # Loop over L filters
        U_1_c = cdgmm(U_0_c, psi[l])  # (BC, 1, M, N, 2) - old-style complex
        U_1_c = subsample_fourier(U_1_c, k=2 ** J)  # (BC, 1, M//2, N//2, 2)
        U_1_c = ifft(U_1_c)  # (BC, 1, M', N', 2) - old-style complex
        # Concatenate [real, imag] parts for U_1_c and -U_1_c along last dimension
        S_1_l = F.relu(torch.cat([U_1_c, -U_1_c], dim=-1))  # (BC, 1, M', N', 4)
        S_1_l = S_1_l.squeeze(1)  # (BC, M', N', 4)
        S_1_l = S_1_l.permute(0, -1, 1, 2).contiguous()  # (BC, 4, M', N')
        S_1_l = unpad_real(S_1_l)  # (BC, 4, M'-pad, N'-pad)
        U_1_list.append(S_1_l)
    
    S_1 = torch.cat(U_1_list, dim=1)  # (BC, L*4, M'-pad, N'-pad)
    out_S.append(S_1)  # (BC, L*4, M'-pad, N'-pad)

    out_S = torch.cat(out_S, dim=1)
    return out_S


__all__ = ['ScatteringTorch2D_wph', 'ScatteringTorch2D_wphTrainable']
