"""Trainable wrapper for Kymatio's Scattering2D with learnable filters."""

import torch
import torch.nn as nn
from kymatio.torch import Scattering2D as KymatioScattering2D

class Scattering2DTrainable(KymatioScattering2D):
    """Trainable variant of Kymatio's Scattering2D with learnable filters.
    
    This class extends Kymatio's Scattering2D to support trainable filter parameters.
    Filters are registered as nn.Parameter instead of buffers, allowing them to be 
    updated during training via gradient descent.
    """
    
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend='torch',
                 out_type='array', learnable_filters=False):
        """Initialize trainable Kymatio scattering.
        
        Parameters
        ----------
        J : int
            Number of scales
        shape : tuple or list
            Spatial shape (M, N)
        L : int, optional
            Number of orientations (default: 8)
        max_order : int, optional
            Maximum scattering order (default: 2)
        pre_pad : bool, optional
            Whether to pre-pad (default: False)
        backend : str, optional
            Backend type (default: 'torch')
        out_type : str, optional
            Output type 'array' or 'list' (default: 'array')
        learnable_filters : bool, optional
            Whether filters are trainable (default: False - frozen during phase_a)
        """
        self.learnable_filters = bool(learnable_filters)
        self._param_dict = {}  # Cache for filter parameters
        super().__init__(J=J, shape=shape, L=L, max_order=max_order, 
                        pre_pad=pre_pad, backend=backend, out_type=out_type)
    
    def register_single_filter(self, v, n):
        """Register a single filter as nn.Parameter instead of buffer.
        
        Parameters
        ----------
        v : numpy.ndarray
            Filter array
        n : int
            Filter index
        """
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        param_name = 'tensor' + str(n)
        # Register as Parameter instead of buffer
        setattr(self, param_name, nn.Parameter(current_filter, requires_grad=self.learnable_filters))
        self._param_dict[param_name] = current_filter
        return current_filter
    
    def load_filters(self):
        """Load filters from the module's parameters.
        
        This method loads filters that were registered as nn.Parameter instead of buffers
        and updates self.phi and self.psi with the current parameter values.
        """
        param_dict = dict(self.named_parameters())
        
        n = 0
        
        # Load phis
        phis = {k: v for k, v in self.phi.items() if k != 'levels'}
        phis['levels'] = []
        for phi_level in self.phi['levels']:
            param_name = 'tensor' + str(n)
            if param_name in param_dict:
                phis['levels'].append(param_dict[param_name])
            else:
                # Fallback to buffer if not found in parameters
                phis['levels'].append(getattr(self, param_name))
            n = n + 1
        
        # Load psis
        psis = [{} for _ in range(len(self.psi))]
        for j in range(len(self.psi)):
            psis[j] = {k: v for k, v in self.psi[j].items() if k != 'levels'}
            psis[j]['levels'] = []
            for psi_level in self.psi[j]['levels']:
                param_name = 'tensor' + str(n)
                if param_name in param_dict:
                    psis[j]['levels'].append(param_dict[param_name])
                else:
                    # Fallback to buffer if not found in parameters
                    psis[j]['levels'].append(getattr(self, param_name))
                n = n + 1
        
        # Update instance with loaded filters so they are used in forward pass
        phis, psis = self._reshape_phis_psis(phis, psis)
        self.phi = phis
        self.psi = psis
        
        return phis, psis
    
    def forward(self, x):
        """Forward pass.
        
        Note: Filters are loaded lazily via hooks to avoid performance overhead.
        On first forward pass, filters are loaded from parameters.
        After each backward pass, filters are reloaded to capture gradient updates.
        """
        # Load filters from parameters on first call or when needed
        if not hasattr(self, '_filters_loaded'):
            self.load_filters()
            self._filters_loaded = True
        return super().forward(x)
    
    def set_filters_trainable(self, trainable):
        """Set whether filters are trainable.
        
        Parameters
        ----------
        trainable : bool
            If True, filters will have gradients computed. If False, frozen.
        """
        self.learnable_filters = bool(trainable)
        for param in self.parameters():
            param.requires_grad = self.learnable_filters
        # Mark that filters need to be reloaded on next forward
        self._filters_loaded = False
    
    def filter_named_parameters(self, prefix=""):
        """Iterate over filter parameters.
        
        Yields
        ------
        tuple
            (name, parameter) for each filter parameter (all parameters in this case)
        """
        for name, param in self.named_parameters():
            if name.startswith('tensor'):  # Only yield actual filter parameters
                full_name = prefix + name if prefix else name
                yield (full_name, param)
    
    def __repr__(self):
        trainable_str = "trainable" if self.learnable_filters else "frozen"
        return (f"Scattering2DTrainable(J={self.J}, shape={self.shape}, L={self.L}, "
                f"max_order={self.max_order}, out_type='{self.out_type}', filters={trainable_str})")

    def _reshape_phis_psis(self, phis, psis):
        """No-op reshape - filters from parameters already have correct shape."""
        return phis, psis

class Scattering2DRandomFilters(KymatioScattering2D):
    """Scattering2D variant with randomly initialized learnable filters.
    
    This class extends Kymatio's Scattering2D but replaces the pre-computed wavelet
    filters with randomly initialized filters. All filters are registered as nn.Parameter
    and are trainable from the start.
    """
    
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend='torch',
                 out_type='array', init_std=0.1, seed=None):
        """Initialize scattering with random filters.
        
        Parameters
        ----------
        J : int
            Number of scales
        shape : tuple or list
            Spatial shape (M, N)
        L : int, optional
            Number of orientations (default: 8)
        max_order : int, optional
            Maximum scattering order (default: 2)
        pre_pad : bool, optional
            Whether to pre-pad (default: False)
        backend : str, optional
            Backend type (default: 'torch')
        out_type : str, optional
            Output type 'array' or 'list' (default: 'array')
        init_std : float, optional
            Standard deviation for random filter initialization (default: 0.1)
        seed : int, optional
            Random seed for reproducibility (default: None)
        """
        self.init_std = float(init_std)
        self.seed = seed
        super().__init__(J=J, shape=shape, L=L, max_order=max_order, 
                        pre_pad=pre_pad, backend=backend, out_type=out_type)
        
        # Replace all filters with random initialization
        self._randomize_filters()
    
    def _randomize_filters(self):
        """Replace all wavelet filters with random tensors.
        
        Filters are represented as complex tensors with last dimension 2 (real, imag).
        We generate random real and imaginary parts independently.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        n = 0
        
        # Randomize phis
        for i, phi_level in enumerate(self.phi['levels']):
            filter_shape = phi_level.shape
            # Generate complex-valued random filter
            # Filter shape is (..., 2) where last dim is [real, imag]
            random_filter = self._generate_complex_random(filter_shape)
            param_name = 'tensor' + str(n)
            setattr(self, param_name, nn.Parameter(random_filter, requires_grad=True))
            n = n + 1
        
        # Randomize psis
        for j in range(len(self.psi)):
            for i, psi_level in enumerate(self.psi[j]['levels']):
                filter_shape = psi_level.shape
                # Generate complex-valued random filter
                random_filter = self._generate_complex_random(filter_shape)
                param_name = 'tensor' + str(n)
                setattr(self, param_name, nn.Parameter(random_filter, requires_grad=True))
                n = n + 1
    
    def _generate_complex_random(self, filter_shape):
        """Generate a random tensor with proper complex structure.
        
        Parameters
        ----------
        filter_shape : tuple
            Shape of the filter, expected to end with 2 (for complex dimension)
        
        Returns
        -------
        torch.Tensor
            Random tensor with real and imaginary parts as last dimension
        """
        if len(filter_shape) > 0 and filter_shape[-1] == 2:
            # Complex filter: generate real and imaginary parts separately and stack them
            spatial_shape = filter_shape[:-1]
            real_part = torch.randn(spatial_shape) * self.init_std
            imag_part = torch.randn(spatial_shape) * self.init_std
            # Stack along last dimension to create [..., 2] format (not complex dtype)
            return torch.stack([real_part, imag_part], dim=-1)
        else:
            # Fallback for non-complex filters
            filters = torch.randn(filter_shape) * self.init_std
            if filters.ndim == 2:
                # If 2D, add complex dimension
                return filters.unsqueeze(-1)
            return filters
    
    def register_single_filter(self, v, n):
        """Register a single filter as nn.Parameter.
        
        In random filter mode, this is overridden during initialization,
        but we keep it for compatibility with parent class.
        
        Parameters
        ----------
        v : numpy.ndarray
            Filter array (used to determine shape)
        n : int
            Filter index
        """
        # Generate random filter with proper complex structure
        filter_shape = v.shape if hasattr(v, 'shape') else (64, 64)
        random_filter = self._generate_complex_random(filter_shape)
        param_name = 'tensor' + str(n)
        setattr(self, param_name, nn.Parameter(random_filter, requires_grad=True))
        return random_filter
    
    def load_filters(self):
        """Load filters from the module's random parameters.
        
        Retrieves the random filter parameters and updates self.phi and self.psi
        so they are used in the forward pass.
        """
        param_dict = dict(self.named_parameters())
        
        n = 0
        
        # Load phis
        phis = {k: v for k, v in self.phi.items() if k != 'levels'}
        phis['levels'] = []
        for phi_level in self.phi['levels']:
            param_name = 'tensor' + str(n)
            if param_name in param_dict:
                phis['levels'].append(param_dict[param_name])
            else:
                # Fallback if not found
                phis['levels'].append(getattr(self, param_name))
            n = n + 1
        
        # Load psis
        psis = [{} for _ in range(len(self.psi))]
        for j in range(len(self.psi)):
            psis[j] = {k: v for k, v in self.psi[j].items() if k != 'levels'}
            psis[j]['levels'] = []
            for psi_level in self.psi[j]['levels']:
                param_name = 'tensor' + str(n)
                if param_name in param_dict:
                    psis[j]['levels'].append(param_dict[param_name])
                else:
                    # Fallback if not found
                    psis[j]['levels'].append(getattr(self, param_name))
                n = n + 1
        
        # Update instance with loaded filters so they are used in forward pass
        phis, psis = self._reshape_phis_psis(phis, psis)
        self.phi = phis
        self.psi = psis
        
        return phis, psis
    
    def filter_named_parameters(self, prefix=""):
        """Iterate over filter parameters.
        
        Yields
        ------
        tuple
            (name, parameter) for each filter parameter
        """
        for name, param in self.named_parameters():
            if name.startswith('tensor'):  # Only yield actual filter parameters
                full_name = prefix + name if prefix else name
                yield (full_name, param)
    
    def forward(self, x):
        """Forward pass.
        
        Note: Filters are loaded lazily to avoid performance overhead.
        On first forward pass, random filters are loaded from parameters.
        """
        # Load filters from parameters on first call
        if not hasattr(self, '_filters_loaded'):
            self.load_filters()
            self._filters_loaded = True
        return super().forward(x)
    
    def __repr__(self):
        return (f"Scattering2DRandomFilters(J={self.J}, shape={self.shape}, L={self.L}, "
                f"max_order={self.max_order}, out_type='{self.out_type}', "
                f"init_std={self.init_std}, seed={self.seed})")
    
    def _reshape_phis_psis(self, phis, psis):
        """No-op reshape - filters from parameters already have correct shape."""
        return phis, psis


__all__ = ['Scattering2DTrainable', 'Scattering2DRandomFilters']
