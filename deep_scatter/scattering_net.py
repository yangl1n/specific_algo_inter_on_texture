"""Scattering transform model built by stacking ScatteringBlocks.

Stacks J single-scale ScatteringBlock modules to approximate a J-order
scattering transform, followed by a linear classifier.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

from spatial_scattering import ScatteringBlock, ConcatProjectedScatteringBlock


class ScatteringNet(nn.Module):
    """Multi-order scattering network with optional Analysis layer.

    Parameters
    ----------
    n_blocks : int
        Number of stacked scattering blocks (= max scattering order).
    in_channels : int
        Input image channels (3 for RGB, 1 for grayscale).
    in_spatial : int
        Input spatial size (32 for CIFAR, 28 for MNIST).
    nb_classes : int
        Number of output classes.
    L : int
        Number of wavelet orientations per block.
    kernel_size : int
        Wavelet Conv2d kernel size (odd).
    learnable : bool
        Whether wavelet filters are learnable.
    modulus_type : str
        ``'phase_relu'`` or ``'complex_modulus'``.
    mixing_horizon : int or None
        Caps how many input channels each wavelet filter can see per block.
        ``None`` = dense convolutions (all channels visible).  Recommended
        to set (e.g. 27) when ``n_blocks >= 4`` to control memory growth.
    global_avg_pool : bool
        If True, apply adaptive average pooling to 1x1 before the
        classifier, collapsing spatial dimensions.  Dramatically reduces
        classifier size for large spatial outputs.
    lowpass_last : bool
        If True, the final block outputs only its low-pass (phi) path,
        reducing output channels from ``C_in * (1+L)`` to ``C_in``.
        This is more faithful to the standard scattering transform, which
        only uses band-pass paths as input to the next scattering order.
    use_batchnorm : bool
        If True, insert BatchNorm2d standardization after each scattering
        block and before the classifier head. If False, all such layers are
        replaced by Identity.
    random_init : bool
        If True, skip wavelet initialization and keep PyTorch defaults
        (Kaiming uniform).  Used for random-baseline experiments.
    use_concat_projection : bool
        If True, use :class:`ConcatProjectedScatteringBlock` instead of
        :class:`ScatteringBlock`. Adds optional skip connections and/or 1x1 projection
        layers for channel mixing and dimension control.
    use_skip_connections : bool
        If ``use_concat_projection=True``, whether to include skip connections
        (concatenate spatially-downsampled input).
    use_projection : bool
        If ``use_concat_projection=True``, whether to apply 1x1 projection layers.
        When False, outputs concatenated features without projection.
    projection_out_channels : list or None
        If ``use_concat_projection=True`` and ``use_projection=True`` and this is 
        provided, specifies the output channels for each projection layer. If None, 
        defaults to keeping the same channel count. Must have length ``n_blocks`` if provided.
    use_analysis : bool
        If True, add an optional Analysis tight frame layer after scattering
        blocks and before the classifier.
    analysis_frame_width : int
        Analysis frame width (-1 for 2x overcomplete default).
    analysis_frame_kernel_size : int
        Analysis frame kernel size.
    analysis_frame_stride : int
        Analysis frame stride.
    analysis_non_linearity : str
        Analysis non-linearity type: 'relu', 'absolute', or 'softshrink'.
    analysis_zero_bias : bool
        ReLU with zero bias for Analysis layer.
    analysis_frame_init_mode : str
        Analysis frame initialization mode: 'orthogonal', 'random', or 'i_minus_i'.
    """

    def __init__(self, n_blocks=3, in_channels=3, in_spatial=32,
                 nb_classes=10, L=8, kernel_size=7,
                 learnable=False, modulus_type='complex_modulus',
                 mixing_horizon=None, global_avg_pool=False,
                 lowpass_last=False, use_batchnorm=True,
                 random_init=False,
                 use_concat_projection=False, use_skip_connections=True, 
                 use_projection=True, projection_out_channels=None,
                 use_analysis=False, analysis_frame_width=-1,
                 analysis_frame_kernel_size=1, analysis_frame_stride=1,
                 analysis_non_linearity='relu', analysis_zero_bias=False,
                 analysis_frame_init_mode='orthogonal'):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        min_spatial = 2 ** n_blocks
        assert in_spatial >= min_spatial, (
            f"in_spatial={in_spatial} too small for {n_blocks} blocks "
            f"(need at least {min_spatial})"
        )

        if not use_concat_projection and (not use_skip_connections or not use_projection):
            raise ValueError(
                "use_skip_connections and use_projection only apply when "
                "use_concat_projection=True"
            )

        if projection_out_channels is not None:
            if not isinstance(projection_out_channels, (list, tuple)):
                raise ValueError("projection_out_channels must be a list or tuple")
            if len(projection_out_channels) != n_blocks:
                raise ValueError(
                    f"projection_out_channels length ({len(projection_out_channels)}) "
                    f"must match n_blocks ({n_blocks})"
                )
            if not use_projection:
                raise ValueError(
                    "projection_out_channels requires use_projection=True"
                )
        
        self.n_blocks = n_blocks
        self.global_avg_pool = global_avg_pool
        self.lowpass_last = lowpass_last
        self.use_batchnorm = use_batchnorm
        self.use_analysis = use_analysis
        self.use_concat_projection = use_concat_projection
        self.use_skip_connections = use_skip_connections
        self.use_projection = use_projection

        blocks = []
        ch = in_channels
        spatial = in_spatial
        for i in range(n_blocks):
            is_last = (i == n_blocks - 1)
            lp_only = lowpass_last and is_last
            
            if use_concat_projection:
                # out_channels only matters when projecting; default keeps the input width.
                if use_projection and projection_out_channels is not None:
                    out_ch = projection_out_channels[i]
                else:
                    out_ch = ch
                
                block = ConcatProjectedScatteringBlock(
                    in_channels=ch,
                    out_channels=out_ch,
                    L=L,
                    kernel_size=kernel_size,
                    modulus_type=modulus_type,
                    use_skip=use_skip_connections,
                    use_projection=use_projection
                )
                
                ch = block.out_channels
                
            else:
                block = ScatteringBlock(
                    in_channels=ch, L=L,
                    kernel_size=kernel_size,
                    learnable=learnable,
                    modulus_type=modulus_type,
                    mixing_horizon=mixing_horizon,
                    lowpass_only=lp_only,
                    random_init=random_init
                )
                ch = block.out_channels
            
            blocks.append(block)
            spatial //= 2

        self.blocks = nn.ModuleList(blocks)
        
        # ConcatProjectedScatteringBlock with projection has its own BatchNorm.
        if use_batchnorm and not (use_concat_projection and use_projection):
            self.standardization = nn.ModuleList(
                [nn.BatchNorm2d(b.out_channels) for b in self.blocks]
            )
        else:
            self.standardization = nn.ModuleList(
                [nn.Identity() for _ in self.blocks]
            )

        adjusted = [
            row for row in self.block_summary()
            if row.get("requested_horizon") is not None
            and row.get("mixing_horizon") is not None
            and row["mixing_horizon"] != row["requested_horizon"]
        ]
        if adjusted:
            details = ", ".join(
                f"block {r['block']}: {r['requested_horizon']}→{r['mixing_horizon']}"
                for r in adjusted
            )
            print(f"Mixing horizon adjusted for divisibility: {details}")

        if use_analysis:
            if analysis_zero_bias and analysis_non_linearity == 'relu':
                lambda_ = 1.
            elif analysis_non_linearity == 'softshrink':
                lambda_ = 1.5
            else:
                lambda_ = 0  # unused by zero-bias ReLU / absolute value
            
            self.analysis = Analysis(
                nb_channels_in=ch,
                frame_width=analysis_frame_width,
                lambda_=lambda_ / np.sqrt(analysis_frame_width if analysis_frame_width > 0 else ch * 2),
                frame_kernel_size=analysis_frame_kernel_size,
                frame_stride=analysis_frame_stride,
                non_linearity=analysis_non_linearity,
                n_space=spatial,
                frame_init_mode=analysis_frame_init_mode
            )
        else:
            self.analysis = None

        self._classifier_channels = ch
        self.bn = nn.BatchNorm2d(ch)
        if global_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(ch, nb_classes)
        else:
            self.avg_pool = None
            self.classifier = nn.Linear(ch * spatial * spatial, nb_classes)

    def block_summary(self):
        """Per-block channel and mixing-horizon info for logging."""
        rows = []
        for i, b in enumerate(self.blocks):
            if isinstance(b, ConcatProjectedScatteringBlock):
                # For ConcatProjectedScatteringBlock, get info from its internal scatter block
                rows.append({
                    "block": i,
                    "type": "ConcatProjected",
                    "in_channels": b.scatter.in_channels,
                    "out_channels": b.proj.out_channels if b.use_projection else b.out_channels,
                    "mixing_horizon": b.scatter.actual_horizon,
                    "requested_horizon": b.scatter.requested_horizon,
                    "lowpass_only": b.scatter.lowpass_only,
                })
            else:
                # Regular ScatteringBlock
                rows.append({
                    "block": i,
                    "type": "ScatteringBlock",
                    "in_channels": b.in_channels,
                    "out_channels": b.out_channels,
                    "mixing_horizon": b.actual_horizon,
                    "requested_horizon": b.requested_horizon,
                    "lowpass_only": b.lowpass_only,
                })
        return rows

    def param_summary(self):
        """Return a dict with parameter counts by component.

        Keys: ``extractor``, ``analysis`` (if present), ``classifier``, ``total``, ``trainable``.
        """
        n_ext = (sum(p.numel() for p in self.blocks.parameters())
                 + sum(p.numel() for p in self.standardization.parameters()))
        n_cls = (sum(p.numel() for p in self.bn.parameters())
                 + sum(p.numel() for p in self.classifier.parameters()))
        
        summary = {
            "extractor": n_ext,
            "classifier": n_cls,
            "total": n_ext + n_cls,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        if self.use_analysis and self.analysis is not None:
            n_ana = sum(p.numel() for p in self.analysis.parameters())
            summary["analysis"] = n_ana
            summary["total"] += n_ana
        
        return summary

    def forward(self, x):
        # Scattering blocks
        for block, standardize in zip(self.blocks, self.standardization):
            x = standardize(block(x))

        # Optional Analysis layer
        if self.use_analysis and self.analysis is not None:
            x = self.analysis(x)
        
        # Classifier
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = self.bn(x)
        x = x.flatten(1)
        return self.classifier(x)


class Analysis(nn.Module):
    """Tight frame Analysis layer with learnable frame atoms.
    
    Performs analysis using a tight frame (overcomplete basis), applies
    non-linearity, and synthesizes back. Can be used as a regularization
    layer or feature transformation.
    """
    def __init__(self, nb_channels_in, frame_width, lambda_, frame_kernel_size=1, frame_stride=1,
                 non_linearity='relu', n_space=None, frame_init_mode='orthogonal'):

        super(Analysis, self).__init__()

        input_size = nb_channels_in * frame_kernel_size ** 2
        
        if frame_width == -1:
            frame_width = nb_channels_in * 2  # default to 2x overcomplete frame
        if frame_width < nb_channels_in:
            raise ValueError(
                f"frame_width ({frame_width}) must be >= nb_channels_in ({nb_channels_in})"
            )

        frame = nn.Conv2d(nb_channels_in, frame_width, kernel_size=frame_kernel_size, stride=1, padding=0,
                          bias=False).weight.data
        self._init_frame(frame, nb_channels_in, frame_width, frame_kernel_size, frame_init_mode)

        self.trace_gram = min(input_size, frame_width)
        if frame_width > input_size:
            self.frame_norm_mean = np.sqrt(input_size / frame_width)
        else:  # controlled by Parseval regularization in this case
            self.frame_norm_mean = 1.
        frame.data *= self.frame_norm_mean / frame.data.norm(p=2, dim=(1, 2, 3), keepdim=True)

        self.frame_weight = nn.parameter.Parameter(frame)

        self.frame_width = frame_width
        self.frame_kernel_size = frame_kernel_size
        self.frame_stride = frame_stride

        self.register_buffer("lambda_", torch.FloatTensor(1).fill_(lambda_))
        self.non_linearity = non_linearity

        if self.frame_kernel_size > 1:
            shape = (1, nb_channels_in, n_space, n_space)
            ones = torch.ones(shape)
            fold_params = dict(stride=self.frame_stride, kernel_size=self.frame_kernel_size)
            patch_count = nn.functional.fold(nn.functional.unfold(ones, **fold_params),
                                             **fold_params, output_size=shape[2:])
            factor = 1 / patch_count
            self.register_buffer("factor", factor)

    def forward(self, x):
        with torch.no_grad():
            self.frame_weight.data *= self.frame_norm_mean / \
                                      self.frame_weight.data.norm(p=2, dim=(1, 2, 3), keepdim=True)

        if self.frame_kernel_size > 1:
            # In this case, x is represented by a redundant tensor of patches of shape (B, Ck^2, M', N')
            z = nn.functional.conv2d(x, self.frame_weight.reshape(self.frame_weight.shape[0], -1, 1, 1), stride=1)
        else:
            z = nn.functional.conv2d(x, self.frame_weight, stride=self.frame_stride)
        z = self._dispatch_non_linearity(z, self.lambda_)

        D_z = nn.functional.conv_transpose2d(z, self.frame_weight, stride=self.frame_stride)
        if self.frame_kernel_size > 1:
            # Average overlapping patches instead of summing them.
            D_z = D_z * self.factor

        return D_z

    @staticmethod
    def _init_frame(frame, nb_channels_in, frame_width, frame_kernel_size, frame_init_mode):
        with torch.no_grad():
            if frame_init_mode == 'orthogonal':
                nn.init.orthogonal_(frame)
                return

            if frame_init_mode == 'random':
                nn.init.kaiming_uniform_(frame, a=np.sqrt(5))
                return

            if frame_init_mode != 'i_minus_i':
                raise ValueError(f"Unsupported frame_init_mode: {frame_init_mode}")

            # Build [I, -I] over channels in flattened filter coordinates.
            frame.zero_()
            base_dim = nb_channels_in

            if frame_width > 2 * base_dim:
                raise ValueError("frame_width must be <= 2*nb_channels_in for 'i_minus_i' init")
            
            if frame_width < 2 * base_dim:
                warnings.warn(
                    f"frame_width ({frame_width}) < 2*nb_channels_in ({2 * base_dim}); using partial [I,-I] init",
                    stacklevel=2,
                )

            identity_count = min(base_dim, frame_width)
            remaining = max(frame_width - identity_count, 0)
            neg_count = min(base_dim, remaining)

            # Vectorized Identity Initialization (+I)
            if identity_count > 0:
                frame[:identity_count, :identity_count, 0, 0] = torch.eye(
                    identity_count, dtype=frame.dtype, device=frame.device
                )

            # Vectorized Negative Identity Initialization (-I)
            if neg_count > 0:
                frame[identity_count:identity_count + neg_count, :neg_count, 0, 0] = -torch.eye(
                    neg_count, dtype=frame.dtype, device=frame.device
                )

    def _dispatch_non_linearity(self, x, lambd):
        """Apply non-linearity based on configured type."""
        if self.non_linearity == 'relu':
            return nn.functional.relu(x - lambd)
        elif self.non_linearity == 'absolute':
            return torch.abs(x)
        elif self.non_linearity == 'softshrink':
            return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)
        else:
            raise ValueError(f"Unknown non-linearity: {self.non_linearity}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scattering_net(args, random_init=None, learnable=None, device=None):
    """Construct a :class:`ScatteringNet` from a parsed ``args`` namespace.

    ``random_init``/``learnable`` default to ``args.random_init``; overriding them
    builds the same architecture under a different initialization (used for the
    ``--l2-anchor scattering`` reference weights).
    """
    ri = args.random_init if random_init is None else random_init
    ln = args.random_init if learnable is None else learnable
    net = ScatteringNet(
        n_blocks=args.n_blocks,
        in_channels=args.in_channels,
        in_spatial=args.in_spatial,
        nb_classes=args.nb_classes,
        L=args.L,
        kernel_size=args.kernel_size,
        learnable=ln,
        modulus_type=args.modulus_type,
        mixing_horizon=args.mixing_horizon,
        global_avg_pool=args.global_avg_pool,
        lowpass_last=args.lowpass_last,
        use_batchnorm=args.use_batchnorm,
        random_init=ri,
        use_concat_projection=args.use_concat_projection,
        use_skip_connections=args.use_skip_connections,
        use_projection=args.use_projection,
        projection_out_channels=args.projection_out_channels,
        use_analysis=args.use_analysis,
        analysis_frame_width=args.analysis_frame_width,
        analysis_frame_kernel_size=args.analysis_frame_kernel_size,
        analysis_frame_stride=args.analysis_frame_stride,
        analysis_non_linearity=args.analysis_non_linearity,
        analysis_zero_bias=args.analysis_zero_bias,
        analysis_frame_init_mode=args.analysis_frame_init_mode,
    )
    return net if device is None else net.to(device)
