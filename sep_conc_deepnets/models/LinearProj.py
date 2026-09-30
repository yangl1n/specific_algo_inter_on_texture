import torch.nn as nn
import torch
import numpy as np


class LinearProj(nn.Module):
    def __init__(self, standardization, proj, P_kernel_size=1, frame_kernel_size=1,
                 frame_stride=1, use_batchnorm=True):
        super(LinearProj, self).__init__()
        self.standardization = standardization if use_batchnorm else nn.Identity()
        self.proj = proj
        self.P_kernel_size = P_kernel_size
        self.frame_kernel_size = frame_kernel_size
        self.frame_stride = frame_stride

    @classmethod
    def create_projection_layer(cls, in_channels, proj_size, p_kernel_size, p_init, 
                                frame_kernel_size=1, frame_stride=1, use_batchnorm=True):
        """Factory method to create and initialize a LinearProj with projection layer.
        
        Args:
            in_channels: input channel dimension
            proj_size: output dimension of projection
            p_kernel_size: kernel size of projection conv
            p_init: initialization mode ('orthogonal', 'identity', 'random')
            frame_kernel_size: kernel size for frame (used in forward pass)
            frame_stride: stride for frame (used in forward pass)
            use_batchnorm: whether to use BatchNorm2d before projection
            
        Returns:
            LinearProj instance with initialized projection layer
        """
        if proj_size > in_channels:
            raise ValueError(f"Proj dimension ({proj_size}) must be <= input dimension ({in_channels})")
        
        if use_batchnorm:
            standardization = nn.BatchNorm2d(in_channels, affine=False)
        else:
            standardization = nn.Identity()
        
        proj = nn.Conv2d(in_channels, proj_size, kernel_size=p_kernel_size,
                        stride=1, padding=0, bias=False)
        cls.initialize_projection(
            proj=proj,
            p_init=p_init,
            p_kernel_size=p_kernel_size,
            in_channels=in_channels,
        )
        
        return cls(standardization=standardization, proj=proj, P_kernel_size=p_kernel_size,
                   frame_kernel_size=frame_kernel_size, frame_stride=frame_stride,
                   use_batchnorm=use_batchnorm)

    @staticmethod
    def initialize_projection(proj, p_init, p_kernel_size, in_channels):
        if p_init == 'orthogonal':
            nn.init.orthogonal_(proj.weight.data)
            return

        if p_init == 'random':
            nn.init.kaiming_uniform_(proj.weight.data, a=np.sqrt(5))
            return

        if p_init != 'identity':
            raise ValueError(f"Unsupported --p-init {p_init}")

        if p_kernel_size != 1:
            raise ValueError("P identity init requires --P-kernel-size 1")
        if proj.weight.shape[0] != in_channels:
            raise ValueError(
                f"P identity init requires P_proj_size ({proj.weight.shape[0]}) == in_channels ({in_channels})"
            )

        proj.weight.data.zero_()
        for c in range(in_channels):
            proj.weight.data[c, c, 0, 0] = 1.0

    def forward(self, x):
        output = self.standardization(x)
        if self.P_kernel_size > 1:
            output = nn.functional.pad(output, ((self.P_kernel_size-1)//2,) * 4, mode='reflect')
        output = self.proj(output)

        if self.frame_kernel_size > 1:
            # x is represented by a redundant tensor of patches of shape (B, Ck^2, M', N'), which we normalize.
            output = torch.nn.functional.unfold(output, kernel_size=self.frame_kernel_size, stride=self.frame_stride)
            output = torch.div(output, output.norm(p=2, dim=1, keepdim=True))  # (B, Ck^2, M'N')
            output = output.reshape(output.shape[0], output.shape[1],
                                    int(np.sqrt(output.shape[2])), int(np.sqrt(output.shape[2])))  # (B, Ck^2, M', N')
        else:
            # Equivalent to the above but faster.
            output = torch.div(output, output.norm(p=2, dim=1, keepdim=True))

        return output

