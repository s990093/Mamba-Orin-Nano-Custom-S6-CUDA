"""
Metal implementation of RMSNorm with gating.

Provides drop-in replacement for mamba_ssm.ops.triton.layernorm_gated.RMSNorm.
"""

import torch
import torch.nn as nn
import numpy as np
import Metal
from typing import Optional
from .metal_utils import (
    MetalContext,
    numpy_to_metal_buffer,
    metal_buffer_to_numpy,
    allocate_metal_buffer,
    create_command_buffer,
    execute_and_wait,
)


# Global context
_ctx = None

def _get_context():
    global _ctx
    if _ctx is None:
        _ctx = MetalContext()
        _ctx.library = _ctx.compile_library("src/metal/mamba_ssm_full.metal", "mamba_kernels")
    return _ctx


def rmsnorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    RMSNorm functional interface.
    
    Args:
        x: Input tensor (..., D)
        weight: Scale weights (D,)
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor same shape as x
    """
    ctx = _get_context()
    
    # Flatten batch dimensions
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    N, D = x_flat.shape
    
    # Convert to numpy
    x_np = x_flat.detach().cpu().numpy().astype(np.float32)
    weight_np = weight.detach().cpu().numpy().astype(np.float32)
    
    # Create Metal buffers
    x_buf = numpy_to_metal_buffer(ctx, x_np)
    weight_buf = numpy_to_metal_buffer(ctx, weight_np)
    out_buf = allocate_metal_buffer(ctx, N * D)
    
    # Get pipeline
    pipeline = ctx.get_pipeline(ctx.library, "rms_norm_kernel")
    
    # Create command buffer and encoder
    cmd = create_command_buffer(ctx)
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    
    # Set buffers
    enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
    
    # Set parameters
    enc.setBytes_length_atIndex_(np.uint32(N).tobytes(), 4, 3)
    enc.setBytes_length_atIndex_(np.uint32(D).tobytes(), 4, 4)
    
    # Dispatch
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(N, 1, 1),
        Metal.MTLSizeMake(256, 1, 1)
    )
    enc.endEncoding()
    
    # Execute
    execute_and_wait(cmd)
    
    # Convert back to torch
    result = metal_buffer_to_numpy(out_buf, (N, D))
    return torch.from_numpy(result).reshape(orig_shape).to(x.device).to(x.dtype)


class RMSNormGated(nn.Module):
    """
    RMSNorm with optional gating.
    
    This is a drop-in replacement for mamba_ssm.ops.triton.layernorm_gated.RMSNorm.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
        group_size: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            hidden_size: Size of hidden dimension
            eps: Epsilon for numerical stability
            norm_before_gate: If True, normalize before gating; else normalize after
            group_size: Optional group size for grouped normalization
            device: Device (ignored for Metal)
            dtype: Data type (ignored, always uses float32 for Metal)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size if group_size is not None else hidden_size
        
        # Weight parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (..., hidden_size)
            z: Optional gating tensor same shape as x
            
        Returns:
            Normalized (and optionally gated) tensor
        """
        if z is None:
            # Simple RMSNorm without gating
            return rmsnorm_fn(x, self.weight, self.eps)
        
        # With gating
        if self.norm_before_gate:
            # Normalize first, then gate
            x_norm = rmsnorm_fn(x, self.weight, self.eps)
            return x_norm * torch.nn.functional.silu(z)
        else:
            # Gate first, then normalize
            x_gated = x * torch.nn.functional.silu(z)
            return rmsnorm_fn(x_gated, self.weight, self.eps)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}, norm_before_gate={self.norm_before_gate}, group_size={self.group_size}"
