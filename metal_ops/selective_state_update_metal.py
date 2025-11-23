"""
Metal implementation of selective state update operation.

Provides drop-in replacement for mamba_ssm.ops.triton.selective_state_update.
"""

import torch
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


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,  # Renamed from B to avoid conflict with batch size
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
) -> torch.Tensor:
    """
    Selective state space update for inference.
    
    Updates the state in-place and returns the output.
    
    Args:
        state: State tensor (B, H, E, N) - updated in-place
        x: Input tensor (B, H, E)
        dt: Delta tensor (B, H, E)
        A: A matrix (H, E, N) - log space
        B_mat: B matrix (B, G, N)
        C: C matrix (B, G, N)
        D: Optional D skip connection (H, E) or (H,)
        z: Optional gating tensor (B, H, E)
        dt_bias: Optional dt bias (H, E) or (H,)
        dt_softplus: Whether to apply softplus to dt
        
    Returns:
        Output tensor (B, H, E)
    """
    ctx = _get_context()
    
    # Get dimensions
    B, H, E, N = state.shape
    assert x.shape == (B, H, E), f"x shape {x.shape} doesn't match expected ({B}, {H}, {E})"
    assert dt.shape == (B, H, E), f"dt shape {dt.shape} doesn't match expected ({B}, {H}, {E})"
    
    # Infer ngroups from B_mat shape
    _, G, N_b = B_mat.shape
    assert N_b == N, f"B state dimension {N_b} doesn't match state {N}"
    
    # Convert to numpy
    state_np = state.detach().cpu().numpy().astype(np.float32)
    x_np = x.detach().cpu().numpy().astype(np.float32)
    dt_np = dt.detach().cpu().numpy().astype(np.float32)
    A_np = A.detach().cpu().numpy().astype(np.float32)
    B_np = B_mat.detach().cpu().numpy().astype(np.float32)
    C_np = C.detach().cpu().numpy().astype(np.float32)
    
    D_np = D.detach().cpu().numpy().astype(np.float32) if D is not None else np.zeros((H, E), dtype=np.float32)
    if D_np.ndim == 1:  # (H,) -> (H, E)
        D_np = np.repeat(D_np[:, np.newaxis], E, axis=1)
    
    z_np = z.detach().cpu().numpy().astype(np.float32) if z is not None else None
    dt_bias_np = dt_bias.detach().cpu().numpy().astype(np.float32) if dt_bias is not None else np.zeros((H, E), dtype=np.float32)
    if dt_bias_np.ndim == 1:  # (H,) -> (H, E)
        dt_bias_np = np.repeat(dt_bias_np[:, np.newaxis], E, axis=1)
    
    # Create Metal buffers
    state_buf = numpy_to_metal_buffer(ctx, state_np)
    x_buf = numpy_to_metal_buffer(ctx, x_np)
    dt_buf = numpy_to_metal_buffer(ctx, dt_np)
    dt_bias_buf = numpy_to_metal_buffer(ctx, dt_bias_np)
    A_buf = numpy_to_metal_buffer(ctx, A_np)
    B_buf = numpy_to_metal_buffer(ctx, B_np)
    C_buf = numpy_to_metal_buffer(ctx, C_np)
    D_buf = numpy_to_metal_buffer(ctx, D_np)
    z_buf = numpy_to_metal_buffer(ctx, z_np) if z_np is not None else None
    out_buf = allocate_metal_buffer(ctx, B * H * E)
    
    # Get pipeline
    pipeline = ctx.get_pipeline(ctx.library, "selective_scan_update_kernel")
    
    # Create command buffer and encoder
    cmd = create_command_buffer(ctx)
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    
    # Set buffers (order matches kernel signature)
    enc.setBuffer_offset_atIndex_(state_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(x_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(dt_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(dt_bias_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(A_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(B_buf, 0, 5)
    enc.setBuffer_offset_atIndex_(C_buf, 0, 6)
    enc.setBuffer_offset_atIndex_(D_buf, 0, 7)
    enc.setBuffer_offset_atIndex_(z_buf, 0, 8)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 9)
    
    # Set parameters
    for i, val in enumerate([B, H, E, N, G, int(dt_softplus), int(z is not None)], 10):
        enc.setBytes_length_atIndex_(np.uint32(val).tobytes(), 4, i)
    
    # Dispatch
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(E, H, B),
        Metal.MTLSizeMake(8, 8, 1)
    )
    enc.endEncoding()
    
    # Execute
    execute_and_wait(cmd)
    
    # Update state in-place
    new_state = metal_buffer_to_numpy(state_buf, (B, H, E, N))
    state.copy_(torch.from_numpy(new_state))
    
    # Get output
    result = metal_buffer_to_numpy(out_buf, (B, H, E))
    return torch.from_numpy(result).to(x.device).to(x.dtype)
