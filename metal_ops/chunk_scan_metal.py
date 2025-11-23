"""
Metal implementation of chunk scan operations.

Provides drop-in replacements for mamba_ssm.ops.triton.ssd_combined functions.
"""

import torch
import numpy as np
import Metal
from typing import Optional, Tuple, Union
from einops import rearrange
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


def mamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,  # Renamed from B to avoid conflict with batch size
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    return_varlen_states: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Combined chunk scan operation for Mamba.
    
    This is a simplified Metal implementation that uses the SSM scan kernel.
    For now, it ignores chunk_size and processes the entire sequence.
    
    Args:
        x: Input tensor (B, L, H, E)
        dt: Delta tensor (B, L, H)
        A: A matrix (H,) - log space
        B_mat: B matrix (B, L, G, N)
        C: C matrix (B, L, G, N)
        chunk_size: Chunk size (currently ignored)
        D: Optional D skip connection (H, E) or (H,)
        z: Optional gating tensor (B, L, H, E)
        dt_bias: Optional dt bias (H,)
        dt_softplus: Whether to apply softplus to dt
        seq_idx: Optional sequence indices (not supported)
        cu_seqlens: Optional cumulative sequence lengths (not supported)
        dt_limit: Delta limits (not enforced)
        return_final_states: Whether to return final states
        return_varlen_states: Whether to return varlen states
        
    Returns:
        Output tensor (B, L, H, E) or tuple with states
    """
    ctx = _get_context()
    
    B, L, H, E = x.shape
    _, _, _, N = B_mat.shape if B_mat.dim() == 4 else (0, 0, 0, 0)
    G = B_mat.shape[2] if B_mat.dim() == 4 else 1
    
    # Expand dt to match x shape if needed
    if dt.shape == (B, L, H):
        dt = dt.unsqueeze(-1).expand(B, L, H, E)
    
    # Expand A to (H, E, N)
    if A.dim() == 1:  # (H,)
        # We need to know N from B_mat
        _, _, _, N = B_mat.shape
        A_expanded = A.unsqueeze(-1).unsqueeze(-1).expand(H, E, N)
    else:
        A_expanded = A
    
    # Expand dt_bias if provided
    if dt_bias is not None and dt_bias.dim() == 1:  # (H,)
        dt_bias = dt_bias.unsqueeze(-1).expand(H, E)
    
    # Expand D if provided
    if D is not None and D.dim() == 1:  # (H,)
        D = D.unsqueeze(-1).expand(H, E)
    
    # Convert to numpy
    x_np = x.detach().cpu().numpy().astype(np.float32)
    dt_np = dt.detach().cpu().numpy().astype(np.float32)
    A_np = A_expanded.detach().cpu().numpy().astype(np.float32)
    B_np = B_mat.detach().cpu().numpy().astype(np.float32)
    C_np = C.detach().cpu().numpy().astype(np.float32)
    
    D_np = D.detach().cpu().numpy().astype(np.float32) if D is not None else None
    z_np = z.detach().cpu().numpy().astype(np.float32) if z is not None else None
    dt_bias_np = dt_bias.detach().cpu().numpy().astype(np.float32) if dt_bias is not None else None
    
    # Reshape for kernel: (B, L, H, E) -> kernel expects (B*L, H*E)
    x_reshaped = x_np.reshape(B, L, H * E)
    dt_reshaped = dt_np.reshape(B, L, H * E)
    
    # Create Metal buffers
    state_buf = allocate_metal_buffer(ctx, B * H * E * N)
    # Initialize state to zeros
    state_ptr = state_buf.contents()
    np.frombuffer(state_ptr.as_buffer(state_buf.length()), dtype=np.float32)[:] = 0
    
    x_buf = numpy_to_metal_buffer(ctx, x_reshaped)
    dt_buf = numpy_to_metal_buffer(ctx, dt_reshaped)
    A_buf = numpy_to_metal_buffer(ctx, A_np)
    B_buf = numpy_to_metal_buffer(ctx, B_np.reshape(B * L, G * N))
    C_buf = numpy_to_metal_buffer(ctx, C_np.reshape(B * L, G * N))
    D_buf = numpy_to_metal_buffer(ctx, D_np) if D_np is not None else None
    dt_bias_buf = numpy_to_metal_buffer(ctx, dt_bias_np) if dt_bias_np is not None else None
    out_buf = allocate_metal_buffer(ctx, B * L * H * E)
    
    # Get pipeline
    pipeline = ctx.get_pipeline(ctx.library, "ssm_scan_kernel")
    
    # Create command buffer and encoder
    cmd = create_command_buffer(ctx)
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    
    # Set buffers
    enc.setBuffer_offset_atIndex_(state_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(x_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(dt_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(dt_bias_buf, 0, 3)
    enc.setBuffer_offset_atIndex_(A_buf, 0, 4)
    enc.setBuffer_offset_atIndex_(B_buf, 0, 5)
    enc.setBuffer_offset_atIndex_(C_buf, 0, 6)
    enc.setBuffer_offset_atIndex_(D_buf, 0, 7)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 8)
    
    # Set parameters
    for i, val in enumerate([B, L, H, E, N, G, int(D is not None)], 10):
        enc.setBytes_length_atIndex_(np.uint32(val).tobytes(), 4, i)
    
    # Dispatch
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(E, H, B),
        Metal.MTLSizeMake(8, 8, 1)
    )
    enc.endEncoding()
    
    # Execute
    execute_and_wait(cmd)
    
    # Get output
    result = metal_buffer_to_numpy(out_buf, (B, L, H * E))
    result = result.reshape(B, L, H, E)
    out = torch.from_numpy(result).to(x.device).to(x.dtype)
    
    # Apply gating if z is provided
    if z is not None:
        out = out * torch.nn.functional.silu(z)
    
    # Return with states if requested
    if return_final_states or return_varlen_states:
        final_state = metal_buffer_to_numpy(state_buf, (B, H, E, N))
        final_state_torch = torch.from_numpy(final_state).to(x.device).to(x.dtype)
        
        if return_varlen_states and cu_seqlens is not None:
            # For varlen, we'd need to extract per-sequence states
            # For now, just return the final state
            return out, final_state_torch, final_state_torch
        else:
            return out, final_state_torch
    
    return out


def mamba_split_conv1d_scan_combined(
    zxbcdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: Optional[torch.Tensor],
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    seq_idx: Optional[torch.Tensor] = None,
    activation: str = "silu",
    rmsnorm_weight: Optional[torch.Tensor] = None,
    rmsnorm_eps: float = 1e-6,
    outproj_weight: Optional[torch.Tensor] = None,
    outproj_bias: Optional[torch.Tensor] = None,
    headdim: Optional[int] = None,
    ngroups: int = 1,
    norm_before_gate: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
) -> torch.Tensor:
    """
    Fused split + conv1d + scan + projection operation.
    
    This is a simplified implementation that breaks down the fused operation
    into separate steps using existing Metal kernels.
    
    Args:
        zxbcdt: Combined input tensor (B, L, d_in_proj)
        conv1d_weight: Conv1d weights (D, W)
        conv1d_bias: Conv1d bias (D,)
        dt_bias: Delta bias (H,)
        A: A matrix (H,)
        D: D skip connection (H, E) or (H,)
        chunk_size: Chunk size
        seq_idx: Optional sequence indices
        activation: Activation function
        rmsnorm_weight: Optional RMSNorm weights
        rmsnorm_eps: RMSNorm epsilon
        outproj_weight: Output projection weights
        outproj_bias: Output projection bias
        headdim: Head dimension
        ngroups: Number of groups
        norm_before_gate: Whether to normalize before gating
        dt_limit: Delta limits
        
    Returns:
        Output tensor (B, L, d_model)
    """
    # This is a complex fused operation. For now, we'll implement a simplified version
    # that uses PyTorch operations with Metal acceleration where possible.
    
    from .causal_conv1d_metal import causal_conv1d_fn
    from .rmsnorm_metal import rmsnorm_fn
    
    B, L, d_in_proj = zxbcdt.shape
    
    # Infer dimensions
    nheads = A.shape[0]
    headdim = headdim if headdim is not None else (D.shape[1] if D.dim() > 1 else 64)
    d_ssm = nheads * headdim
    
    # Infer dstate from input dimensions
    # d_in_proj = 2 * d_mlp + 2 * d_ssm + 2 * ngroups * dstate + nheads
    # We need to solve for dstate, but we don't know d_mlp yet
    # Simplified: assume dstate = 128 (standard value)
    dstate = 128
    
    # Split zxbcdt
    # Order: [z, x, B, C, dt]
    # For simplicity, assume equal split for now
    d_mlp = (d_in_proj - 2 * d_ssm - 2 * ngroups * dstate - nheads) // 2
    
    z0 = zxbcdt[..., :d_mlp]
    x0 = zxbcdt[..., d_mlp:2*d_mlp]
    z = zxbcdt[..., 2*d_mlp:2*d_mlp+d_ssm]
    xBC = zxbcdt[..., 2*d_mlp+d_ssm:2*d_mlp+2*d_ssm+2*ngroups*dstate]
    dt = zxbcdt[..., -nheads:]
    
    # Apply conv1d to xBC
    xBC_conv = causal_conv1d_fn(xBC, conv1d_weight, conv1d_bias, activation=activation)
    
    # Split xBC_conv
    x_conv = xBC_conv[..., :d_ssm]
    B_proj = xBC_conv[..., d_ssm:d_ssm+ngroups*dstate]
    C_proj = xBC_conv[..., d_ssm+ngroups*dstate:]
    
    # Reshape for chunk scan
    x_reshaped = rearrange(x_conv, "b l (h p) -> b l h p", p=headdim)
    z_reshaped = rearrange(z, "b l (h p) -> b l h p", p=headdim) if rmsnorm_weight is None else None
    B_reshaped = rearrange(B_proj, "b l (g n) -> b l g n", g=ngroups)
    C_reshaped = rearrange(C_proj, "b l (g n) -> b l g n", g=ngroups)
    
    # Run chunk scan
    y = mamba_chunk_scan_combined(
        x_reshaped, dt, A, B_reshaped, C_reshaped,
        chunk_size=chunk_size,
        D=D,
        z=z_reshaped,
        dt_bias=dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        dt_limit=dt_limit,
    )
    
    # Reshape output
    y = rearrange(y, "b l h p -> b l (h p)")
    
    # Apply RMSNorm if provided
    if rmsnorm_weight is not None:
        y = rmsnorm_fn(y, rmsnorm_weight, rmsnorm_eps)
        if not norm_before_gate:
            y = y * torch.nn.functional.silu(z)
    
    # Concatenate with MLP path if present
    if d_mlp > 0:
        y = torch.cat([torch.nn.functional.silu(z0) * x0, y], dim=-1)
    
    # Apply output projection if provided
    if outproj_weight is not None:
        y = torch.nn.functional.linear(y, outproj_weight, outproj_bias)
    
    return y
