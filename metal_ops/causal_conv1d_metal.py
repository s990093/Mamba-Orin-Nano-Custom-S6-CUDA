"""
Metal implementation of causal 1D convolution operations.

Provides drop-in replacements for causal_conv1d package functions.
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
    set_buffer_data,
    create_command_buffer,
    execute_and_wait,
)


# Global context
_ctx = None

def _get_context():
    global _ctx
    if _ctx is None:
        _ctx = MetalContext()
        # Compile the kernel library
        _ctx.library = _ctx.compile_library("src/metal/mamba_ssm_full.metal", "mamba_kernels")
    return _ctx


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    seq_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Causal 1D convolution forward pass.
    
    Args:
        x: Input tensor (B, L, D) or (B, D, L)
        weight: Conv weights (D, 1, W) or (D, W)
        bias: Optional bias (D,)
        activation: Activation function ("silu" or "swish")
        seq_idx: Optional sequence indices for variable-length sequences
        
    Returns:
        Output tensor same shape as x
    """
    ctx = _get_context()
    
    # Handle input shape
    # The function can receive (B, L, D) or (B, D, L)
    # Conv1d typically uses (B, D, L) but we'll work with (B, L, D) internally
    orig_shape = x.shape
    if x.dim() == 3:
        # Determine which format we have by checking weight
        # weight is (D, W) or (D, 1, W)
        if weight.dim() == 3:
            weight = weight.squeeze(1)  # (D, 1, W) -> (D, W)
        D_w, W = weight.shape
        
        # If x is (B, D, L) where D matches weight, transpose to (B, L, D)
        if x.shape[1] == D_w:
            x = x.transpose(1, 2)  # (B, D, L) -> (B, L, D)
            was_transposed = True
        else:
            was_transposed = False
        
        B, L, D = x.shape
        assert D == D_w, f"Weight dimension {D_w} doesn't match input dimension {D}"
    else:
        raise ValueError(f"Expected 3D input, got shape {x.shape}")
    
    # Convert to numpy
    x_np = x.detach().cpu().numpy().astype(np.float32)
    weight_np = weight.detach().cpu().numpy().astype(np.float32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32) if bias is not None else np.zeros(D, dtype=np.float32)
    
    # Create Metal buffers
    x_buf = numpy_to_metal_buffer(ctx, x_np)
    weight_buf = numpy_to_metal_buffer(ctx, weight_np)
    bias_buf = numpy_to_metal_buffer(ctx, bias_np)
    out_buf = allocate_metal_buffer(ctx, B * L * D)
    
    # Get pipeline
    pipeline = ctx.get_pipeline(ctx.library, "conv1d_causal_kernel")
    
    # Create command buffer and encoder
    cmd = create_command_buffer(ctx)
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    
    # Set buffers
    enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
    
    # Set parameters
    for i, val in enumerate([B, L, D, W], 4):
        enc.setBytes_length_atIndex_(np.uint32(val).tobytes(), 4, i)
    
    # Dispatch
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(D, L, B),
        Metal.MTLSizeMake(32, 4, 1)
    )
    enc.endEncoding()
    
    # Execute
    execute_and_wait(cmd)
    
    # Convert back to torch
    result = metal_buffer_to_numpy(out_buf, (B, L, D))
    return torch.from_numpy(result).to(x.device).to(x.dtype)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Causal 1D convolution update for inference (single time step).
    
    Args:
        x: Input tensor (B, D)
        conv_state: Convolution state (B, D, W)
        weight: Conv weights (D, W)
        bias: Optional bias (D,)
        activation: Activation function
        
    Returns:
        Output tensor (B, D)
    """
    ctx = _get_context()
    
    B, D = x.shape
    _, D_s, W = conv_state.shape
    assert D == D_s, f"Dimension mismatch: x has {D}, conv_state has {D_s}"
    
    # Handle weight shape
    if weight.dim() == 3:
        weight = weight.squeeze(1)
    D_w, W_w = weight.shape
    assert D_w == D and W_w == W
    
    # Convert to numpy
    x_np = x.detach().cpu().numpy().astype(np.float32)
    conv_state_np = conv_state.detach().cpu().numpy().astype(np.float32)
    weight_np = weight.detach().cpu().numpy().astype(np.float32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32) if bias is not None else np.zeros(D, dtype=np.float32)
    
    # Update conv_state: shift left and append new input
    # conv_state shape: (B, D, W)
    new_state = np.roll(conv_state_np, -1, axis=2)
    new_state[:, :, -1] = x_np
    
    # Compute output: sum over window dimension
    out_np = np.sum(new_state * weight_np[np.newaxis, :, :], axis=2) + bias_np[np.newaxis, :]
    
    # Apply activation
    if activation in ["silu", "swish"]:
        out_np = out_np * (1.0 / (1.0 + np.exp(-out_np)))
    
    # Update conv_state in-place
    conv_state.copy_(torch.from_numpy(new_state))
    
    return torch.from_numpy(out_np).to(x.device).to(x.dtype)


def causal_conv1d_varlen_states(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_len: int,
) -> torch.Tensor:
    """
    Extract variable-length sequence states for causal conv1d.
    
    Args:
        x: Input tensor (total_tokens, D) - packed sequences
        cu_seqlens: Cumulative sequence lengths (num_seqs + 1,)
        state_len: Length of conv state to extract
        
    Returns:
        States tensor (num_seqs, D, state_len)
    """
    # This is a CPU implementation for now
    # For Metal, we'd need a custom kernel
    
    cu_seqlens_np = cu_seqlens.cpu().numpy()
    num_seqs = len(cu_seqlens_np) - 1
    _, D = x.shape
    
    states = torch.zeros(num_seqs, D, state_len, dtype=x.dtype, device=x.device)
    
    for i in range(num_seqs):
        start = cu_seqlens_np[i]
        end = cu_seqlens_np[i + 1]
        seq_len = end - start
        
        # Extract last state_len tokens (or pad if shorter)
        if seq_len >= state_len:
            states[i] = x[end - state_len:end].T
        else:
            states[i, :, -seq_len:] = x[start:end].T
    
    return states
