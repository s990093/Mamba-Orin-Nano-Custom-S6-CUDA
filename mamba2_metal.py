"""
macOS Metal implementation of Mamba2.

This module provides a drop-in replacement for the original Mamba2 module
that uses Metal acceleration on macOS instead of CUDA/Triton.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

# Import Metal operations
try:
    from metal_ops import (
        causal_conv1d_fn,
        causal_conv1d_update,
        causal_conv1d_varlen_states,
        selective_state_update,
        RMSNormGated,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        METAL_AVAILABLE,
    )
except ImportError:
    print("[Mamba2Metal] Warning: metal_ops not available, falling back to CPU")
    METAL_AVAILABLE = False
    causal_conv1d_fn = None
    causal_conv1d_update = None
    causal_conv1d_varlen_states = None
    selective_state_update = None
    RMSNormGated = None
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    # Fallback if huggingface_hub not available
    class PyTorchModelHubMixin:
        pass


class Mamba2Metal(nn.Module, PyTorchModelHubMixin):
    """
    Mamba2 module with Metal acceleration for macOS.
    
    This is a drop-in replacement for the original Mamba2 module that uses
    Metal instead of CUDA/Triton. Distributed training features are simplified
    for single-device usage.
    """
    
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        process_group=None,  # Ignored for Metal
        sequence_parallel=True,  # Ignored for Metal
        device=None,
        dtype=None,
    ):
        """
        Initialize Mamba2Metal module.
        
        Args match the original Mamba2 module. process_group and sequence_parallel
        are ignored as Metal implementation is single-device only.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Basic parameters
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        
        # Simplified for single device (no distributed training)
        self.process_group = None
        self.sequence_parallel = False
        self.world_size = 1
        self.local_rank = 0
        
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.ngroups = ngroups
        
        assert self.d_ssm % self.headdim == 0, f"d_ssm ({self.d_ssm}) must be divisible by headdim ({self.headdim})"
        self.nheads = self.d_ssm // self.headdim
        
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path and METAL_AVAILABLE
        self.layer_idx = layer_idx
        
        # Input projection
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        
        # Conv1d
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        
        self.act = nn.SiLU()
        
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        # Initialize A
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True
        
        # RMSNorm
        if self.rmsnorm:
            if RMSNormGated is not None:
                self.norm = RMSNormGated(
                    self.d_ssm,
                    eps=1e-5,
                    norm_before_gate=self.norm_before_gate,
                    group_size=self.d_ssm // ngroups,
                    **factory_kwargs
                )
            else:
                # Fallback to standard LayerNorm
                self.norm = nn.LayerNorm(self.d_ssm, eps=1e-5, **factory_kwargs)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
    
    def forward(
        self,
        u: torch.Tensor,
        seqlen: Optional[int] = None,
        seq_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            u: Input tensor (batch, seqlen, hidden_dim) or (batch * seqlen, hidden_dim)
            seqlen: Optional sequence length for packed format
            seq_idx: Optional sequence indices (not fully supported)
            cu_seqlens: Optional cumulative sequence lengths (limited support)
            inference_params: Optional inference parameters for caching
            
        Returns:
            Output tensor same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen
        
        # Handle inference caching
        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # Single-step inference
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
        
        # Input projection
        zxbcdt = self.in_proj(u)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # Compute A
        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # Use memory-efficient fused path if available
        if self.use_mem_eff_path and inference_params is None and mamba_split_conv1d_scan_combined is not None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
        else:
            # Fallback path: split and process separately
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            
            # Update conv state if in inference mode
            if conv_state is not None:
                if cu_seqlens is None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))
                else:
                    if causal_conv1d_varlen_states is not None:
                        assert batch == 1, "varlen inference only supports batch dimension 1"
                        conv_varlen_states = causal_conv1d_varlen_states(
                            xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                        )
                        conv_state.copy_(conv_varlen_states)
            
            # Apply conv1d
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            
            # Split xBC
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            
            # Run SSM scan
            if mamba_chunk_scan_combined is not None:
                y = mamba_chunk_scan_combined(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    cu_seqlens=cu_seqlens,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                    return_varlen_states=cu_seqlens is not None and inference_params is not None,
                )
                
                # Handle state returns
                if ssm_state is not None:
                    y, last_state, *rest = y
                    if cu_seqlens is None:
                        ssm_state.copy_(last_state)
                    else:
                        varlen_states = rest[0]
                        ssm_state.copy_(varlen_states)
                
                y = rearrange(y, "b l h p -> b l (h p)")
            else:
                # CPU fallback - implement basic SSM scan
                raise NotImplementedError("CPU fallback for SSM scan not yet implemented")
            
            # Apply normalization
            if self.rmsnorm:
                if hasattr(self.norm, 'forward') and len(inspect.signature(self.norm.forward).parameters) > 1:
                    y = self.norm(y, z)
                else:
                    y = self.norm(y)
            
            # Concatenate with MLP path
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            
            # Reshape if needed
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            
            # Output projection
            out = self.out_proj(y)
        
        return out
    
    def step(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """
        Single-step inference.
        
        Args:
            hidden_states: Input (B, 1, D)
            conv_state: Convolution state (B, D_conv, W)
            ssm_state: SSM state (B, H, E, N)
            
        Returns:
            Tuple of (output, conv_state, ssm_state)
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        
        # Project input
        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        
        # Split
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        
        # SSM step
        if selective_state_update is None:
            # CPU fallback
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        else:
            # Metal accelerated
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        
        # Normalization
        if self.rmsnorm:
            if hasattr(self.norm, 'forward') and len(inspect.signature(self.norm.forward).parameters) > 1:
                y = self.norm(y, z)
            else:
                y = self.norm(y)
        
        # MLP path
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        
        # Output projection
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Allocate inference cache for fast generation."""
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state
    
    def _get_states_from_cache(self, inference_params, batch_size: int, initialize_states: bool = False):
        """Get or create states from inference cache."""
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


# Add missing import
import inspect
