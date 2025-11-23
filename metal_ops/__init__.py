"""
Metal operations package for Mamba on macOS.

This package provides Metal-accelerated implementations of Mamba operations
as drop-in replacements for CUDA/Triton operations.
"""

import platform

# Check if we're on macOS
IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    try:
        import Metal
        METAL_AVAILABLE = True
    except ImportError:
        METAL_AVAILABLE = False
        print("[metal_ops] Warning: Metal module not available. Install pyobjc-framework-Metal")
else:
    METAL_AVAILABLE = False

from .metal_utils import MetalContext

__all__ = [
    "MetalContext",
    "IS_MACOS",
    "METAL_AVAILABLE",
]

# Import operations if Metal is available
if METAL_AVAILABLE:
    from .causal_conv1d_metal import (
        causal_conv1d_fn,
        causal_conv1d_update,
        causal_conv1d_varlen_states,
    )
    from .selective_state_update_metal import selective_state_update
    from .rmsnorm_metal import RMSNormGated, rmsnorm_fn
    from .chunk_scan_metal import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )
    
    __all__.extend([
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "causal_conv1d_varlen_states",
        "selective_state_update",
        "RMSNormGated",
        "rmsnorm_fn",
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
    ])
