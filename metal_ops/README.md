# Metal Operations for Mamba

This package provides macOS Metal-accelerated implementations of Mamba operations as drop-in replacements for CUDA/Triton operations.

## Overview

The `metal_ops` package includes:

- **causal_conv1d_metal.py**: Causal 1D convolution operations
  - `causal_conv1d_fn` - Forward pass
  - `causal_conv1d_update` - Single-step update for inference
  - `causal_conv1d_varlen_states` - Variable-length sequence state extraction

- **selective_state_update_metal.py**: Selective state space update
  - `selective_state_update` - In-place state update with output

- **rmsnorm_metal.py**: RMSNorm with gating
  - `RMSNormGated` - PyTorch module
  - `rmsnorm_fn` - Functional interface

- **chunk_scan_metal.py**: Chunk-based scanning operations
  - `mamba_chunk_scan_combined` - Combined chunk scan
  - `mamba_split_conv1d_scan_combined` - Fused conv1d + scan + projection

- **metal_utils.py**: Shared utilities
  - `MetalContext` - Device and kernel management
  - Buffer allocation and conversion helpers

## Requirements

- macOS with Metal support (Apple Silicon or Intel with Metal-capable GPU)
- Python 3.8+
- PyTorch
- pyobjc-framework-Metal

Install dependencies:
```bash
pip install torch einops pyobjc-framework-Metal
```

## Usage

### Using Mamba2Metal

```python
from mamba2_metal import Mamba2Metal

# Create model
model = Mamba2Metal(
    d_model=768,
    d_state=128,
    d_conv=4,
    expand=2,
    headdim=64,
    ngroups=8,
)

# Forward pass
import torch
x = torch.randn(2, 128, 768)
y = model(x)

# Inference with caching
conv_state, ssm_state = model.allocate_inference_cache(batch_size=1, max_seqlen=1024)
x_step = torch.randn(1, 1, 768)
y_step, conv_state, ssm_state = model.step(x_step, conv_state, ssm_state)
```

### Using Individual Operations

```python
from metal_ops import causal_conv1d_fn, selective_state_update, RMSNormGated

# Causal conv1d
x = torch.randn(2, 100, 256)
weight = torch.randn(256, 4)
bias = torch.randn(256)
y = causal_conv1d_fn(x, weight, bias, activation="silu")

# RMSNorm with gating
norm = RMSNormGated(hidden_size=768)
x = torch.randn(2, 10, 768)
z = torch.randn(2, 10, 768)
y = norm(x, z)

# Selective state update
state = torch.zeros(2, 12, 64, 128)  # (B, H, E, N)
x = torch.randn(2, 12, 64)
dt = torch.randn(2, 12, 64)
A = torch.randn(12, 64, 128)
B = torch.randn(2, 8, 128)
C = torch.randn(2, 8, 128)
D = torch.randn(12, 64)
y = selective_state_update(state, x, dt, A, B, C, D, dt_softplus=True)
```

## Architecture

### Metal Kernels

The Metal kernels are located in `src/metal/mamba_ssm_full.metal` and include:

- `rms_norm_kernel` - RMSNorm computation
- `linear_proj_kernel` - Linear projection
- `conv1d_causal_kernel` - Causal 1D convolution with SiLU activation
- `ssm_scan_kernel` - SSM sequential scan
- `gating_kernel` - Gating with SiLU
- `selective_scan_update_kernel` - Single-step selective state update

### Python Wrappers

Python wrappers handle:
- PyTorch tensor conversion to/from Metal buffers
- Kernel dispatch and synchronization
- Shape handling and broadcasting
- Fallback to CPU when Metal is unavailable

## Performance

Metal acceleration provides significant speedups on Apple Silicon:

- **M1/M2/M3**: 2-5x faster than CPU for typical sequence lengths
- **Memory Efficient**: Shared memory buffers reduce CPU-GPU transfer overhead
- **Optimized Kernels**: Hand-tuned Metal shaders for SSM operations

## Limitations

Current limitations compared to CUDA version:

1. **Single Device Only**: No distributed training support (tensor parallelism disabled)
2. **Variable-Length Sequences**: Limited support for `cu_seqlens` parameter
3. **Precision**: Float32 only (no FP16/BF16 support yet)
4. **Chunk Size**: Simplified chunk scanning (processes full sequence)

## Testing

Run the test suite:

```bash
python test_mamba2_metal.py
```

This will test:
- Individual Metal operations
- Mamba2Metal module forward pass
- Inference caching and single-step generation
- Compatibility with original Mamba2 (if available)

## Troubleshooting

### Metal not available

If you see "Metal not available", ensure:
- You're running on macOS
- pyobjc-framework-Metal is installed: `pip install pyobjc-framework-Metal`
- Your Mac has Metal support (all Apple Silicon and most Intel Macs do)

### Kernel compilation errors

If Metal kernels fail to compile:
- Check that `src/metal/mamba_ssm_full.metal` exists
- Verify the Metal shader syntax is valid
- Try updating macOS to the latest version

### Shape mismatches

If you encounter shape errors:
- Ensure input tensors match expected dimensions
- Check that `headdim` divides `d_ssm` evenly
- Verify `ngroups` divides `nheads` evenly

## Contributing

To extend the Metal operations:

1. Add new kernel to `src/metal/mamba_ssm_full.metal`
2. Create Python wrapper in `metal_ops/`
3. Add to `metal_ops/__init__.py`
4. Update tests in `test_mamba2_metal.py`

## License

Same as the parent Mamba project.
