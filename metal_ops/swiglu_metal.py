"""
SwiGLU Metal Kernel Integration for MambaEngine

Provides fused SwiGLU activation (x * sigmoid(x) * y) for MLP layers.
This is the Metal equivalent of Triton's SwiGLU kernel.

Usage:
    from metal_ops.swiglu_metal import integrate_swiglu_kernel
    integrate_swiglu_kernel(engine)
"""

import numpy as np
import torch
import Metal


def integrate_swiglu_kernel(engine):
    """
    Add SwiGLU kernel to MambaEngine.
    
    Args:
        engine: MambaEngine instance
    """
    # Compile SwiGLU library
    if not hasattr(engine, 'lib_swiglu'):
        engine.lib_swiglu = engine.ctx.compile_library("src/metal/mamba_swiglu.metal", "mamba_swiglu")
    
    # Create pipelines
    engine.pipelines["swiglu_fwd"] = engine.ctx.get_pipeline(engine.lib_swiglu, "swiglu_fwd_kernel")
    engine.pipelines["swiglu_bwd"] = engine.ctx.get_pipeline(engine.lib_swiglu, "swiglu_bwd_kernel")
    engine.pipelines["silu"] = engine.ctx.get_pipeline(engine.lib_swiglu, "silu_kernel")
    
    # Add dispatch methods
    def _dispatch_swiglu(enc, x_buf, y_buf, out_buf, M, N):
        """
        Dispatch SwiGLU forward kernel.
        
        Args:
            enc: Metal compute encoder
            x_buf: Input X buffer (M, N)
            y_buf: Input Y buffer (M, N)
            out_buf: Output buffer (M, N)
            M, N: Dimensions
        """
        enc.setComputePipelineState_(engine.pipelines["swiglu_fwd"])
        enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(y_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        enc.setBytes_length_atIndex_(np.uint32(M).tobytes(), 4, 3)
        enc.setBytes_length_atIndex_(np.uint32(N).tobytes(), 4, 4)
        
        # Grid: cover all (M, N) elements
        grid_size = Metal.MTLSizeMake(N, M, 1)
        group_size = Metal.MTLSizeMake(32, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
    
    def _dispatch_silu(enc, x_buf, out_buf, N):
        """
        Dispatch SiLU activation kernel.
        
        Args:
            enc: Metal compute encoder
            x_buf: Input buffer (N,)
            out_buf: Output buffer (N,)
            N: Number of elements
        """
        enc.setComputePipelineState_(engine.pipelines["silu"])
        enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
        enc.setBytes_length_atIndex_(np.uint32(N).tobytes(), 4, 2)
        
        grid_size = Metal.MTLSizeMake(N, 1, 1)
        group_size = Metal.MTLSizeMake(256, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
    
    # Bind methods
    import types
    engine._dispatch_swiglu = types.MethodType(_dispatch_swiglu, engine)
    engine._dispatch_silu = types.MethodType(_dispatch_silu, engine)
    
    print("[SwiGLU Integration] SwiGLU kernel integrated successfully")


def swiglu_metal(x, y, ctx=None):
    """
    Apply SwiGLU activation using Metal kernel.
    
    Args:
        x: Tensor (*, N) - gate input
        y: Tensor (*, N) - value input
        ctx: MetalContext (optional, will create if None)
        
    Returns:
        out: Tensor (*, N) - x * sigmoid(x) * y
    """
    from metal_ops.metal_utils import MetalContext, numpy_to_metal_buffer, metal_buffer_to_numpy
    
    if ctx is None:
        ctx = MetalContext()
    
    # Compile kernel if needed
    if 'swiglu_fwd' not in ctx._pipelines:
        lib = ctx.compile_library("src/metal/mamba_swiglu.metal", "mamba_swiglu")
        ctx._pipelines['swiglu_fwd'] = ctx.get_pipeline(lib, "swiglu_fwd_kernel")
    
    # Flatten to 2D
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    M, N = x_flat.shape
    
    # Convert to Metal
    x_buf = numpy_to_metal_buffer(ctx, x_flat.numpy() if isinstance(x, torch.Tensor) else x_flat)
    y_buf = numpy_to_metal_buffer(ctx, y_flat.numpy() if isinstance(y, torch.Tensor) else y_flat)
    
    # Allocate output
    from metal_ops.metal_utils import allocate_metal_buffer
    out_buf = allocate_metal_buffer(ctx, M * N)
    
    # Dispatch
    cmd = ctx.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    
    enc.setComputePipelineState_(ctx._pipelines['swiglu_fwd'])
    enc.setBuffer_offset_atIndex_(x_buf, 0, 0)
    enc.setBuffer_offset_atIndex_(y_buf, 0, 1)
    enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
    enc.setBytes_length_atIndex_(np.uint32(M).tobytes(), 4, 3)
    enc.setBytes_length_atIndex_(np.uint32(N).tobytes(), 4, 4)
    
    grid_size = Metal.MTLSizeMake(N, M, 1)
    group_size = Metal.MTLSizeMake(32, 1, 1)
    enc.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
    
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    
    # Read result
    out_np = metal_buffer_to_numpy(out_buf, (M, N))
    out_tensor = torch.from_numpy(out_np).reshape(orig_shape)
    
    return out_tensor


def benchmark_swiglu(M=1024, N=1024):
    """
    Benchmark SwiGLU Metal kernel vs PyTorch.
    """
    import time
    
    x = torch.randn(M, N, dtype=torch.float32)
    y = torch.randn(M, N, dtype=torch.float32)
    
    # PyTorch reference
    start = time.perf_counter()
    for _ in range(100):
        out_torch = x * torch.sigmoid(x) * y
    end = time.perf_counter()
    torch_time = (end - start) / 100 * 1000
    
    # Metal kernel
    ctx = MetalContext()
    start = time.perf_counter()
    for _ in range(100):
        out_metal = swiglu_metal(x, y, ctx)
    end = time.perf_counter()
    metal_time = (end - start) / 100 * 1000
    
    # Verify correctness
    max_err = (out_torch - out_metal).abs().max().item()
    
    print(f"[SwiGLU Benchmark] M={M}, N={N}")
    print(f"  PyTorch: {torch_time:.3f} ms")
    print(f"  Metal:   {metal_time:.3f} ms")
    print(f"  Speedup: {torch_time/metal_time:.2f}x")
    print(f"  Max err: {max_err:.6f}")
    
    return metal_time, torch_time, max_err


if __name__ == "__main__":
    # Run benchmark
    benchmark_swiglu()
