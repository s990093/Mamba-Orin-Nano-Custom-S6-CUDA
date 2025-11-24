"""
Tiled Block Matrix Multiply Integration for MambaEngine

This file provides Python wrapper functions to integrate the Metal tiled BMM kernel
into the Mamba2 inference pipeline.

Usage:
    from metal_ops.mamba_bmm_wrapper import integrate_bmm_kernel
    integrate_bmm_kernel(engine)  # Adds BMM dispatch methods to MambaEngine
"""

import numpy as np
import Metal


def integrate_bmm_kernel(engine):
    """
    Integrate tiled BMM kernel into an existing MambaEngine instance.
    
    Args:
        engine: MambaEngine instance to add BMM capability to
    """
    # Compile BMM library if not already done
    if not hasattr(engine, 'lib_bmm'):
        engine.lib_bmm = engine.ctx.compile_library("src/metal/mamba_bmm.metal", "mamba_bmm")
    
    # Create pipeline with function constants
    fc = Metal.MTLFunctionConstantValues.new()
    
    # Set tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)
    # These are tuned for Apple M-series GPUs
    import ctypes
    val_64 = ctypes.c_uint(64)
    val_32 = ctypes.c_uint(32)
    val_false = ctypes.c_bool(False)
    
    fc.setConstantValue_type_atIndex_(ctypes.addressof(val_64), Metal.MTLDataTypeUInt, 0)  # BLOCK_M
    fc.setConstantValue_type_atIndex_(ctypes.addressof(val_64), Metal.MTLDataTypeUInt, 1)  # BLOCK_N
    fc.setConstantValue_type_atIndex_(ctypes.addressof(val_32), Metal.MTLDataTypeUInt, 2)  # BLOCK_K
    fc.setConstantValue_type_atIndex_(ctypes.addressof(val_false), Metal.MTLDataTypeBool, 3)  # IS_CAUSAL
    
    # Get specialized function
    desc = Metal.MTLComputePipelineDescriptor.new()
    func = engine.lib_bmm.newFunctionWithName_constantValues_error_("bmm_chunk_fwd_kernel", fc, None)[0]
    if func is None:
        raise ValueError("Failed to get bmm_chunk_fwd_kernel function")
    desc.setComputeFunction_(func)
    state, err = engine.ctx.device.newComputePipelineStateWithDescriptor_options_reflection_error_(desc, 0, None, None)
    if err:
        raise RuntimeError(f"Failed to create BMM pipeline: {err}")
    
    engine.pipelines["bmm_chunk_fwd"] = state
    
    # Add dispatch method
    def _dispatch_bmm_chunk(enc, a_buf, b_buf, c_buf, batch, M, N, K, ngroups=1,
                            stride_a=(None, None, None),
                            stride_b=(None, None, None),
                            stride_c=(None, None, None)):
        """
        Dispatch tiled BMM kernel.
        
        Args:
            enc: Metal compute encoder
            a_buf: Input A buffer (batch, M, K)
            b_buf: Input B buffer (batch, K, N)
            c_buf: Output C buffer (batch, M, N)
            batch: Batch size
            M, N, K: Matrix dimensions
            ngroups: Number of groups (for grouped attention)
            stride_a: Strides for A (batch, M, K)
            stride_b: Strides for B (batch, K, N)
            stride_c: Strides for C (batch, M, N)
        """
        # Default strides (row-major)
        if stride_a[0] is None:
            stride_a = (M * K, K, 1)
        if stride_b[0] is None:
            stride_b = (K * N, N, 1)
        if stride_c[0] is None:
            stride_c = (M * N, N, 1)
        
        enc.setComputePipelineState_(engine.pipelines["bmm_chunk_fwd"])
        
        # Set buffers
        enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(c_buf, 0, 2)
        # Buffer 3 (seq_idx) is optional, set to null for now
        
        # Set dimensions
        enc.setBytes_length_atIndex_(np.uint32(batch).tobytes(), 4, 4)
        enc.setBytes_length_atIndex_(np.uint32(M).tobytes(), 4, 5)
        enc.setBytes_length_atIndex_(np.uint32(N).tobytes(), 4, 6)
        enc.setBytes_length_atIndex_(np.uint32(K).tobytes(), 4, 7)
        enc.setBytes_length_atIndex_(np.uint32(ngroups).tobytes(), 4, 8)
        
        # Set strides
        enc.setBytes_length_atIndex_(np.uint32(stride_a[0]).tobytes(), 4, 9)
        enc.setBytes_length_atIndex_(np.uint32(stride_a[1]).tobytes(), 4, 10)
        enc.setBytes_length_atIndex_(np.uint32(stride_a[2]).tobytes(), 4, 11)
        enc.setBytes_length_atIndex_(np.uint32(stride_b[0]).tobytes(), 4, 12)
        enc.setBytes_length_atIndex_(np.uint32(stride_b[1]).tobytes(), 4, 13)
        enc.setBytes_length_atIndex_(np.uint32(stride_b[2]).tobytes(), 4, 14)
        enc.setBytes_length_atIndex_(np.uint32(stride_c[0]).tobytes(), 4, 15)
        enc.setBytes_length_atIndex_(np.uint32(stride_c[1]).tobytes(), 4, 16)
        enc.setBytes_length_atIndex_(np.uint32(stride_c[2]).tobytes(), 4, 17)
        
        # Calculate grid dimensions
        # Each threadgroup handles a BLOCK_M x BLOCK_N tile
        BLOCK_M = 64
        BLOCK_N = 64
        
        num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
        num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
        
        grid_size = Metal.MTLSizeMake(num_tiles_m, num_tiles_n, batch)
        group_size = Metal.MTLSizeMake(8, 8, 1)  # 64 threads: 8x8
        
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, group_size)
    
    # Bind method to engine
    import types
    engine._dispatch_bmm_chunk = types.MethodType(_dispatch_bmm_chunk, engine)
    
    print("[BMM Integration] Tiled BMM kernel integrated successfully")


def benchmark_bmm_kernel(engine, size=1024):
    """
    Benchmark the tiled BMM kernel vs naive implementation.
    
    Args:
        engine: MambaEngine with BMM integrated
        size: Matrix size for benchmarking
    """
    import time
    from metal_ops.metal_utils import allocate_metal_buffer, metal_buffer_to_numpy
    
    batch = 1
    M = N = K = size
    
    # Allocate buffers
    a_buf = allocate_metal_buffer(engine.ctx, batch * M * K)
    b_buf = allocate_metal_buffer(engine.ctx, batch * K * N)
    c_buf = allocate_metal_buffer(engine.ctx, batch * M * N)
    
    # Fill with random data
    import torch
    a_data = torch.randn(batch, M, K, dtype=torch.float32).numpy()
    b_data = torch.randn(batch, K, N, dtype=torch.float32).numpy()
    
    from metal_ops.metal_utils import numpy_to_metal_buffer
    a_buf = numpy_to_metal_buffer(engine.ctx, a_data)
    b_buf = numpy_to_metal_buffer(engine.ctx, b_data)
    
    # Warmup
    cmd = engine.ctx.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    engine._dispatch_bmm_chunk(enc, a_buf, b_buf, c_buf, batch, M, N, K)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    
    # Benchmark
    num_iters = 10
    start = time.perf_counter()
    for _ in range(num_iters):
        cmd = engine.ctx.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        engine._dispatch_bmm_chunk(enc, a_buf, b_buf, c_buf, batch, M, N, K)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    end = time.perf_counter()
    
    avg_time = (end - start) / num_iters * 1000  # ms
    flops = 2 * M * N * K  # FMA operations
    gflops = flops / (avg_time / 1000) / 1e9
    
    print(f"[BMM Benchmark] Size: {M}x{N}x{K}")
    print(f"  Avg time: {avg_time:.2f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    
    # Verify correctness
    c_result = metal_buffer_to_numpy(c_buf, (batch, M, N))
    c_expected = a_data @ b_data
    max_err = np.abs(c_result - c_expected).max()
    print(f"  Max error vs CPU: {max_err:.6f}")
    
    return avg_time, gflops, max_err


# Example usage in MambaEngine.__init__:
"""
# After compiling other kernels:
from metal_ops.mamba_bmm_wrapper import integrate_bmm_kernel
integrate_bmm_kernel(self)

# Now you can use self._dispatch_bmm_chunk() in step()
"""
