"""
Metal utilities for Mamba operations on macOS.

Provides shared utilities for Metal device management, buffer allocation,
and kernel compilation.
"""

from __future__ import annotations

import os
import numpy as np
import Metal
from typing import Optional, Tuple


class MetalContext:
    """Manages Metal device and kernel compilation."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal is not available on this system")
            
        self.queue = self.device.newCommandQueue()
        self._libraries = {}
        self._pipelines = {}
        self._initialized = True
        
        print(f"[MetalContext] Initialized with device: {self.device.name()}")
    
    def compile_library(self, source_path: str, key: Optional[str] = None) -> Metal.MTLLibrary:
        """
        Compile a Metal shader library from source file.
        
        Args:
            source_path: Path to .metal source file
            key: Optional cache key (defaults to source_path)
            
        Returns:
            Compiled Metal library
        """
        if key is None:
            key = source_path
            
        if key in self._libraries:
            return self._libraries[key]
        
        abs_path = os.path.abspath(source_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Metal source file not found: {abs_path}")
            
        with open(abs_path, 'r') as f:
            source = f.read()
        
        opts = Metal.MTLCompileOptions.new()
        library, err = self.device.newLibraryWithSource_options_error_(source, opts, None)
        
        if err:
            raise RuntimeError(f"Metal compilation error: {err.localizedDescription()}")
        
        self._libraries[key] = library
        return library
    
    def get_pipeline(self, library: Metal.MTLLibrary, kernel_name: str) -> Metal.MTLComputePipelineState:
        """
        Get or create a compute pipeline for a kernel function.
        
        Args:
            library: Compiled Metal library
            kernel_name: Name of kernel function
            
        Returns:
            Compute pipeline state
        """
        cache_key = (id(library), kernel_name)
        
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]
        
        fn = library.newFunctionWithName_(kernel_name)
        if fn is None:
            raise RuntimeError(f"Kernel function '{kernel_name}' not found in library")
        
        pipe, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(f"Pipeline creation error: {err.localizedDescription()}")
        
        self._pipelines[cache_key] = pipe
        return pipe


def numpy_to_metal_buffer(ctx: MetalContext, arr: np.ndarray) -> Metal.MTLBuffer:
    """
    Convert NumPy array to Metal buffer.
    
    Args:
        ctx: Metal context
        arr: NumPy array (will be converted to float32 and made contiguous)
        
    Returns:
        Metal buffer in shared memory
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return ctx.device.newBufferWithBytes_length_options_(
        arr.tobytes(),
        arr.nbytes,
        Metal.MTLResourceStorageModeShared
    )


def metal_buffer_to_numpy(buffer: Metal.MTLBuffer, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Convert Metal buffer to NumPy array.
    
    Args:
        buffer: Metal buffer
        shape: Desired output shape
        
    Returns:
        NumPy array (copy of buffer data)
    """
    ptr = buffer.contents()
    arr = np.frombuffer(ptr.as_buffer(buffer.length()), dtype=np.float32)
    return arr.reshape(shape).copy()


def allocate_metal_buffer(ctx: MetalContext, num_elements: int) -> Metal.MTLBuffer:
    """
    Allocate a Metal buffer for float32 data.
    
    Args:
        ctx: Metal context
        num_elements: Number of float32 elements
        
    Returns:
        Allocated Metal buffer in shared memory
    """
    return ctx.device.newBufferWithLength_options_(
        num_elements * 4,  # 4 bytes per float32
        Metal.MTLResourceStorageModeShared
    )


def set_buffer_data(buffer: Metal.MTLBuffer, data: np.ndarray):
    """
    Copy NumPy array data into existing Metal buffer.
    
    Args:
        buffer: Metal buffer
        data: NumPy array (will be flattened and converted to float32)
    """
    data = np.ascontiguousarray(data.flatten(), dtype=np.float32)
    ptr = buffer.contents()
    np_view = np.frombuffer(ptr.as_buffer(buffer.length()), dtype=np.float32)
    np_view[:data.size] = data


def dispatch_kernel(
    encoder: Metal.MTLComputeCommandEncoder,
    pipeline: Metal.MTLComputePipelineState,
    grid_size: Tuple[int, int, int],
    threadgroup_size: Tuple[int, int, int]
):
    """
    Dispatch a compute kernel with specified grid and threadgroup sizes.
    
    Args:
        encoder: Compute command encoder
        pipeline: Pipeline state to use
        grid_size: (width, height, depth) of grid
        threadgroup_size: (width, height, depth) of threadgroup
    """
    encoder.setComputePipelineState_(pipeline)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid_size),
        Metal.MTLSizeMake(*threadgroup_size)
    )


def create_command_buffer(ctx: MetalContext) -> Metal.MTLCommandBuffer:
    """Create a new command buffer from the context's queue."""
    return ctx.queue.commandBuffer()


def execute_and_wait(cmd_buffer: Metal.MTLCommandBuffer):
    """Execute command buffer and wait for completion."""
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
