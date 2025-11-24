"""
Metal utilities for Mamba operations on macOS.

Provides shared utilities for Metal device management, buffer allocation,
and kernel compilation.
"""

from __future__ import annotations

import os
import numpy as np
import torch
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
            
        # Check for cached .metallib
        metallib_path = abs_path.replace(".metal", ".metallib")
        lock_path = metallib_path + ".lock"
        should_compile = True
        
        # Use file lock to prevent race conditions in parallel compilation
        import fcntl
        import tempfile
        
        # Create lock file
        lock_fd = None
        try:
            # Open lock file (create if doesn't exist)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            
            # Acquire exclusive lock (blocks if another process is compiling)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            # Re-check cache after acquiring lock (another process might have compiled it)
            if os.path.exists(metallib_path):
                src_mtime = os.path.getmtime(abs_path)
                lib_mtime = os.path.getmtime(metallib_path)
                if lib_mtime > src_mtime:
                    should_compile = False
                    
            if should_compile:
                print(f"[MetalContext] Compiling {os.path.basename(source_path)}...")
                # Use xcrun to compile to .metallib
                # 1. Compile to .air
                air_path = abs_path.replace(".metal", ".air")
                ret = os.system(f"xcrun -sdk macosx metal -c {abs_path} -o {air_path} 2>&1")
                if ret != 0:
                    raise RuntimeError("Failed to compile Metal source to AIR")
                    
                # 2. Link to .metallib
                ret = os.system(f"xcrun -sdk macosx metallib {air_path} -o {metallib_path} 2>&1")
                if ret != 0:
                    raise RuntimeError("Failed to link AIR to metallib")
                    
                # Cleanup .air
                if os.path.exists(air_path):
                    os.remove(air_path)
            else:
                print(f"[MetalContext] Loading cached {os.path.basename(metallib_path)}...")
                
        finally:
            # Release lock
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            
        # Load the library
        library, err = self.device.newLibraryWithFile_error_(metallib_path, None)
        
        if err:
            # Fallback to source compilation if loading failed
            print(f"[MetalContext] Failed to load metallib: {err.localizedDescription()}. Falling back to source.")
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


def metal_buffer_to_numpy(buffer: Metal.MTLBuffer, shape: Tuple[int, ...], dtype=np.float32, copy=True, _unsafe_no_sync=False) -> np.ndarray:
    """
    Convert Metal buffer to NumPy array.
    
    ⚠️ CRITICAL SAFETY NOTE:
    When copy=False (zero-copy view), the returned array is a DIRECT VIEW into Metal shared memory.
    This is ONLY SAFE if:
    1. The command buffer that wrote to this buffer has COMPLETED (called waitUntilCompleted)
    2. No kernel is currently writing to this buffer
    3. You will not modify the array (read-only)
    
    Violating these conditions causes:
    - Reading dirty/incomplete data
    - Race conditions
    - Crashes or NaN values
    
    Args:
        buffer: Metal buffer
        shape: Desired output shape
        dtype: Data type
        copy: Whether to copy the data (SAFE) or return a view (FAST but DANGEROUS)
        _unsafe_no_sync: Internal flag to skip sync check (use with extreme caution)
        
    Returns:
        NumPy array
    """
    if not copy and not _unsafe_no_sync:
        import warnings
        warnings.warn(
            "metal_buffer_to_numpy(copy=False) returns UNSAFE zero-copy view. "
            "Ensure command buffer has completed before reading! "
            "Set _unsafe_no_sync=True to suppress this warning.",
            RuntimeWarning,
            stacklevel=2
        )
    
    ptr = buffer.contents()
    arr = np.frombuffer(ptr.as_buffer(buffer.length()), dtype=dtype)
    arr = arr.reshape(shape)
    
    if copy:
        return arr.copy()
    return arr


def allocate_metal_buffer(ctx: MetalContext, num_bytes: int, options=None, alignment=256) -> Metal.MTLBuffer:
    """
    Allocate a Metal buffer with proper alignment.
    
    Args:
        ctx: Metal context
        num_bytes: Number of bytes to allocate
        options: MTLResourceStorageMode (default: Shared)
        alignment: Byte alignment (default: 256 for optimal Apple GPU performance)
        
    Returns:
        Metal buffer
        
    Note:
        For mixed-precision (FP16/FP32/INT8), caller should compute num_bytes
        based on actual dtype size, not assume float32.
    """
    if options is None:
        options = Metal.MTLResourceStorageModeShared
    
    # Align to specified boundary for optimal GPU memory access
    aligned_size = ((num_bytes + alignment - 1) // alignment) * alignment
    
    buffer = ctx.device.newBufferWithLength_options_(aligned_size, options)
    if buffer is None:
        raise RuntimeError(f"Failed to allocate {aligned_size} bytes of Metal buffer")
    
    return buffer


def allocate_metal_buffer_for_array(ctx: MetalContext, num_elements: int, dtype=np.float32, options=None) -> Metal.MTLBuffer:
    """
    Allocate a Metal buffer sized for an array of specific dtype.
    
    Args:
        ctx: Metal context
        num_elements: Number of elements
        dtype: NumPy dtype (e.g., np.float32, np.float16, np.int8)
        options: MTLResourceStorageMode
        
    Returns:
        Metal buffer
    """
    element_size = np.dtype(dtype).itemsize
    num_bytes = num_elements * element_size
    return allocate_metal_buffer(ctx, num_bytes, options)


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


def torch_tensor_to_metal_buffer(ctx: MetalContext, tensor: torch.Tensor) -> Metal.MTLBuffer:
    """
    Efficiently copy a PyTorch tensor to a Metal buffer.
    
    Args:
        ctx: Metal context
        tensor: PyTorch tensor (will be moved to CPU and made contiguous)
        
    Returns:
        Metal buffer containing the tensor data
    """
    # Ensure tensor is on CPU and contiguous float32
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        
    # Get underlying numpy array without copy if possible
    # Note: tensor.numpy() shares memory if on CPU
    arr = tensor.detach().cpu().numpy()
    
    return ctx.device.newBufferWithBytes_length_options_(
        arr.tobytes(), # This might still copy, but it's safer for now
        arr.nbytes,
        Metal.MTLResourceStorageModeShared
    )
