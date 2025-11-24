"""
Enhanced Metal dispatch utilities with improved performance and safety.

Provides:
- dispatchThreads API for precise thread allocation
- Thread-safe execute_and_wait with callbacks
- Performance helpers
"""

import Metal
import threading
from typing import Tuple, Optional


def dispatch_threads(encoder: Metal.MTLComputeCommandEncoder,
                    grid_size: Tuple[int, int, int],
                    threadgroup_size: Tuple[int, int, int]):
    """
    Dispatch compute kernel using dispatchThreads (precise thread count).
    
    This is often faster than dispatchThreadgroups on M-series GPUs because:
    - No wasted threads if grid doesn't align to threadgroup boundaries
    - Hardware handles thread distribution optimally
    
    Args:
        encoder: Metal compute command encoder
        grid_size: Total threads (width, height, depth)
        threadgroup_size: Threads per threadgroup (width, height, depth)
    """
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid_size),
        Metal.MTLSizeMake(*threadgroup_size)
    )


def dispatch_threadgroups(encoder: Metal.MTLComputeCommandEncoder,
                         grid_size: Tuple[int, int, int],
                         threadgroup_size: Tuple[int, int, int]):
    """
    Dispatch compute kernel using dispatchThreadgroups (grid of threadgroups).
    
    Use this when:
    - Kernel uses threadgroup memory heavily
    - Grid size is already a multiple of threadgroup size
    
    Args:
        encoder: Metal compute command encoder
        grid_size: Number of threadgroups (width, height, depth)
        threadgroup_size: Threads per threadgroup (width, height, depth)
    """
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid_size),
        Metal.MTLSizeMake(*threadgroup_size)
    )


def execute_and_wait_threadsafe(cmd_buffer: Metal.MTLCommandBuffer, timeout: Optional[float] = None) -> bool:
    """
    Execute command buffer and wait for completion (thread-safe).
    
    This uses a completion handler instead of waitUntilCompleted(),
    which is more thread-friendly and doesn't block the main thread.
    
    Args:
        cmd_buffer: Metal command buffer to execute
        timeout: Max wait time in seconds (None = infinite)
        
    Returns:
        True if completed successfully, False if timeout
    """
    event = threading.Event()
    
    def completion_handler(_):
        event.set()
    
    cmd_buffer.addCompletedHandler_(completion_handler)
    cmd_buffer.commit()
    
    return event.wait(timeout=timeout)


def execute_and_wait_blocking(cmd_buffer: Metal.MTLCommandBuffer):
    """
    Execute command buffer and wait for completion (blocking).
    
    This is the simple blocking version. Use for:
    - Single-threaded inference
    - When you need guaranteed synchronization
    
    Args:
        cmd_buffer: Metal command buffer to execute
    """
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()


# Maintain backward compatibility
execute_and_wait = execute_and_wait_blocking


class MetalKernelTimer:
    """
    Helper to measure kernel execution time.
    
    Usage:
        timer = MetalKernelTimer()
        with timer:
            # dispatch kernels
            enc.endEncoding()
            execute_and_wait(cmd)
        print(f"Took {timer.elapsed_ms:.2f} ms")
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return False


def create_fence(device: Metal.MTLDevice) -> Metal.MTLFence:
    """
    Create a Metal fence for fine-grained synchronization.
    
    Use fences to:
    - Synchronize between different encoders
    - Ensure dependencies without full command buffer sync
    
    Args:
        device: Metal device
        
    Returns:
        Metal fence object
    """
    return device.newFence()
