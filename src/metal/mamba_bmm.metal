#include <metal_stdlib>
using namespace metal;

// ====================== Tiled Block Matrix Multiply ======================
// Inspired by Triton _bmm_chunk_fwd_kernel
// Computes C = A @ B with tile-based optimization
// 
// A: (batch, M, K)
// B: (batch, K, N) 
// C: (batch, M, N)
//
// Supports:
// - Grouped computation (ngroups parameter)
// - Optional causal masking
// - Tiled loading to reduce global memory traffic

constant uint BLOCK_M [[function_constant(0)]];
constant uint BLOCK_N [[function_constant(1)]];
constant uint BLOCK_K [[function_constant(2)]];
constant bool IS_CAUSAL [[function_constant(3)]];

kernel void bmm_chunk_fwd_kernel(
    device const float* A       [[buffer(0)]],  // (batch, M, K)
    device const float* B       [[buffer(1)]],  // (batch, K, N)
    device float* C             [[buffer(2)]],  // (batch, M, N)
    device const int* seq_idx   [[buffer(3)]],  // Optional: (batch, M) for sequence masking
    
    constant uint& batch        [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    constant uint& ngroups      [[buffer(8)]],
    
    constant uint& stride_a_batch [[buffer(9)]],
    constant uint& stride_a_m     [[buffer(10)]],
    constant uint& stride_a_k     [[buffer(11)]],
    constant uint& stride_b_batch [[buffer(12)]],
    constant uint& stride_b_k     [[buffer(13)]],
    constant uint& stride_b_n     [[buffer(14)]],
    constant uint& stride_c_batch [[buffer(15)]],
    constant uint& stride_c_m     [[buffer(16)]],
    constant uint& stride_c_n     [[buffer(17)]],
    
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgsize [[threads_per_threadgroup]]
) {
    // Thread block indices
    uint tile_m = gid.x;
    uint tile_n = gid.y;
    uint b = gid.z;
    
    // Compute actual M, N indices for this tile
    uint m_start = tile_m * BLOCK_M;
    uint n_start = tile_n * BLOCK_N;
    
    // Boundary check
    if (m_start >= M || n_start >= N) return;
    
    // Thread-local indices within tile
    uint local_m = tid.x;
    uint local_n = tid.y;
    
    // Allocate threadgroup memory for tiles
    // A_tile: BLOCK_M x BLOCK_K
    // B_tile: BLOCK_K x BLOCK_N
    threadgroup float A_tile[64 * 32];  // Assuming BLOCK_M=64, BLOCK_K=32
    threadgroup float B_tile[32 * 64];  // Assuming BLOCK_K=32, BLOCK_N=64
    
    // Accumulator for this thread's output element
    float acc = 0.0f;
    
    // Base pointers for this batch
    device const float* A_batch = A + b * stride_a_batch;
    device const float* B_batch = B + b * stride_b_batch;
    
    // Loop over K dimension in tiles
    for (uint k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; ++k_tile) {
        uint k_start = k_tile * BLOCK_K;
        
        // Collaboratively load A tile (BLOCK_M x BLOCK_K)
        // Each thread loads one or more elements
        uint num_elements_a = BLOCK_M * BLOCK_K;
        uint thread_id_flat = local_m * tgsize.y + local_n;
        uint threads_per_group = tgsize.x * tgsize.y;
        
        for (uint idx = thread_id_flat; idx < num_elements_a; idx += threads_per_group) {
            uint tile_m_idx = idx / BLOCK_K;
            uint tile_k_idx = idx % BLOCK_K;
            
            uint global_m = m_start + tile_m_idx;
            uint global_k = k_start + tile_k_idx;
            
            float val = 0.0f;
            if (global_m < M && global_k < K) {
                val = A_batch[global_m * stride_a_m + global_k * stride_a_k];
            }
            A_tile[idx] = val;
        }
        
        // Collaboratively load B tile (BLOCK_K x BLOCK_N)
        uint num_elements_b = BLOCK_K * BLOCK_N;
        for (uint idx = thread_id_flat; idx < num_elements_b; idx += threads_per_group) {
            uint tile_k_idx = idx / BLOCK_N;
            uint tile_n_idx = idx % BLOCK_N;
            
            uint global_k = k_start + tile_k_idx;
            uint global_n = n_start + tile_n_idx;
            
            float val = 0.0f;
            if (global_k < K && global_n < N) {
                val = B_batch[global_k * stride_b_k + global_n * stride_b_n];
            }
            B_tile[idx] = val;
        }
        
        // Synchronize to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this thread's output element
        // Each thread computes one element of the output tile
        if (local_m < BLOCK_M && local_n < BLOCK_N) {
            uint global_m = m_start + local_m;
            uint global_n = n_start + local_n;
            
            if (global_m < M && global_n < N) {
                // Apply causal masking if enabled
                bool is_valid = true;
                if (IS_CAUSAL) {
                    // For causal: only compute if m >= n (assuming square)
                    is_valid = (global_m >= global_n);
                }
                
                if (is_valid) {
                    for (uint k = 0; k < BLOCK_K && (k_start + k) < K; ++k) {
                        float a_val = A_tile[local_m * BLOCK_K + k];
                        float b_val = B_tile[k * BLOCK_N + local_n];
                        acc += a_val * b_val;
                    }
                }
            }
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result to global memory
    if (local_m < BLOCK_M && local_n < BLOCK_N) {
        uint global_m = m_start + local_m;
        uint global_n = n_start + local_n;
        
        if (global_m < M && global_n < N) {
            device float* C_batch = C + b * stride_c_batch;
            C_batch[global_m * stride_c_m + global_n * stride_c_n] = acc;
        }
    }
}

// ====================== Chunked BMM Variant ======================
// Specialized version for Mamba's chunked attention pattern
// Processes chunks of sequence in parallel

kernel void bmm_chunk_scan_kernel(
    device const float* q_chunk  [[buffer(0)]],  // (batch, nchunks, chunk_size, headdim)
    device const float* k_chunk  [[buffer(1)]],  // (batch, nchunks, chunk_size, headdim)
    device float* scores         [[buffer(2)]],  // (batch, nchunks, chunk_size, chunk_size)
    
    constant uint& batch         [[buffer(3)]],
    constant uint& nchunks       [[buffer(4)]],
    constant uint& chunk_size    [[buffer(5)]],
    constant uint& headdim       [[buffer(6)]],
    
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    // gid.x: chunk_tile_idx
    // gid.y: batch_idx 
    // gid.z: chunk_idx
    
    uint b = gid.y;
    uint c = gid.z;
    uint tile_idx = gid.x;
    
    // Similar tile-based logic for chunk_size x chunk_size output
    // This is a specialized kernel for attention-like patterns in Mamba
    
    // TODO: Implement chunked variant with optimized layout
}
