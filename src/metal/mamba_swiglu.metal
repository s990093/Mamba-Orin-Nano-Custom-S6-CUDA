#include <metal_stdlib>
using namespace metal;

// ====================== SwiGLU Activation ======================
// Fused kernel for SwiGLU(x, y) = (x * sigmoid(x)) * y
// This is used in Mamba2's MLP/FFN layers for gating
//
// Forward: out = x * sigmoid(x) * y
// Backward: 
//   dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * y * dout
//   dy = x * sigmoid(x) * dout

// Fast sigmoid approximation
inline float fast_sigmoid_swiglu(float x) {
    float x_abs = fabs(x);
    if (x_abs > 5.0f) [[likely]] {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    float a = x * x;
    float b = x * (1.0f + a * (0.166666f + a * 0.008333f));
    return 0.5f + b / (1.0f + sqrt(1.0f + 4.0f * fabs(b)));
}

// ====================== Forward Kernel ======================
kernel void swiglu_fwd_kernel(
    device const float* x       [[buffer(0)]],  // (M, N)
    device const float* y       [[buffer(1)]],  // (M, N)
    device float* out           [[buffer(2)]],  // (M, N)
    constant uint& M            [[buffer(3)]],
    constant uint& N            [[buffer(4)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    uint idx = row * N + col;
    
    float x_val = x[idx];
    float y_val = y[idx];
    
    // SwiGLU: x * sigmoid(x) * y
    float sig_x = fast_sigmoid_swiglu(x_val);
    out[idx] = x_val * sig_x * y_val;
}

// ====================== Backward Kernel ======================
kernel void swiglu_bwd_kernel(
    device const float* x       [[buffer(0)]],  // (M, N)
    device const float* y       [[buffer(1)]],  // (M, N)
    device const float* dout    [[buffer(2)]],  // (M, N)
    device float* dx            [[buffer(3)]],  // (M, N)
    device float* dy            [[buffer(4)]],  // (M, N)
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    uint idx = row * N + col;
    
    float x_val = x[idx];
    float y_val = y[idx];
    float dout_val = dout[idx];
    
    float sig = fast_sigmoid_swiglu(x_val);
    
    // dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * y * dout
    dx[idx] = sig * (1.0f + x_val * (1.0f - sig)) * y_val * dout_val;
    
    // dy = x * sigmoid(x) * dout
    dy[idx] = x_val * sig * dout_val;
}

// ====================== Vectorized Forward (Block-based) ======================
// Process multiple elements per thread for better memory bandwidth
constant uint BLOCK_SIZE [[function_constant(0)]];

kernel void swiglu_fwd_blocked_kernel(
    device const float* x       [[buffer(0)]],
    device const float* y       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant uint& total_elems  [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    uint start_idx = tid * BLOCK_SIZE;
    
    for (uint i = 0; i < BLOCK_SIZE; ++i) {
        uint idx = start_idx + i;
        if (idx >= total_elems) break;
        
        float x_val = x[idx];
        float y_val = y[idx];
        float sig_x = fast_sigmoid_swiglu(x_val);
        out[idx] = x_val * sig_x * y_val;
    }
}

// ====================== SiLU variant (for reference) ======================
// SiLU(x) = x * sigmoid(x), used in some Mamba configurations
kernel void silu_kernel(
    device const float* x       [[buffer(0)]],
    device float* out           [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    
    float x_val = x[tid];
    float sig_x = fast_sigmoid_swiglu(x_val);
    out[tid] = x_val * sig_x;
}

// ====================== GELU variant (for reference) ======================
// GELU(x) = x * Φ(x), where Φ is Gaussian CDF
kernel void gelu_kernel(
    device const float* x       [[buffer(0)]],
    device float* out           [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    
    float x_val = x[tid];
    
    // Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    float x3 = x_val * x_val * x_val;
    float inner = 0.7978845608f * (x_val + 0.044715f * x3);
    out[tid] = 0.5f * x_val * (1.0f + tanh(inner));
}
