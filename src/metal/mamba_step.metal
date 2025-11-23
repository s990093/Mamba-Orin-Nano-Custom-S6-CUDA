#include <metal_stdlib>
using namespace metal;

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast::exp(-x));
}

inline float softplus_custom(float x) {
    return x > 20.0f ? x : fast::log(1.0f + fast::exp(x));
}

kernel void mamba_step_kernel(
    device float* state       [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    device const float* dt    [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* A     [[buffer(4)]],
    device const float* B_param [[buffer(5)]],
    device const float* C_param [[buffer(6)]],
    device const float* D_param [[buffer(7)]],
    device const float* z     [[buffer(8)]],
    device float* out         [[buffer(9)]],

    constant uint4& s_state   [[buffer(10)]],
    constant uint4& s_x       [[buffer(11)]],
    constant uint4& s_dt      [[buffer(12)]],
    constant uint4& s_dt_bias [[buffer(13)]],
    constant uint4& s_A       [[buffer(14)]],
    constant uint4& s_B       [[buffer(15)]],
    constant uint4& s_C       [[buffer(16)]],
    constant uint4& s_D       [[buffer(17)]],
    constant uint4& s_z       [[buffer(18)]],
    constant uint4& s_out     [[buffer(19)]],

    constant uint& batch      [[buffer(20)]],
    constant uint& nheads     [[buffer(21)]],
    constant uint& dim        [[buffer(22)]],
    constant uint& dstate     [[buffer(23)]],
    constant uint& n_groups   [[buffer(24)]],

    constant bool& has_z      [[buffer(25)]],
    constant bool& has_D      [[buffer(26)]],
    constant bool& has_dt_bias[[buffer(27)]],

    uint3 gid [[thread_position_in_grid]],
    uint simd_idx [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]]
) {
    uint idx_d = gid.x;
    uint idx_h = gid.y;
    uint idx_b = gid.z;
    if (idx_d >= dim || idx_h >= nheads || idx_b >= batch) return;

    uint ratio = nheads / n_groups;
    uint idx_g = idx_h / ratio;

    float val_x = x[idx_b * s_x.x + idx_h * s_x.y + idx_d * s_x.z];

    float val_dt = dt[idx_b * s_dt.x + idx_h * s_dt.y + idx_d * s_dt.z];
    if (has_dt_bias) {
        val_dt += dt_bias[idx_h * s_dt_bias.x + idx_d * s_dt_bias.y];
    }
    val_dt = softplus_custom(val_dt);

    uint base_state = idx_b * s_state.x + idx_h * s_state.y + idx_d * s_state.z;
    uint base_A     = idx_h * s_A.x     + idx_d * s_A.y;
    uint base_B     = idx_b * s_B.x     + idx_g * s_B.y;
    uint base_C     = idx_b * s_C.x     + idx_g * s_C.y;

    // Each thread handles multiple n via SIMD
    float acc = 0.0f;
    for (uint n = simd_idx; n < dstate; n += simd_size) {
        float val_A = A[base_A + n * s_A.z];
        float val_B = B_param[base_B + n * s_B.z];
        float val_C = C_param[base_C + n * s_C.z];

        uint ptr_state = base_state + n * s_state.w;
        float prev_h = state[ptr_state];

        float dA = fast::exp(val_A * val_dt);
        float dB = val_B * val_dt;
        float new_h = prev_h * dA + val_x * dB;

        state[ptr_state] = new_h;
        acc += new_h * val_C;
    }

    // SIMD group reduction
    threadgroup float shared_acc[32]; // assume simdgroup <= 32
    shared_acc[simd_idx] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduction
    for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
        if (simd_idx < offset) {
            shared_acc[simd_idx] += shared_acc[simd_idx + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_idx == 0) {
        float out_acc = shared_acc[0];
        if (has_D) {
            out_acc += val_x * D_param[idx_h * s_D.x + idx_d * s_D.y];
        }
        if (has_z) {
            float val_z = z[idx_b * s_z.x + idx_h * s_z.y + idx_d * s_z.z];
            out_acc = out_acc * val_z * sigmoid(val_z);
        }
        out[idx_b * s_out.x + idx_h * s_out.y + idx_d * s_out.z] = out_acc;
    }
}
