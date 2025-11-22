#include <metal_stdlib>
using namespace metal;


// 輔助函數：Softplus
// 對應 Triton: tl.where(dt <= 20.0, softplus(dt), dt)
inline float custom_softplus(float dt) {
    if (dt <= 20.0f) {
        return log(1.0f + exp(dt));
    }
    return dt;
}

// 核心 Kernel
kernel void selective_scan_update_kernel(
    device float* state             [[ buffer(0) ]], // (Batch, NHeads, Dim, DState)
    device const float* x           [[ buffer(1) ]], // (Batch, NHeads, Dim)
    device const float* dt          [[ buffer(2) ]], // (Batch, NHeads, Dim)
    device const float* dt_bias     [[ buffer(3) ]], // (NHeads, Dim)
    device const float* A           [[ buffer(4) ]], // (NHeads, Dim, DState)
    device const float* B           [[ buffer(5) ]], // (Batch, NGroups, DState)
    device const float* C           [[ buffer(6) ]], // (Batch, NGroups, DState)
    device const float* D           [[ buffer(7) ]], // (NHeads, Dim)
    device const float* z           [[ buffer(8) ]], // (Batch, NHeads, Dim)
    device float* out               [[ buffer(9) ]], // (Batch, NHeads, Dim)
    
    // 常數參數
    constant uint& batch_size       [[ buffer(10) ]],
    constant uint& nheads           [[ buffer(11) ]],
    constant uint& dim              [[ buffer(12) ]],
    constant uint& dstate           [[ buffer(13) ]],
    constant uint& n_groups         [[ buffer(14) ]],
    constant bool& has_z            [[ buffer(15) ]],
    constant bool& has_dt_bias      [[ buffer(16) ]],
    constant bool& has_D            [[ buffer(17) ]],
    
    // Grid 索引
    uint3 gid [[ thread_position_in_grid ]]
) {
    // 1. 計算索引
    // Grid 分配: x=Dim, y=Heads, z=Batch
    uint dim_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_idx = gid.z;

    if (dim_idx >= dim || head_idx >= nheads || batch_idx >= batch_size) {
        return;
    }

    // 計算各個 Tensor 的 offset (假設是 Contiguous Memory，簡化 Stride 計算)
    // 若需支援非連續記憶體，需傳入 stride 參數
    uint ratio = nheads / n_groups;
    uint group_idx = head_idx / ratio;

    // Offset 計算
    long global_idx_bh_d = batch_idx * (nheads * dim) + head_idx * dim + dim_idx; // for x, dt, z, out
    long state_base_offset = batch_idx * (nheads * dim * dstate) + head_idx * (dim * dstate) + dim_idx * dstate;
    long A_base_offset = head_idx * (dim * dstate) + dim_idx * dstate;
    long B_base_offset = batch_idx * (n_groups * dstate) + group_idx * dstate; // B is (Batch, Groups, DState)
    long C_base_offset = batch_idx * (n_groups * dstate) + group_idx * dstate;
    
    // 2. 載入數據 (Scalar)
    float val_x = x[global_idx_bh_d];
    float val_dt = dt[global_idx_bh_d];
    
    // 處理 DT Bias
    if (has_dt_bias) {
        long dt_bias_offset = head_idx * dim + dim_idx;
        val_dt += dt_bias[dt_bias_offset];
    }
    
    // Softplus
    val_dt = custom_softplus(val_dt);

    // 3. 狀態更新迴圈 (處理 DState 維度)
    // 對應 Triton: state * dA + dB * x
    // dA = exp(A * dt)
    // dB = B * dt
    
    float out_acc = 0.0f;
    
    for (uint i = 0; i < dstate; i++) {
        float val_A = A[A_base_offset + i];
        float val_B = B[B_base_offset + i];
        float val_C = C[C_base_offset + i];
        float val_state = state[state_base_offset + i];

        // 計算參數
        float dA = exp(val_A * val_dt);
        float dB = val_B * val_dt;

        // 更新 State
        float new_state = val_state * dA + dB * val_x;
        
        // 寫回 State
        state[state_base_offset + i] = new_state;

        // 累加輸出 (State * C)
        out_acc += new_state * val_C;
    }

    // 4. 加上 D residual
    if (has_D) {
        long D_offset = head_idx * dim + dim_idx;
        out_acc += val_x * D[D_offset];
    }

    // 5. Z Gate (SiLU)
    if (has_z) {
        float val_z = z[global_idx_bh_d];
        // SiLU = z * sigmoid(z) = z / (1 + exp(-z))
        float silu_z = val_z / (1.0f + exp(-val_z));
        out_acc *= silu_z;
    }

    // 6. 寫入結果
    out[global_idx_bh_d] = out_acc;
}

// --- 新增 1: RMSNorm ---
kernel void rms_norm_kernel(
    device const float* input [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* output [[ buffer(2) ]],
    constant uint& B [[ buffer(3) ]],
    constant uint& D [[ buffer(4) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    // Grid: (Batch, 1, 1) -> 這裡簡單處理，每個 Thread 處理一個 Batch row (或優化為 Block reduce)
    // 為了演示簡單起見，我們假設 gid.x 是 Batch index
    // *注意: 實際生產級 RMSNorm 通常需要 Threadgroup memory 進行 reduce
    
    uint batch_idx = gid.x;
    if (batch_idx >= B) return;
    
    float sum_sq = 0.0;
    for (uint i = 0; i < D; i++) {
        float val = input[batch_idx * D + i];
        sum_sq += val * val;
    }
    
    float rsqrt_val = rsqrt(sum_sq / float(D) + 1e-5); // epsilon
    
    for (uint i = 0; i < D; i++) {
        output[batch_idx * D + i] = input[batch_idx * D + i] * rsqrt_val * weight[i];
    }
}

// --- 新增 2: Causal Depthwise Conv1d (k=4) ---
// Mamba 的 Conv 是 (Batch, Dim, Length)，這裡簡化為 (Batch, Length, Dim) 配合你的 layout
kernel void depthwise_conv1d_kernel(
    device const float* input [[ buffer(0) ]],  // (B, L, D)
    device const float* weight [[ buffer(1) ]], // (D, K) -> (D, 4)
    device const float* bias [[ buffer(2) ]],   // (D)
    device float* output [[ buffer(3) ]],       // (B, L, D)
    constant uint& B [[ buffer(4) ]],
    constant uint& L [[ buffer(5) ]],
    constant uint& D [[ buffer(6) ]],
    constant uint& K [[ buffer(7) ]], // Kernel Size usually 4
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint d = gid.x; // Dim
    uint l = gid.y; // Length (Time)
    uint b = gid.z; // Batch
    
    if (b >= B || l >= L || d >= D) return;
    
    float res = bias[d];
    
    // Causal Convolution: 只看當前和過去
    for (uint k = 0; k < K; k++) {
        // 如果索引 < 0 (padding)，則為 0
        int time_idx = (int)l - (int)K + 1 + k; 
        
        if (time_idx >= 0) {
            // Input layout: (B, L, D) -> b * L * D + time_idx * D + d
            float in_val = input[b * L * D + time_idx * D + d];
            // Weight layout: (D, K) -> d * K + k
            float w_val = weight[d * K + k];
            res += in_val * w_val;
        }
    }
    
    // SiLU Activation (Mamba 標準做法: Conv 後接 SiLU)
    float sigmoid = 1.0 / (1.0 + exp(-res));
    float silu = res * sigmoid;
    
    output[b * L * D + l * D + d] = silu;
}

// --- 新增 3: Output Gating (y * silu(z)) ---
kernel void gating_kernel(
    device const float* y [[ buffer(0) ]], // SSM output
    device const float* z [[ buffer(1) ]], // Gate branch
    device float* output [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    float z_val = z[id];
    float sigmoid = 1.0 / (1.0 + exp(-z_val));
    float silu_z = z_val * sigmoid;
    
    output[id] = y[id] * silu_z;
}