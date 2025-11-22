#include <metal_stdlib>
using namespace metal;

#define CH_PER_THREAD 4
#define T_TILE 128
#define UNROLL 8

kernel void mamba_s6_kernel_fp32_vec(
    device const float*      u        [[ buffer(0) ]],
    device const float*      delta    [[ buffer(1) ]],
    device const float*      A_param  [[ buffer(2) ]],
    device const float*      B_param  [[ buffer(3) ]],
    device const float*      C_param  [[ buffer(4) ]],
    device const float*      D_param  [[ buffer(5) ]],
    device float*            y        [[ buffer(6) ]],
    device const uint*       params   [[ buffer(7) ]],
    uint3                    gid      [[ thread_position_in_grid ]],
    uint3                    tid      [[ thread_position_in_threadgroup ]],
    uint3                    tgid     [[ threadgroup_position_in_grid ]]
)
{
    const uint B = params[0];
    const uint L = params[1];
    const uint D = params[2];
    // N not used in demo but kept
    // const uint N = params[3];

    const uint ch_tile_idx = gid.x;  // each gid.x corresponds to a *vector tile* of channels
    const uint b = gid.y;
    // Compute base channel index handled by this thread
    const uint d_base = ch_tile_idx * CH_PER_THREAD;

    // bounds
    if (b >= B || d_base >= D) return;

    // time tiling: tgid.x enumerates tiles in time (host should set grid accordingly)
    const uint time_tile_idx = tgid.x;
    const uint tile_start = time_tile_idx * T_TILE;
    const uint tile_end = min(tile_start + T_TILE, L);

    // private accumulators: one per channel processed by this thread
    float state[CH_PER_THREAD];
    #pragma unroll
    for (uint c = 0; c < CH_PER_THREAD; ++c) state[c] = 0.0f;

    // A simple per-channel scale (demo)
    const float IN_SCALE = 0.08f;
    const float DECAY = 0.005f;

    // Process tile in UNROLL blocks for speed
    for (uint tt = tile_start; tt < tile_end; tt += UNROLL) {
        #pragma unroll
        for (uint uo = 0; uo < UNROLL; ++uo) {
            uint t = tt + uo;
            if (t >= tile_end) break;

            // compute base index for (b,t, d_base)
            // idx0 corresponds to channel d_base
            uint base_idx = ((b * L) + t) * D + d_base;

            // vectorized loads: load CH_PER_THREAD consecutive floats
            // careful if d_base + c >= D -> handle boundary by checking
            float in_vals[CH_PER_THREAD];
            float delta_vals[CH_PER_THREAD];

            #pragma unroll
            for (uint c = 0; c < CH_PER_THREAD; ++c) {
                uint d_idx = d_base + c;
                if (d_idx < D) {
                    uint idx = base_idx + c; // contiguous in channel dimension
                    in_vals[c] = u[idx] * IN_SCALE;
                    delta_vals[c] = delta[idx] * IN_SCALE;
                } else {
                    in_vals[c] = 0.0f;
                    delta_vals[c] = 0.0f;
                }
            }

            // perform update for each channel
            #pragma unroll
            for (uint c = 0; c < CH_PER_THREAD; ++c) {
                // example recurrence
                state[c] = state[c] * (1.0f - DECAY) + in_vals[c] + 0.1f * delta_vals[c];

                // clamp guard
                if (!isfinite(state[c])) state[c] = 0.0f;
                const float SAFE = 1e6f;
                if (state[c] > SAFE) state[c] = SAFE;
                if (state[c] < -SAFE) state[c] = -SAFE;

                // write back
                uint d_idx = d_base + c;
                if (d_idx < D) {
                    uint out_idx = base_idx + c;
                    y[out_idx] = state[c];
                }
            }
        }
    }
}
