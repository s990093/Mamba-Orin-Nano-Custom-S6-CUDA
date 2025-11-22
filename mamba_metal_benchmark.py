# -*- coding: utf-8 -*-
import Metal
import Foundation
import objc
import numpy as np
import time
import os
import sys
import argparse

# === Metal 初始化 ===
def setup_metal():
    device = Metal.MTLCreateSystemDefaultDevice()  # type: ignore
    if device is None:
        print("錯誤: 找不到 Metal 支援的設備")
        sys.exit(1)
    print(f"使用設備: {device.name()}")
    return device

def compile_library(device, path):
    if not os.path.exists(path):
        print(f"錯誤: 找不到 shader 檔案 {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        source = f.read()
    options = Metal.MTLCompileOptions.alloc().init()
    lib, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err:
        print("Shader 編譯失敗:", err)
        sys.exit(1)
    return lib

def new_pipeline(device, library, func_name):
    func = library.newFunctionWithName_(func_name)
    if func is None:
        print(f"找不到函數: {func_name}")
        return None
    pso, err = device.newComputePipelineStateWithFunction_error_(func, None)
    if err:
        print(f"建立 PSO 失敗 ({func_name}):", err)
        return None
    return pso

def create_buffer(device, array, storage_mode=Metal.MTLResourceStorageModeShared):
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    data = array.tobytes()
    return device.newBufferWithBytes_length_options_(data, len(data), storage_mode)

def read_buffer_to_numpy(buf, dtype, shape):
    size = buf.length()
    pybuf = buf.contents().as_buffer(size)
    arr = np.frombuffer(pybuf, dtype=dtype).copy()
    return arr.reshape(shape)

def run_once(device, cmd_queue, pso, buffers, grid_size, threads_per_group, iterations=1):
    for _ in range(iterations):
        cmd_buffer = cmd_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pso)
        for idx, b in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(b, 0, idx)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threads_per_group)
        encoder.endEncoding()
        cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
    return

def detect_nan_inf(np_array):
    arr = np_array.astype(np.float32)
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    return int(nan_count), int(inf_count)

# === 主流程 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16","fp32"], default="fp16", help="Run mode: fp16 or fp32")
    parser.add_argument("--iters", type=int, default=100, help="benchmark iterations")
    args = parser.parse_args()

    B, L, D, N = 1, 1024, 768, 16
    print(f"配置: B={B}, L={L}, D={D}, N={N}, mode={args.mode}")

    device = setup_metal()
    cmd_queue = device.newCommandQueue()
    lib = compile_library(device, "src/metal/mamba_s6.metal")

    pso_fp16 = new_pipeline(device, lib, "mamba_s6_kernel_fp16")
    pso_fp32 = new_pipeline(device, lib, "mamba_s6_kernel_fp32_tile")
    if pso_fp16 is None or pso_fp32 is None:
        print("至少一個 kernel compile 失敗，停止。")
        sys.exit(1)

    rng = np.random.RandomState(42)
    u_f32 = (rng.randn(B,L,D)).astype(np.float32) * 0.5
    delta_f32 = (rng.randn(B,L,D)).astype(np.float32) * 0.05
    A_f32 = (rng.randn(D,N)).astype(np.float32)
    Bp_f32 = (rng.randn(B,L,N)).astype(np.float32)
    Cp_f32 = (rng.randn(B,L,N)).astype(np.float32)
    D_f32 = (rng.randn(D)).astype(np.float32)
    y_f32 = np.zeros((B,L,D), dtype=np.float32)

    # FP16
    u_f16, delta_f16, A_f16, Bp_f16, Cp_f16, D_f16, y_f16 = [x.astype(np.float16) for x in [u_f32, delta_f32, A_f32, Bp_f32, Cp_f32, D_f32, y_f32]]

    params = np.array([np.uint32(B), np.uint32(L), np.uint32(D), np.uint32(N)], dtype=np.uint32)
    buf_params = create_buffer(device, params)

    # buffers
    buf_u_f16 = create_buffer(device, u_f16)
    buf_delta_f16 = create_buffer(device, delta_f16)
    buf_A_f16 = create_buffer(device, A_f16)
    buf_Bp_f16 = create_buffer(device, Bp_f16)
    buf_Cp_f16 = create_buffer(device, Cp_f16)
    buf_D_f16 = create_buffer(device, D_f16)
    buf_y_f16 = create_buffer(device, y_f16)

    buf_u_f32 = create_buffer(device, u_f32)
    buf_delta_f32 = create_buffer(device, delta_f32)
    buf_A_f32 = create_buffer(device, A_f32)
    buf_Bp_f32 = create_buffer(device, Bp_f32)
    buf_Cp_f32 = create_buffer(device, Cp_f32)
    buf_D_f32 = create_buffer(device, D_f32)
    buf_y_f32 = create_buffer(device, y_f32)

    grid_fp16 = Metal.MTLSize(D, B, 1)
    threads_per_group_fp16 = Metal.MTLSize(int(pso_fp16.threadExecutionWidth()), 1, 1)

    T_TILE = 64
    time_tiles = (L + T_TILE - 1) // T_TILE
    grid_fp32 = Metal.MTLSize(D, B * time_tiles, 1)
    threads_per_group_fp32 = Metal.MTLSize(int(pso_fp32.threadExecutionWidth()), 1, 1)

    iterations = args.iters

    def run_mode(mode):
        if mode=="fp16":
            buffers = [buf_u_f16, buf_delta_f16, buf_A_f16, buf_Bp_f16, buf_Cp_f16, buf_D_f16, buf_y_f16, buf_params]
            pso, grid, tpg, out_buf, dtype, shape = pso_fp16, grid_fp16, threads_per_group_fp16, buf_y_f16, np.float16, (B,L,D)
        else:
            buffers = [buf_u_f32, buf_delta_f32, buf_A_f32, buf_Bp_f32, buf_Cp_f32, buf_D_f32, buf_y_f32, buf_params]
            pso, grid, tpg, out_buf, dtype, shape = pso_fp32, grid_fp32, threads_per_group_fp32, buf_y_f32, np.float32, (B,L,D)

        # Warmup
        for _ in range(10):
            run_once(device, cmd_queue, pso, buffers, grid, tpg, iterations=1)

        start = time.time()
        run_once(device, cmd_queue, pso, buffers, grid, tpg, iterations=iterations)
        end = time.time()
        avg_lat_ms = (end - start) / iterations * 1000.0
        throughput = (B*L) / (avg_lat_ms/1000.0)

        out_np = read_buffer_to_numpy(out_buf, dtype, shape)
        nan_count, inf_count = detect_nan_inf(out_np)
        stats = {"mode": mode, "avg_latency_ms": avg_lat_ms, "throughput_tokens_s": throughput,
                 "nan": nan_count, "inf": inf_count,
                 "mean": float(np.nanmean(out_np.astype(np.float32))),
                 "max": float(np.nanmax(out_np.astype(np.float32)))}
        return stats, out_np

    stats, out = run_mode(args.mode)
    print("第一次結果:", stats)

    if args.mode=="fp16" and (stats["nan"]>0 or stats["inf"]>0):
        print("偵測到 fp16 含 nan/inf，改用 fp32 重跑...")
        stats_fp32, out_fp32 = run_mode("fp32")
        print("fp32 結果:", stats_fp32)
    print("完成。")

if __name__=="__main__":
    main()
