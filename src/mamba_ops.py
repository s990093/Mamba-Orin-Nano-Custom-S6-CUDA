import Metal
import Foundation
import objc
import numpy as np
import time

# 設定維度參數 (模擬小型 LLM layer)
BATCH = 4
N_HEADS = 32
DIM = 64      # 每個 Head 的維度
D_STATE = 16  # SSM State 維度
N_GROUPS = 32 # Mamba 1通常是 1 group, Mamba 2 可能不同

def load_metal_device():
    device = Metal.MTLCreateSystemDefaultDevice()
    if not device:
        raise RuntimeError("Metal device not found. Are you on a Mac?")
    print(f"Running on: {device.name()}")
    return device

def compile_kernel(device, source_file="selective_scan.metal"):
    try:
        with open(source_file, "r") as f:
            source = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Cannot find {source_file}. Please ensure the .metal file is in the same directory.")
    
    options = Metal.MTLCompileOptions.new()
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    
    if error:
        raise RuntimeError(f"Compile Error: {error}")
    
    func = library.newFunctionWithName_("selective_scan_update_kernel")
    if not func:
        raise RuntimeError("Function 'selective_scan_update_kernel' not found in .metal file")
        
    pipeline_state, error = device.newComputePipelineStateWithFunction_error_(func, None)
    
    if error:
        raise RuntimeError(f"Pipeline Error: {error}")
        
    return pipeline_state

def np_to_metal(device, np_array):
    """將 Numpy array 轉為 Metal Buffer"""
    data = np_array.astype(np.float32).copy(order='C') 
    return device.newBufferWithBytes_length_options_(
        data.tobytes(),
        data.nbytes,
        Metal.MTLResourceStorageModeShared
    )

def run_benchmark():
    device = load_metal_device()
    pipeline = compile_kernel(device, 'src/metal/mamba_ssd.metal')
    command_queue = device.newCommandQueue()

    print(f"\nConfig: B={BATCH}, H={N_HEADS}, D={DIM}, State={D_STATE}")

    # --- 1. 準備數據 (修正初始化) ---
    np.random.seed(0)
    shape_x = (BATCH, N_HEADS, DIM)
    shape_state = (BATCH, N_HEADS, DIM, D_STATE)
    shape_A = (N_HEADS, DIM, D_STATE)
    shape_B = (BATCH, N_GROUPS, D_STATE)
    
    h_state = np.random.randn(*shape_state).astype(np.float32)
    h_x = np.random.randn(*shape_x).astype(np.float32)
    h_dt = np.random.randn(*shape_x).astype(np.float32)
    h_dt_bias = np.random.randn(N_HEADS, DIM).astype(np.float32)
    
    # --- 重要修正：A 必須是負數以確保數值穩定 ---
    # Mamba 初始化通常是從 uniform 採樣 log 值，這裡簡單設為 -1.0 附近的負數
    h_A = -np.abs(np.random.randn(*shape_A).astype(np.float32)) 
    
    h_B = np.random.randn(*shape_B).astype(np.float32)
    h_C = np.random.randn(*shape_B).astype(np.float32)
    h_D = np.random.randn(N_HEADS, DIM).astype(np.float32)
    h_z = np.random.randn(*shape_x).astype(np.float32)
    h_out = np.zeros_like(h_x)

    buffers_data = [h_state, h_x, h_dt, h_dt_bias, h_A, h_B, h_C, h_D, h_z, h_out]
    # 建立 Metal Buffer
    buffers = [np_to_metal(device, b) for b in buffers_data]
    
    # 保留 state buffer 的參照，以便在迴圈中重置 (如果需要更嚴謹的單步測試)
    buf_state = buffers[0] 
    buf_out = buffers[-1]
    
    # 原始 state 數據的 bytes，用於重置
    initial_state_bytes = h_state.astype(np.float32).tobytes()

    # --- 2. 準備 Scalar 參數 ---
    params = [
        BATCH, N_HEADS, DIM, D_STATE, N_GROUPS,
        True, True, True # has_z, has_dt_bias, has_D
    ]

    # --- 3. 執行 Benchmark ---
    grid_size = Metal.MTLSize(DIM, N_HEADS, BATCH)
    thread_group_size = Metal.MTLSize(min(DIM, 512), 1, 1)

    print("\nRunning loop benchmark (100 iters)...")
    
    # 預熱
    cmd_buffer = command_queue.commandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for i, buf in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)
    
    current_idx = 10
    for p in params:
        val = np.uint32(p)
        encoder.setBytes_length_atIndex_(val.tobytes(), 4, current_idx)
        current_idx += 1
        
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    # 正式測試
    total_time = 0
    ITERATIONS = 100
    
    for _ in range(ITERATIONS):
        # 選項：每次重置 State 以模擬單步推論，避免累積誤差 (對於單步效能測試這更準確)
        # 使用 replaceRegion 快速重寫 buffer 內容
        # buf_state.contents().as_buffer(len(initial_state_bytes))[:] = initial_state_bytes
        # 註：為了追求極致速度測試，通常不包含記憶體重置時間。
        # 因為我們修復了 A < 0，即使不重置，數值也會收斂，不會 NaN。

        cmd_buffer = command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            
        current_idx = 10
        for p in params:
            val = np.uint32(p)
            encoder.setBytes_length_atIndex_(val.tobytes(), 4, current_idx)
            current_idx += 1
            
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, thread_group_size)
        encoder.endEncoding()
        
        t0 = time.time()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()
        t1 = time.time()
        total_time += (t1 - t0)
        
    avg_time = (total_time / ITERATIONS) * 1000
    print(f"Avg Time per iter: {avg_time:.4f} ms")

    # --- 4. 驗證輸出 ---
    result_ptr = buf_out.contents().as_buffer(buf_out.length())
    result_np = np.frombuffer(result_ptr, dtype=np.float32).reshape(shape_x)
    
    print(f"\nOutput Sample (Batch 0, Head 0, first 5 dims):")
    print(result_np[0, 0, :5])
    
    # 檢查是否還有 NaN
    if np.isnan(result_np).any():
        print("Error: NaN detected in output!")
    else:
        print(f"Mean Output: {np.mean(result_np)}")
        print("Success: Output is stable.")

if __name__ == "__main__":
    run_benchmark()