import tensorrt as trt
import ctypes
import numpy as np
import time
import os

# 1. 載入自定義 Plugin 庫
PLUGIN_LIB_PATH = "./build/libmamba_plugin.so"
if not os.path.exists(PLUGIN_LIB_PATH):
    print(f"錯誤: 找不到 {PLUGIN_LIB_PATH}。請先執行 cmake 與 make。")
    exit(1)

ctypes.CDLL(PLUGIN_LIB_PATH)

# 註冊 TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_plugin_creator(plugin_name):
    registry = trt.get_plugin_registry()
    creators = registry.plugin_creator_list
    for creator in creators:
        if creator.name == plugin_name:
            return creator
    return None

def build_engine(batch_size, seq_len, d_model, d_state):
    print(f"正在構建 TensorRT Engine (B={batch_size}, L={seq_len}, D={d_model})...")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # 啟用 FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 模式: 已啟用")
    else:
        print("警告: 此平台不支援 FP16")

    # 定義輸入
    u = network.add_input("u", trt.float16, (batch_size, seq_len, d_model))
    delta = network.add_input("delta", trt.float16, (batch_size, seq_len, d_model))
    A = network.add_input("A", trt.float16, (d_model, d_state))
    B = network.add_input("B", trt.float16, (batch_size, seq_len, d_state))
    C = network.add_input("C", trt.float16, (batch_size, seq_len, d_state))
    D = network.add_input("D", trt.float16, (d_model,))

    # 建立 Plugin
    creator = get_plugin_creator("MambaS6Plugin")
    if not creator:
        raise RuntimeError("找不到 MambaS6Plugin，請檢查 .so 是否載入成功")

    # 設定 Plugin 屬性 (d_state)
    d_state_field = trt.PluginField("d_state", np.array([d_state], dtype=np.int32), trt.PluginFieldType.INT32)
    field_collection = trt.PluginFieldCollection([d_state_field])
    plugin = creator.create_plugin("mamba_s6", field_collection)

    # 將 Plugin 加入網路
    inputs = [u, delta, A, B, C, D]
    layer = network.add_plugin_v2(inputs, plugin)
    layer.get_output(0).name = "output"
    
    network.mark_output(layer.get_output(0))

    # 構建 Serialized Engine
    plan = builder.build_serialized_network(network, config)
    return plan

def run_inference(engine_bytes):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # 獲取維度資訊
    batch_size = 1
    seq_len = 1024
    d_model = 768 # Mamba small 規模
    d_state = 16
    
    # 準備 Dummy Data (FP16)
    # 注意：在 Python 中使用 float16 需依賴 numpy
    input_shapes = [
        (batch_size, seq_len, d_model), # u
        (batch_size, seq_len, d_model), # delta
        (d_model, d_state),             # A
        (batch_size, seq_len, d_state), # B
        (batch_size, seq_len, d_state), # C
        (d_model,)                      # D
    ]
    
    inputs_host = [np.random.randn(*shape).astype(np.float16) for shape in input_shapes]
    output_host = np.empty((batch_size, seq_len, d_model), dtype=np.float16)

    # 分配 GPU 記憶體
    import pycuda.driver as cuda
    import pycuda.autoinit

    inputs_device = [cuda.mem_alloc(inp.nbytes) for inp in inputs_host]
    output_device = cuda.mem_alloc(output_host.nbytes)

    # 複製資料到 GPU
    for inp_h, inp_d in zip(inputs_host, inputs_device):
        cuda.memcpy_htod(inp_d, inp_h)

    # 綁定位址
    bindings = [int(inp_d) for inp_d in inputs_device] + [int(output_device)]

    # 暖身 (Warmup)
    print("暖身中...")
    for _ in range(10):
        context.execute_v2(bindings)

    # 效能測試
    print("開始基準測試...")
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        context.execute_v2(bindings)
    end_time = time.time()

    # 計算結果
    avg_latency = (end_time - start_time) / iterations * 1000 # ms
    throughput = (batch_size * seq_len) / (avg_latency / 1000) # tokens/sec

    print(f"--- 測試結果 (Jetson Orin Nano) ---")
    print(f"Seq Len: {seq_len}, D_model: {d_model}")
    print(f"平均延遲: {avg_latency:.4f} ms")
    print(f"吞吐量: {throughput:.2f} tokens/sec")
    
    # 複製結果回 Host (驗證用)
    cuda.memcpy_dtoh(output_host, output_device)
    print(f"輸出形狀: {output_host.shape}")

if __name__ == "__main__":
    try:
        # 參數設定
        B, L, D, N = 1, 1024, 768, 16
        engine_bytes = build_engine(B, L, D, N)
        run_inference(engine_bytes)
    except Exception as e:
        print(f"發生錯誤: {e}")