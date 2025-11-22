# 🚀 Mamba-Orin-Nano-Custom-S6-CUDA & Metal

## **跨平台極致加速 Mamba SSM：CUDA + TensorRT Plugin + Metal S6 Kernel**

![架構圖](assets/architecture_diagram.png)

### **Jetson Orin Nano × macOS M1/M2/M3 全面支援**

從嵌入式邊緣推理到 Apple Silicon 的 Metal GPU，我們打造一套 **跨硬體、跨後端、專為 Mamba 結構化狀態空間模型（SSM）打造的極速 S6 Kernel 加速方案**。

這是一個 **「一次寫 Kernel，兩邊都超快」** 的野心專案。
當然，還有一點點「工程師靈魂不滅」的浪漫。

---

# 🌟 專案亮點一覽

## 🔥 Orin Nano：CUDA + TensorRT Plugin 版本

為 **Jetson Orin Nano (Ampere)** 量身打造的 **S6 遞歸 Selective Scan 加速器**：

- **自定義 CUDA S6 Kernel**

  - Shared Memory + 記憶體訪問重排
  - Tiling 以消除長序列遞歸瓶頸
  - Register reuse / 避免 spilling
  - 終極目標：讓 DRAM 休息一下，讓 Compute 裝忙一點

- **TensorRT Plugin 整合**

  - 避免 Graph Break
  - 與 TensorRT-LLM GEMM 融合
  - 支援 FP16 / INT8 / INT4 量化

- **邊緣裝置最佳化**

  - Zero-copy（但要用得剛剛好，不然會變反效果）
  - 異步 Stream pipeline
  - SWAP 與 NVMe 調教

---

## 🍏 macOS：Metal S6 Benchmark（M1/M2/M3）

同一個 S6 遞歸邏輯，這次換成 **Metal Shading Language (MSL)**：

- 完整對應 CUDA 版的 S6 遞歸運算
- 使用 **Unified Memory** 避免 CPU/GPU 複製
- **FP16 half precision** 加速
- 每個 Thread 處理一個 Channel（完全 SIMD-friendly）
- 做到「Apple Silicon 也跑得飛快」的精神使命

你可以把這想像成：

> CUDA 是肌肉硬漢版，Metal 是優雅忍者版。
> 目的只有一個：把遞歸 S6 打到快到飛起來。

---

# 🧱 重新整理後的專案結構（含 CUDA + Metal）

```
Mamba-Orin-Nano-Custom-S6-CUDA/
├── src/
│   ├── custom_s6_kernel/           # CUDA Kernel (.cu / .cuh)
│   ├── tensorrt_s6_plugin/         # TensorRT Plugin (.cpp / .hpp)
│   └── metal/                      # Metal S6 Kernel (.metal)
│       └── mamba_s6.metal
│
├── models/
│   └── mamba_weights/              # 模型權重
│
├── scripts/
│   ├── convert_model.py
│   ├── build_tensorrt_engine.py
│   ├── run_inference.py
│   └── mamba_metal_benchmark.py    # 專給 macOS
│
├── docs/
│   └── design_report.md            # 技術報告（放你的論文級分析）
│
├── assets/
│   └── architecture_diagram.png
│
├── README.md
└── requirements.txt
```

---

# ⚙️ 安裝與環境設定

---

# 1️⃣ Jetson Orin Nano（CUDA + TensorRT）

### **需求**

- JetPack (含 CUDA, cuDNN, TensorRT)
- Python 3.8+
- Build-essential / CMake

### **安裝依賴**

```bash
sudo apt update
sudo apt install -y build-essential

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **編譯 S6 CUDA Kernel + TensorRT Plugin**

```bash
cd src/tensorrt_s6_plugin
mkdir build && cd build
cmake .. -DCUDA_ARCHITECTURES="8.7"
make -j$(nproc)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
```

---

# 2️⃣ macOS（Metal S6 Benchmark）

### **需求**

- macOS 12.0+
- M1 / M2 / M3（Intel Mac 無法 GPU 測試）
- Python 3.x

### **安裝 Metal Python binding**

```bash
pip3 install numpy pyobjc-framework-Metal pyobjc-framework-Cocoa
```

### **執行 Metal S6 benchmark**

```bash
python3 scripts/mamba_metal_benchmark.py
```

---

# 🚀 性能預期

## Jetson Orin Nano（FP16 / INT8）

| 方法                                         | 關鍵優化                         | 延遲 (ms/token) | 吞吐量      |
| -------------------------------------------- | -------------------------------- | --------------- | ----------- |
| PyTorch baseline                             | S6 100% memory-bound             | >40             | <25         |
| TensorRT-LLM                                 | GEMM fused                       | 10–20           | 50–100      |
| **本專案：S6 Custom CUDA Kernel + TensorRT** | **Shared Mem + Tiling + Plugin** | **5–10**        | **100–200** |
| **本專案（INT8）**                           | **量化 + custom kernel**         | **< 5**         | **> 200**   |

## macOS Metal（M1/M2/M3）

- FP16 = 完全用原生 half precision
- Unified Memory = 真零拷貝
- Threadgroup = 不需要協作也能大殺特殺

Metal 版本實際上會很接近 CUDA FP16 版的「理想 memory-bound 上限」，可用來驗證：

> **S6 遞歸演算法的硬體可攜性**
> → 這份比較在你的論文裡會超級加分。

---

# 🧪 使用方法

---

## 1️⃣ 模型轉換（PyTorch → ONNX）

```bash
python scripts/convert_model.py \
  --model_name "mamba-2.8b" \
  --output_path "models/mamba_onnx/mamba.onnx"
```

---

## 2️⃣ 建構 TensorRT 引擎（含 S6 Plugin）

```bash
python scripts/build_tensorrt_engine.py \
  --onnx_model_path "models/mamba_onnx/mamba.onnx" \
  --output_engine_path "models/mamba_tensorrt_engine.trt" \
  --s6_plugin_path "src/tensorrt_s6_plugin/build/libs6plugin.so" \
  --precision "fp16"
```

---

## 3️⃣ 執行推理

```bash
python scripts/run_inference.py \
  --tensorrt_engine_path "models/mamba_tensorrt_engine.trt" \
  --input_text "The quick brown fox jumps over the lazy dog." \
  --sequence_length 1024 \
  --num_iterations 100 \
  --compare_pytorch
```

---

# 💡 為什麼要同時做 CUDA + Metal？

因為這讓你可以：

- 驗證 S6 遞歸核心演算法的跨硬體一致性
- 測試 memory-bound / compute-bound 行為在兩種架構的差異
- 實現 **portable backend**：「同一個模型，同一邏輯，哪裡有 GPU 我就跑哪裡」

這對於你未來投稿、論文、履歷、面試，都是非常炫砲的亮點。

---

# 🤝 貢獻

歡迎提出：

- Bug report
- Kernel 優化建議
- Metal / CUDA / ROCm / Vulkan 其他後端（對，你完全可以擴展！）

---

# 📄 License

MIT License

---
