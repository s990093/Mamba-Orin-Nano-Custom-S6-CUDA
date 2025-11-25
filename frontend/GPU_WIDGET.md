# GPU Monitoring Widget

## 功能說明

實時監控 GPU 使用狀況的可折疊小視窗，固定在右下角。

## 後端 API

### WebSocket Endpoint: `/ws/gpu`

每秒推送一次 GPU 狀態：

```json
{
  "memory_allocated": 12345678,  // bytes
  "memory_total": 8589934592,    // bytes  
  "utilization": 45              // %
}
```

**實現位置**: `api_server.py`
- `get_gpu_status()` - 獲取 GPU 狀態
- `/ws/gpu` WebSocket endpoint - 推送即時數據
- 非阻塞設計，不影響 `/generate` 主流程

## 前端組件

### GPUWidget

**檔案**: `frontend/components/GPUWidget.tsx`

**功能**:
- ✅ WebSocket 連接後端
- ✅ 即時顯示記憶體使用量
- ✅ 即時顯示 GPU utilization
- ✅ 可折疊/展開
- ✅ 連線狀態指示
- ✅ 動態進度條（依使用率變色）

**位置**: 固定在右下角 (`fixed bottom-6 right-6`)

**樣式**:
- 玻璃材質背景 (`glass-dark`)
- 動態漸變進度條
- 連線狀態指示燈

## 使用方式

1. **後端自動啟動**: WebSocket 端點在 API 啟動時自動可用
2. **前端自動連接**: GPUWidget 組件掛載時自動連接 WebSocket
3. **即時更新**: 每秒自動更新 GPU 狀態

## 顏色指示

進度條根據使用率變色:
- 🟢 綠色: < 30%
- 🟡 黃色: 30-70%
- 🔴 紅色: > 70%

## 非阻塞設計

- WebSocket 使用獨立的 asyncio coroutine
- 不干擾 `/generate` 的生成流程
- 前端使用獨立的 WebSocket 連線
- 組件卸載時自動關閉連線
