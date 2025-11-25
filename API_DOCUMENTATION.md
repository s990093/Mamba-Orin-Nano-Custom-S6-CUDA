# Mamba2 MLX API Documentation

## Overview
FastAPI backend for Mamba2 MLX text generation with comprehensive metrics and top-k candidate tracking.

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. POST `/generate`
Generate text using Mamba2 model with detailed metrics.

#### Request Body
```json
{
  "prompt": "Mamba is a type of snake that",
  "max_tokens": 50,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.2,
  "include_top_k_candidates": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input prompt for generation |
| `max_tokens` | integer | No | 50 | Max tokens to generate (1-500) |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_k` | integer | No | null | Top-k sampling |
| `top_p` | float | No | null | Top-p/nucleus sampling (0.0-1.0) |
| `repetition_penalty` | float | No | 1.0 | Repetition penalty (0.0-2.0) |
| `include_top_k_candidates` | boolean | No | false | Include top-k candidates per step |

#### Response Schema

```typescript
{
  generated_text: string;           // Full text (prompt + generated)
  prompt: string;                   // Original prompt
  generated_only: string;           // Only generated part
  
  generation_steps?: Array<{        // Optional, if include_top_k_candidates=true
    step: number;                   // Generation step number
    token_id: number;               // Selected token ID
    token_text: string;             // Decoded token text
    log_probability: number;        // Log prob of selected token
    top_k_candidates?: Array<{      // Top-k candidates
      token_id: number;
      token_text: string;
      probability: number;
      log_probability: number;
    }>;
  }>;
  
  statistics: {
    prompt_length: number;          // Tokens in prompt
    generated_tokens: number;       // Tokens generated
    total_tokens: number;           // Total tokens
  };
  
  speed_metrics: {
    prefill_time: number;           // Prefill time (seconds)
    prefill_speed: number;          // Prefill speed (tokens/s)
    decode_time: number;            // Decode time (seconds)
    decode_speed: number;           // Decode speed (tokens/s)
    avg_latency: number;            // Avg latency (ms/token)
    total_time: number;             // Total time (seconds)
  };
  
  memory_usage: {
    device_type: string;            // GPU/CPU
    initial_memory: number;         // Initial memory (MB)
    current_memory: number;         // Current memory (MB)
    peak_memory: number;            // Peak memory (MB)
    memory_used: number;            // Memory used (MB)
    tracking_available: boolean;    // Memory tracking available
  };
  
  quality_metrics: {
    avg_log_prob: number;           // Average log probability
    perplexity: number;             // Perplexity
    num_repeats: number;            // Number of repeated tokens
    most_repeated: Array<{          // Most repeated tokens
      token: string;
      count: number;
    }>;
  };
  
  parameters: {                     // Generation parameters used
    temperature: number;
    top_k: number | null;
    top_p: number | null;
    repetition_penalty: number;
    max_tokens: number;
  };
}
```

#### Example Response

```json
{
  "generated_text": "Mamba is a type of snake that lives in Africa...",
  "prompt": "Mamba is a type of snake that",
  "generated_only": " lives in Africa...",
  "generation_steps": [
    {
      "step": 0,
      "token_id": 3160,
      "token_text": " lives",
      "log_probability": -0.0015,
      "top_k_candidates": [
        {
          "token_id": 3160,
          "token_text": " lives",
          "probability": 0.9985,
          "log_probability": -0.0015
        },
        {
          "token_id": 318,
          "token_text": " is",
          "probability": 0.0010,
          "log_probability": -6.9078
        }
      ]
    }
  ],
  "statistics": {
    "prompt_length": 8,
    "generated_tokens": 31,
    "total_tokens": 39
  },
  "speed_metrics": {
    "prefill_time": 1.5336,
    "prefill_speed": 5.22,
    "decode_time": 1.5116,
    "decode_speed": 20.51,
    "avg_latency": 48.76,
    "total_time": 3.0452
  },
  "memory_usage": {
    "device_type": "GPU",
    "initial_memory": 5520.17,
    "current_memory": 5620.32,
    "peak_memory": 11040.34,
    "memory_used": 100.16,
    "tracking_available": true
  },
  "quality_metrics": {
    "avg_log_prob": -0.0085,
    "perplexity": 1.01,
    "num_repeats": 13,
    "most_repeated": [
      {"token": " ", "count": 5},
      {"token": " a", "count": 3},
      {"token": " type", "count": 3}
    ]
  },
  "parameters": {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "max_tokens": 30
  }
}
```

### 2. GET `/health`
Health check endpoint.

#### Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true
}
```

### 3. GET `/model_info`
Get loaded model information.

#### Response
```json
{
  "config": {
    "d_model": 2048,
    "n_layer": 48,
    "vocab_size": 50288,
    "d_state": 128,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "ngroups": 1,
    "tie_word_embeddings": false,
    "rms_norm_eps": 1e-05
  },
  "device": "DEVICE(gpu, 0)"
}
```

## Running the Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
python api_server.py

# Or with uvicorn
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

## Interactive API Documentation

Once running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Mamba is a type of snake that",
        "max_tokens": 50,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "include_top_k_candidates": true
    }
)

result = response.json()
print(result["generated_text"])
print(f"Perplexity: {result['quality_metrics']['perplexity']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Mamba is a type of snake that",
    "max_tokens": 30,
    "temperature": 0.8,
    "include_top_k_candidates": true
  }'
```

### JavaScript/TypeScript
```typescript
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Mamba is a type of snake that',
    max_tokens: 30,
    temperature: 0.8,
    include_top_k_candidates: true
  })
});

const result = await response.json();
console.log(result.generated_text);
```
