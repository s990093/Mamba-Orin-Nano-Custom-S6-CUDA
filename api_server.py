"""
FastAPI backend for Mamba2 MLX text generation.
Provides a REST API to generate text with comprehensive metrics.
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import mlx.core as mx
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerFast
import os
import sys
import asyncio
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.mamba2_mlx import Mamba2Model, Mamba2Config, load_weights
from src.mamba2_mlx.generation import generate_with_metrics


app = FastAPI(
    title="Mamba2 MLX API",
    description="Text generation API using Mamba2 model with MLX backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或限制到你前端的 URL
    allow_credentials=True,
    allow_methods=["*"],  # 支援 GET, POST, OPTIONS...
    allow_headers=["*"],
)

# Request Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation", example="Mamba is a type of snake that")
    max_tokens: int = Field(50, description="Maximum number of tokens to generate", ge=1, le=500)
    temperature: float = Field(0.7, description="Sampling temperature (higher = more random)", ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, description="Top-k sampling (limit to top k tokens)", ge=1)
    top_p: Optional[float] = Field(None, description="Top-p/nucleus sampling", ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, description="Repetition penalty (>1.0 discourages repetition)", ge=0.0, le=2.0)
    include_top_k_candidates: bool = Field(False, description="Include top-k candidate tokens for each generation step")

# Response Models
class TopKCandidate(BaseModel):
    token_id: int = Field(..., description="Token ID")
    token_text: str = Field(..., description="Decoded token text")
    probability: float = Field(..., description="Probability of this token")
    log_probability: float = Field(..., description="Log probability of this token")

class GenerationStep(BaseModel):
    step: int = Field(..., description="Generation step number")
    token_id: int = Field(..., description="Selected token ID")
    token_text: str = Field(..., description="Decoded token text")
    log_probability: float = Field(..., description="Log probability of selected token")
    top_k_candidates: Optional[List[TopKCandidate]] = Field(None, description="Top-k candidate tokens (if requested)")

class GenerationStatistics(BaseModel):
    prompt_length: int = Field(..., description="Number of tokens in prompt")
    generated_tokens: int = Field(..., description="Number of tokens generated")
    total_tokens: int = Field(..., description="Total tokens (prompt + generated)")

class SpeedMetrics(BaseModel):
    prefill_time: float = Field(..., description="Prefill time in seconds")
    prefill_speed: float = Field(..., description="Prefill speed in tokens/s")
    decode_time: float = Field(..., description="Decode time in seconds")
    decode_speed: float = Field(..., description="Decode speed in tokens/s")
    avg_latency: float = Field(..., description="Average latency per token in milliseconds")
    total_time: float = Field(..., description="Total generation time in seconds")

class MemoryUsage(BaseModel):
    device_type: str = Field(..., description="Device type (GPU/CPU)")
    initial_memory: float = Field(..., description="Initial memory in MB")
    current_memory: float = Field(..., description="Current memory in MB")
    peak_memory: float = Field(..., description="Peak memory in MB")
    memory_used: float = Field(..., description="Memory used in MB")
    tracking_available: bool = Field(..., description="Whether memory tracking is available")

class QualityMetrics(BaseModel):
    avg_log_prob: float = Field(..., description="Average log probability of generated tokens")
    perplexity: float = Field(..., description="Perplexity of the generated sequence")
    num_repeats: int = Field(..., description="Number of repeated tokens")
    most_repeated: List[Dict[str, Any]] = Field(..., description="Most repeated tokens with counts")

class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Complete generated text (prompt + generation)")
    prompt: str = Field(..., description="Original input prompt")
    generated_only: str = Field(..., description="Only the generated part (without prompt)")
    generation_steps: Optional[List[GenerationStep]] = Field(None, description="Detailed generation steps (if top-k candidates requested)")
    statistics: GenerationStatistics = Field(..., description="Generation statistics")
    speed_metrics: SpeedMetrics = Field(..., description="Speed and latency metrics")
    memory_usage: MemoryUsage = Field(..., description="Memory usage metrics")
    quality_metrics: QualityMetrics = Field(..., description="Generation quality metrics")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters used")

# Global model state
model = None
tokenizer = None
config = None

def initialize_model(
    model_path: str = "state-spaces/mamba2-130m",
    tokenizer_path: str = "model/mamba2-780m/tokenizer.json",
    dtype_str: str = "float32"
):
    """Initialize the model and tokenizer."""
    global model, tokenizer, config
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Map dtype
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16
    }
    target_dtype = dtype_map[dtype_str]
    
    # Load tokenizer
    if tokenizer_path.endswith(".json"):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    if os.path.isfile(model_path):
        parent_dir = os.path.dirname(model_path)
        if os.path.exists(os.path.join(parent_dir, "config.json")):
            hf_config = AutoConfig.from_pretrained(parent_dir)
        else:
            hf_config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
    else:
        hf_config = AutoConfig.from_pretrained(model_path)
    
    # Helper to get config attributes
    def get_config_attr(cfg, keys, default):
        for k in keys:
            if hasattr(cfg, k):
                return getattr(cfg, k)
        return default
    
    config = Mamba2Config(
        d_model=get_config_attr(hf_config, ["d_model", "hidden_size"], 768),
        n_layer=get_config_attr(hf_config, ["n_layer", "num_hidden_layers"], 24),
        vocab_size=get_config_attr(hf_config, ["vocab_size"], 50277),
        d_state=get_config_attr(hf_config, ["d_state", "state_size"], 128),
        d_conv=get_config_attr(hf_config, ["d_conv", "conv_kernel"], 4),
        expand=get_config_attr(hf_config, ["expand"], 2),
        headdim=get_config_attr(hf_config, ["headdim", "head_dim"], 64),
        ngroups=get_config_attr(hf_config, ["ngroups", "n_groups"], 1),
        tie_word_embeddings=get_config_attr(hf_config, ["tie_word_embeddings"], True),
        rms_norm_eps=get_config_attr(hf_config, ["rms_norm_eps", "layer_norm_epsilon"], 1e-5),
    )
    
    model = Mamba2Model(config)
    mx.eval(model.parameters())
    load_weights(model, model_path, dtype=target_dtype)
    
    print(f"✅ Model initialized: {model_path}")
    print(f"✅ Tokenizer loaded: {tokenizer_path}")

@app.on_event("startup")
async def startup_event():
    """Initialize model when API starts."""
    try:
        # Clean memory before loading model
        print("🧹 Cleaning GPU memory before model load...")
        cleanup_memory()
        
        initialize_model()
        
        # Clean memory after loading
        print("🧹 Cleaning GPU memory after model load...")
        cleanup_memory()
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize default model: {e}")
        print("Model will need to be initialized before first request.")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text using Mamba2 model.
    
    Returns comprehensive metrics including generation statistics, speed metrics,
    memory usage, and quality metrics. Optionally includes top-k candidates for each step.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        result = generate_with_metrics(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            return_dict=True,
            include_top_k_candidates=request.include_top_k_candidates
        )
        
        # Clean up memory after generation
        cleanup_memory()
        
        return GenerateResponse(**result)
    
    except Exception as e:
        # Clean up memory even on error
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the API and model are ready."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "config": {
            "d_model": config.d_model,
            "n_layer": config.n_layer,
            "vocab_size": config.vocab_size,
            "d_state": config.d_state,
            "d_conv": config.d_conv,
            "expand": config.expand,
            "headdim": config.headdim,
            "ngroups": config.ngroups,
            "tie_word_embeddings": config.tie_word_embeddings,
            "rms_norm_eps": config.rms_norm_eps
        },
        "device": str(mx.default_device())
    }

def get_gpu_status():
    """Get current GPU memory and utilization status using system commands."""
    try:
        import subprocess
        import re
        
        # Try to get GPU info from Metal using ioreg command
        # This gives us real-time GPU memory usage on macOS
        result = subprocess.run(
            ['ioreg', '-r', '-c', 'IOAccelerator'],
            capture_output=True,
            text=True,
            timeout=1
        )
        
        if result.returncode == 0:
            output = result.stdout
            
            # Parse memory info from ioreg output
            # Look for "Performance Statistics"
            memory_used = 0
            memory_total = 0
            
            # Try to find allocated memory
            alloc_match = re.search(r'"Device Utilisation %"\s*=\s*(\d+)', output)
            mem_match = re.search(r'"vramUsedBytes"\s*=\s*(\d+)', output)
            
            if mem_match:
                memory_used = int(mem_match.group(1))
            
            # Get MLX memory as fallback total
            try:
                memory_allocated = mx.metal.get_active_memory()
                cache_memory = mx.metal.get_cache_memory()
                memory_total = cache_memory if cache_memory > 0 else memory_allocated * 2
                
                # If we found vramUsedBytes from ioreg, use it
                if memory_used > 0:
                    memory_allocated = memory_used
            except:
                # Fallback to reasonable estimates
                memory_total = 8 * 1024**3  # 8GB default for Apple Silicon
                memory_allocated = memory_used if memory_used > 0 else memory_total // 2
            
            # Calculate utilization
            utilization = 0
            if alloc_match:
                utilization = int(alloc_match.group(1))
            else:
                utilization = int((memory_allocated / memory_total) * 100) if memory_total > 0 else 0
            
            status = {
                "memory_allocated": int(memory_allocated),
                "memory_total": int(memory_total),
                "utilization": min(utilization, 100)
            }
            
            # Debug log
            # print(f"GPU Status: allocated={memory_allocated/1024**2:.1f}MB, total={memory_total/1024**2:.1f}MB, util={utilization}%")
            
            return status
            
    except Exception as e:
        print(f"System GPU status error: {e}")
    
    # Fallback to MLX API
    try:
        mx.eval([])
        
        memory_allocated = mx.metal.get_active_memory()
        peak_memory = mx.metal.get_peak_memory()
        cache_memory = mx.metal.get_cache_memory()
        
        memory_total = cache_memory if cache_memory > 0 else max(peak_memory, memory_allocated * 2)
        utilization = int((memory_allocated / memory_total) * 100) if memory_total > 0 else 0
        
        status = {
            "memory_allocated": int(memory_allocated),
            "memory_total": int(memory_total),
            "utilization": min(utilization, 100)
        }
        
        print(f"GPU Status (MLX): allocated={memory_allocated/1024**2:.1f}MB, total={memory_total/1024**2:.1f}MB, util={utilization}%")
        
        return status
    except Exception as e:
        print(f"GPU status error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "memory_allocated": 0,
            "memory_total": 0,
            "utilization": 0
        }

def cleanup_memory():
    """Clean up GPU memory."""
    try:
        mx.metal.clear_cache()
        print("✅ GPU memory cache cleared")
    except Exception as e:
        print(f"⚠️  Memory cleanup warning: {e}")

@app.websocket("/ws/gpu")
async def websocket_gpu_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time GPU monitoring."""
    await websocket.accept()
    print("✅ GPU monitor WebSocket connected")
    
    try:
        while True:
            # Get GPU status
            status = get_gpu_status()
            
            # Send to client
            await websocket.send_json(status)
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔌 GPU monitor WebSocket disconnected")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,        # 開啟熱重載
        log_level="info"    # 可選，顯示日誌
    )