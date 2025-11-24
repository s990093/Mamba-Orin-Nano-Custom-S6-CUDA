#!/usr/bin/env python
"""
Run Mamba2 inference with Metal acceleration.
"""

import argparse
import torch
import torch.nn as nn
import time
import psutil
import os
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from mamba2_metal import Mamba2ModelMetal

class InferenceParams:
    """Inference parameters for caching."""
    def __init__(self, max_seqlen, batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = batch_size
        self.seqlen_offset = 0
        self.key_value_memory_dict = {}

class Mamba2Inference:
    def __init__(self, model_name="state-spaces/mamba2-130m", device="cpu"):
        """
        Initialize Mamba2 model with Metal acceleration.
        """
        self.process = psutil.Process(os.getpid())
        self.device = device
        self.model_name = model_name
        
        print(f"Loading tokenizer from {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except:
            print("Fallback to GPT-NeoX tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading config from {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Initializing Mamba2Metal model...")
        self.model = Mamba2ModelMetal(self.config, device=device)
        
        print(f"Loading weights...")
        self.model.load_hf_weights(model_name)
        self.model.to(device)
        self.model.eval()
        
        print("✓ Model loaded successfully")

    def _get_memory_mb(self):
        return self.process.memory_info().rss / 1024 / 1024

    def sample_token(self, logits, temperature=1.0, top_k=50, top_p=0.9):
        """
        Sample a token from logits using Temperature, Top-K, and Top-P.
        """
        # logits: (1, vocab_size)
        logits = logits[0, -1, :] # Take last token's logits
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, probs.size(-1))  # Safety check
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum() # Re-normalize

        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum() # Re-normalize
            
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        token_prob = probs[next_token] # Keep as tensor to avoid sync
        
        return next_token.unsqueeze(0), token_prob

    def generate(self, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95, memory_limit_gb=8.0, benchmark=False):
        """
        Generate text with observation mode.
        """
        print("\n" + "=" * 60)
        print(f"🚀 GENERATION START")
        print(f"Config: Temp={temperature}, TopK={top_k}, TopP={top_p}, MemLimit={memory_limit_gb}GB, Benchmark={benchmark}")
        print("=" * 60)
        
        start_mem = self._get_memory_mb() / 1024
        print(f"Initial Memory: {start_mem:.2f} GB")
        
        print(f"Tokenizing prompt...")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        batch_size, seqlen = input_ids.shape
        
        print(f"Input tokens: {seqlen}")
        
        # Initialize inference params
        inference_params = InferenceParams(max_seqlen=max_length + seqlen, batch_size=batch_size)
        
        t0 = time.perf_counter()
        
        # Prefill
        with torch.no_grad():
            logits = self.model(input_ids, inference_params=inference_params)
            inference_params.seqlen_offset += seqlen
            
            # Sample first token
            t_start = time.perf_counter()
            next_token, prob = self.sample_token(logits, temperature, top_k, top_p)
            t_end = time.perf_counter()
            
            latencies = [t_end - t_start]
            
            generated = [input_ids]
            generated.append(next_token)
            
            nll_sum = 0.0
            prob_sum = 0.0
            
            if not benchmark:
                nll_sum = -np.log(prob.item() + 1e-10)
                prob_sum = prob.item()
                print(f"Prompt processed. Starting generation...")
                print("-" * 60)
                print(f"{prompt}", end="", flush=True)
                print(self.tokenizer.decode(next_token[0]), end="", flush=True)
            else:
                print(f"Prompt processed. Starting benchmark generation (no output)...")
            
            gen_count = 0
            
            # Generation loop
            for _ in range(max_length):
                if not benchmark:
                    curr_mem = self._get_memory_mb() / 1024
                    if curr_mem > memory_limit_gb:
                        print(f"\n🛑 EMERGENCY STOP: Memory > {memory_limit_gb} GB")
                        break
                
                
                t_step_start = time.perf_counter()
                
                # Forward pass for single token
                logits = self.model(next_token, inference_params=inference_params)
                inference_params.seqlen_offset += 1
                
                # Sample
                next_token, prob = self.sample_token(logits, temperature, top_k, top_p)
                
                if not benchmark:
                    # Syncs with CPU
                    t_step_end = time.perf_counter()
                    latencies.append(t_step_end - t_step_start)
                    
                    nll_sum += -np.log(prob.item() + 1e-10)
                    prob_sum += prob.item()
                    
                    token_str = self.tokenizer.decode(next_token[0])
                    print(token_str, end="", flush=True)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                else:
                    # Benchmark mode: minimal overhead
                    # We still need to check EOS if we want fair comparison, but .item() is a sync.
                    # For pure throughput, we might skip EOS check or do it periodically.
                    # Here we skip EOS check to measure raw GPU speed.
                    pass
                
                generated.append(next_token)
                gen_count += 1
        
        # Ensure GPU is done
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
            
        t1 = time.perf_counter()
        total_time = t1 - t0
        total_tokens = gen_count + 1
        
        print("\n" + "=" * 60)
        
        # Statistics
        final_mem = self._get_memory_mb() / 1024
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        print(f"📊 FINAL STATISTICS")
        print(f"Model           : {self.model_name}")
        print(f"Config          : Temp={temperature}, TopK={top_k}, TopP={top_p}")
        print(f"Total Time      : {total_time:.2f}s")
        print(f"Throughput      : {throughput:.2f} tokens/s")
        
        if not benchmark:
            ppl = np.exp(nll_sum / total_tokens)
            avg_latency = np.mean(latencies) * 1000
            latency_std = np.std(latencies) * 1000
            avg_conf = prob_sum / total_tokens
            print(f"Avg Latency     : {avg_latency:.2f} ms/token (±{latency_std:.2f} ms)")
            print(f"Generation PPL  : {ppl:.4f} (Lower is better)")
            print(f"Avg Confidence  : {avg_conf:.4f} (Higher is better)")
        else:
            print(f"Avg Latency     : {(total_time / total_tokens * 1000):.2f} ms/token (Amortized)")
            
        print(f"Final Memory    : {final_mem:.2f} GB")
        print("=" * 60)
        
        full_ids = torch.cat(generated, dim=1)
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Run Mamba2 Inference with Metal Acceleration")
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m", help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="Mamba is a type of", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-P sampling")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda/mps)")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (no output, no sync)")
    parser.add_argument("--use_metal_engine", action="store_true", help="Use pure Metal Engine for inference")
    
    args = parser.parse_args()
    
    try:
        model = Mamba2Inference(model_name=args.model, device=args.device)
        
        if args.use_metal_engine:
            if args.device != "mps":
                print("Warning: --use_metal_engine requires --device mps. Switching to mps.")
                args.device = "mps"
            
            from metal_ops.mamba_engine import MambaEngine
            print("🚀 Switching to Pure Metal Engine...")
            engine = MambaEngine(model.model, device="mps")
            
            # Store the original model for prefill
            original_mamba2_model = model.model

            # Define a wrapper
            class MetalEngineWrapper:
                def __init__(self, engine, original_model):
                    self.engine = engine
                    self.original_model = original_model # Store the original PyTorch model
                    
                def __call__(self, input_ids, inference_params=None):
                    # input_ids: (B, L)
                    # If L > 1, it's prefill. Engine currently only supports step (L=1).
                    # For prefill, we might need to fallback or implement prefill in engine.
                    # For now, let's assume prefill is done by PyTorch model, and we switch to engine for decoding.
                    
                    if input_ids.shape[1] > 1:
                        # Prefill: use the original PyTorch model
                        return self.original_model(input_ids, inference_params)
                    else:
                        # Decoding step: use the Metal Engine
                        return self.engine.step(input_ids)
            
            # Replace model.model with the wrapper
            model.model = MetalEngineWrapper(engine, original_mamba2_model)
            
        output = model.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            benchmark=args.benchmark
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()