#!/usr/bin/env python3
"""
Benchmark script to measure Mamba Metal performance optimizations.
Tests both loading time and inference throughput.
"""

import time
import numpy as np
import sys
import os

def benchmark_loading(model_path="state-spaces/mamba-130m"):
    """Benchmark model loading time."""
    print("="*60)
    print("LOADING TIME BENCHMARK")
    print("="*60)
    
    # Measure loading time
    t0 = time.perf_counter()
    from run_inference import MambaMonitorMetal
    model = MambaMonitorMetal(model_path)
    t1 = time.perf_counter()
    
    loading_time = t1 - t0
    print(f"\n✓ Model loaded in {loading_time:.2f}s")
    
    return model, loading_time

def benchmark_inference(model, prompt="Deep learning is", num_tokens=50):
    """Benchmark inference throughput."""
    print("\n" + "="*60)
    print("INFERENCE THROUGHPUT BENCHMARK")
    print("="*60)
    print(f"Prompt: '{prompt}'")
    print(f"Generating {num_tokens} tokens...")
    
    # Warm-up run (initialize caches)
    print("\n[Warm-up run]")
    _ = model.generate(prompt, max_length=5, temperature=1.0, memory_limit_gb=16.0)
    
    # Actual benchmark
    print("\n[Benchmark run]")
    t0 = time.perf_counter()
    output = model.generate(
        prompt, 
        max_length=num_tokens, 
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        memory_limit_gb=16.0
    )
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    throughput = num_tokens / total_time if total_time > 0 else 0
    latency_per_token = (total_time / num_tokens * 1000) if num_tokens > 0 else 0
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total Time       : {total_time:.2f}s")
    print(f"Throughput       : {throughput:.2f} tokens/s")
    print(f"Latency/Token    : {latency_per_token:.2f} ms/token")
    print("="*60)
    
    return {
        'total_time': total_time,
        'throughput': throughput,
        'latency_per_token': latency_per_token
    }

def main():
    print("\n🚀 Mamba Metal Performance Benchmark\n")
    
    # Check if model path argument is provided
    model_path = "state-spaces/mamba-130m"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Check for local model
    if os.path.exists("model/mamba-130m"):
        model_path = "model/mamba-130m"
        print(f"📁 Using local model: {model_path}")
    
    # Benchmark loading
    model, loading_time = benchmark_loading(model_path)
    
    # Benchmark inference
    results = benchmark_inference(
        model, 
        prompt="Deep learning on Apple Silicon is",
        num_tokens=50
    )
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Loading Time     : {loading_time:.2f}s")
    print(f"Inference Speed  : {results['throughput']:.2f} tokens/s")
    print(f"Token Latency    : {results['latency_per_token']:.2f} ms")
    print("="*60)
    
    # Performance targets
    print("\n📊 Performance Targets:")
    print(f"  Loading Time   : {'✓ PASS' if loading_time < 3.0 else '✗ FAIL'} (target: <3s)")
    print(f"  Throughput     : {'✓ PASS' if results['throughput'] > 50 else '✗ FAIL'} (target: >50 tok/s)")
    print(f"  Token Latency  : {'✓ PASS' if results['latency_per_token'] < 100 else '✗ FAIL'} (target: <100ms)")
    print()

if __name__ == "__main__":
    main()
