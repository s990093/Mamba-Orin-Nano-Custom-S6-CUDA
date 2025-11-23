#!/usr/bin/env python3
"""
Quick test to verify model correctness with simple greedy decoding
"""

from run_inference import MambaMonitorMetal
import numpy as np

print("Loading model...")
model = MambaMonitorMetal("state-spaces/mamba-130m")

print("\nTesting with greedy generation (temperature=0)...")
prompt = "Hello, my name is"

# Generate with temperature near 0 for deterministic output
output = model.generate(
    prompt, 
    max_length=20,
    temperature=0.01,  # Nearly greedy
    top_k=1,  # Only take top token
    top_p=1.0,
    memory_limit_gb=16.0
)

print(f"\n\nFinal output: {prompt}{output}")
