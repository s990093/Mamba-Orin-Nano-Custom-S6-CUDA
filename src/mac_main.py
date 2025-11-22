import torch
import torch.optim as optim
import time
import numpy as np
import argparse
from mamba_model import Mamba2SSD

def train_step(model, inputs, targets, optimizer):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def benchmark(model_name, device, use_metal=False):
    print(f"\n--- Benchmarking {model_name} (Metal Kernel: {use_metal}) ---")
    
    # Setup
    B, L, D = 4, 1024*2, 256
    model = Mamba2SSD(d_model=D, chunk_size=64, use_metal_kernel=use_metal).to(device)
    
    # 確保參數在正確設備
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    input_data = torch.randn(B, L, D, device=device)
    target_data = torch.randn(B, L, D, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warmup
    print("Warmup...")
    for _ in range(5):
        _ = train_step(model, input_data, target_data, optimizer)
    
    # Timing
    iters = 20
    if device.type == 'mps': torch.mps.synchronize()
    start = time.time()
    
    losses = []
    for _ in range(iters):
        loss = train_step(model, input_data, target_data, optimizer)
        losses.append(loss)
        if device.type == 'mps': torch.mps.synchronize()
        
    end = time.time()
    avg_time = (end - start) / iters
    
    print(f"Average Time per Step: {avg_time*1000:.2f} ms")
    print(f"Final Loss: {losses[-1]:.4f} (Sanity Check: {'Passed' if not np.isnan(losses[-1]) else 'Failed'})")
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()
    
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    # 1. Benchmark Pure PyTorch (MPS)
    # PyTorch's MPS backend is actually very optimized for Matmuls (Mamba2 dominant)
    t_torch = benchmark("Pure PyTorch SSD", device, use_metal=False)
    
    # 2. Benchmark Hybrid (Metal Kernel for Recurrence)
    # This might show overhead on small models due to data copy, 
    # but demonstrates the working pipeline for large scale scans.
    if device.type == 'mps':
        try:
            t_metal = benchmark("PyTorch + Custom Metal", device, use_metal=True)
            speedup = t_torch / t_metal
            print(f"\nSpeedup (Torch/Metal): {speedup:.2f}x")
            if speedup < 1.0:
                print("Note: Custom Metal kernel overhead (tensor->numpy->buffer) dominates at small scales.")
                print("To see speedups, implement zero-copy via C++ extension or use larger Sequence Length.")
        except Exception as e:
            print(f"\nMetal Benchmark Failed: {e}")
            print("Make sure 'mamba_ssd.metal' is compiled or accessible.")

if __name__ == "__main__":
    main()