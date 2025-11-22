import time
import numpy as np
import torch
import torch.nn as nn

# 假設 MambaBlockMock 在同目錄下 mamba_block_mock.py

from mamba_block_mock import MambaBlockMock
import Metal

def run_comparison_benchmark():
    # --- Config ---
    BATCH_SIZE = 4
    D_MODEL = 64
    N_HEAD = 4
    seq_lengths = [128, 512, 1024, 2048, 4096]


    print(f"{'='*60}")  
    print(f"{'Seq Len':<10} | {'Device':<10} | {'Model':<15} | {'Time (ms)':<10}")  
    print(f"{'='*60}")  

    # --- Transformer (PyTorch MPS) ---  
    transformer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_MODEL*4, batch_first=True, norm_first=True, dropout=0.0)  
    transformer.eval()  
    transformer.to('mps')  

    # --- Mamba (Metal Mock) ---  
    metal_device = Metal.MTLCreateSystemDefaultDevice()  
    mamba_mock = MambaBlockMock()  

    for L in seq_lengths:  
        # --- Data ---  
        x_torch = torch.randn(BATCH_SIZE, L, D_MODEL, device='mps')  
        x_np = np.random.randn(BATCH_SIZE, L, D_MODEL).astype(np.float32)  

        # --- Transformer benchmark ---  
        _ = transformer(x_torch)  # warmup  
        torch.mps.synchronize()  
        t0 = time.time()  
        with torch.no_grad():  
            _ = transformer(x_torch)  
        torch.mps.synchronize()  
        t_transformer = (time.time() - t0) * 1000  

        # --- Mamba benchmark ---  
        t0 = time.time()  
        _ = mamba_mock.forward(x_np)  
        t_mamba = (time.time() - t0) * 1000  

        # --- Print results ---  
        print(f"{L:<10} | {'MPS':<10} | {'Transformer':<15} | {t_transformer:8.2f} ms")  
        print(f"{L:<10} | {'Metal':<10} | {'Mamba':<15} | {t_mamba:8.2f} ms")  
        print('-'*60)  


if __name__ == "__main__":
    run_comparison_benchmark()
  