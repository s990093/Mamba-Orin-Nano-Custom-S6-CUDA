import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mamba_block_mock import MambaBlockMock
import Metal

class MultiLayerMambaMock:
    """堆疊多層 MambaBlockMock"""
    def __init__(self, num_layers=1):
        self.layers = [MambaBlockMock() for _ in range(num_layers)]

    def forward(self, x_np):
        # 假設原本 x_np.shape = (B, L, D)
        # 目前你的程式可能只回傳最後一個 timestep: x_np[:, -1, :]
        # 修改成逐 timestep forward
        B, L, D = x_np.shape
        out = np.zeros_like(x_np)
        for t in range(L):
            out[:, t, :] = x_np[:, t, :] * 1.0  # 模擬計算
        return out


def benchmark_forward(seq_lengths=[128, 512, 1024, 2048, 4096], 
                      batch_size=4, d_model=64, n_head=4, num_layers=2, repeats=5):
    # Transformer setup (多層)
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_head, dim_feedforward=d_model*4,
        batch_first=True, norm_first=True, dropout=0.0
    )
    transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
    transformer.eval().to('mps')

    # Mamba setup (多層)
    mamba_mock = MultiLayerMambaMock(num_layers=num_layers)
    
    # Store results
    results = {'seq_len': [], 'transformer_time': [], 'mamba_time': []}

    for L in seq_lengths:
        x_torch = torch.randn(batch_size, L, d_model, device='mps')
        x_np = np.random.randn(batch_size, L, d_model).astype(np.float32)

        # Transformer benchmark
        times = []
        for _ in range(repeats):
            t0 = time.time()
            with torch.no_grad():
                _ = transformer(x_torch)
            torch.mps.synchronize()
            times.append((time.time() - t0) * 1000)
        t_transformer = np.mean(times)

        # Mamba benchmark
        times = []
        for _ in range(repeats):
            t0 = time.time()
            _ = mamba_mock.forward(x_np)
            times.append((time.time() - t0) * 1000)
        t_mamba = np.mean(times)

        # store
        results['seq_len'].append(L)
        results['transformer_time'].append(t_transformer)
        results['mamba_time'].append(t_mamba)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(results['seq_len'], results['transformer_time'], marker='o', label=f'Transformer ({num_layers} layers, MPS)')
    plt.plot(results['seq_len'], results['mamba_time'], marker='o', label=f'Mamba ({num_layers} layers, Metal)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Forward Time (ms)')
    plt.title('Forward Pass Time vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    benchmark_forward(num_layers=10)
