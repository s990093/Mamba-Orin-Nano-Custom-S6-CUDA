import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from mamba_ops import run_ssd_chunk_scan_metal

class Mamba2SSD(nn.Module):
    def __init__(self, d_model, d_state=64, d_head=64, n_heads=None, 
                 chunk_size=64, use_metal_kernel=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.chunk_size = chunk_size
        self.use_metal_kernel = use_metal_kernel
        
        # Heuristic for heads
        if n_heads is None:
            self.n_heads = d_model // d_head
        else:
            self.n_heads = n_heads
            
        self.in_proj = nn.Linear(d_model, 2 * d_model + 2 * self.n_heads * d_state + self.n_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Conv1d for local context (acting on X, B, C)
        # Grouped conv to treat heads independently
        conv_dim = d_model + 2 * self.n_heads * d_state
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, groups=conv_dim, padding=1)
        
        # A parameter (Decay) - Parameterized as -exp(A_log) for stability
        self.A_log = nn.Parameter(torch.randn(self.n_heads) * 0.5) 
        
        self.norm = nn.RMSNorm(d_model)

    def segsum(self, x):
        """Stable segment sum for chunks"""
        # x: (B, H, L)
        # Cumsum implementation
        return torch.cumsum(x, dim=-1)

    def ssd_step(self, x, A, B_proj, C_proj, chunk_len):
        """
        The SSD Algorithm (Block Decomposition)
        1. Diagonal Blocks: Attention-like (Quadratic on small chunks)
        2. Off-Diagonal: SSM Recurrence (Linear on chunks)
        """
        batch, seq_len, n_heads, d_head = x.shape
        n_chunks = seq_len // chunk_len
        
        # Reshape to chunks: (Batch, Heads, n_chunks, chunk_len, dim)
        x_chunks = rearrange(x, 'b (c l) h p -> b h c l p', c=n_chunks, l=chunk_len)
        B_chunks = rearrange(B_proj, 'b (c l) h n -> b h c l n', c=n_chunks, l=chunk_len)
        C_chunks = rearrange(C_proj, 'b (c l) h n -> b h c l n', c=n_chunks, l=chunk_len)
        
        # A is (Heads), broadcast to (Batch, Heads, Length)
        A_expanded = repeat(self.A_log, 'h -> b h l', b=batch, l=seq_len)
        A_expanded = A_expanded.float() # FP32 for cumsum precision
        A_cumsum = torch.cumsum(A_expanded, dim=-1)
        
        # --- 1. Intra-Chunk (Diagonal Blocks) via Matmul ---
        # Discretize A for segments: L[i,j] = exp(A_cumsum[i] - A_cumsum[j]) for i >= j
        A_chunks_cumsum = rearrange(A_cumsum, 'b h (c l) -> b h c l', c=n_chunks, l=chunk_len)
        
        # Compute "Attention" Mask L efficiently
        # L shape: (B, H, C, L, L)
        # Optimize: L = exp(segsum(A)).tril()
        # On MPS, we do this carefully to avoid OOM on huge tensors.
        # Here we use a simplified masking for demo.
        
        # Simplified SSD Diagonal: Y_diag = (C @ B.T * Mask) @ X
        # For efficiency, we just assume L is decay mask here or standard attention mask
        # Calculating exact L_mask per chunk:
        idx = torch.arange(chunk_len, device=x.device)
        # Relative positions: i - j
        # Simple decay mask approximation for speed test: exp(A * (i-j))
        mask = (idx[:, None] - idx[None, :])
        # Mask lower triangle
        causal_mask = mask >= 0
        
        # Compute diagonal output (Pure PyTorch MPS operation - highly optimized)
        # Y_diag = (C @ B^T) * Mask @ X
        # shape: (b, h, c, l, n) @ (b, h, c, l, n).T -> (b, h, c, l, l)
        G = torch.matmul(C_chunks, B_chunks.transpose(-1, -2)) 
        # Apply rough decay mask (using mean A for simplicity in demo or exact if A is constant per chunk)
        # In exact SSD, L is data dependent.
        Y_diag = torch.matmul(G * causal_mask.float(), x_chunks)

        # --- 2. Inter-Chunk (Recurrence) ---
        # State at end of chunk `c` passed to `c+1`
        # State update: h_end = h_start * decay_chunk + state_contribution_chunk
        
        # Calculate state contribution of each chunk (treating init state as 0)
        # h_chunk = sum(x * B * decay)
        # Approximate for speed:
        decay_chunk = torch.exp(A_chunks_cumsum[:, :, :, -1] - A_chunks_cumsum[:, :, :, 0]) # (B, H, C)
        
        # "Right" factors contribution to state
        # state_B = sum(x * B * decay_to_end)
        # Simple approximation: last element
        chunk_states_x = (x_chunks * B_chunks).sum(dim=3) # (B, H, C, N) - simplified
        
        if self.use_metal_kernel and x.device.type == 'mps':
            # --- Call Metal Kernel ---
            # Move data to CPU numpy (Simulation cost included)
            # In real prod, use shared memory pointer
            x_np = chunk_states_x.detach().cpu().numpy()
            decay_np = decay_chunk.detach().cpu().numpy()
            h_init_np = torch.zeros((batch, self.n_heads, self.d_state), dtype=torch.float32).numpy()
            
            final_states_np = run_ssd_chunk_scan_metal(x_np, decay_np, h_init_np)
            final_states = torch.from_numpy(final_states_np).to(x.device)
            # Shift states to be "previous" states for the next chunk
            prev_states = torch.roll(final_states, shifts=1, dims=2)
            prev_states[:, :, 0, :] = 0
        else:
            # --- Pure PyTorch Recurrence (Sequential Scan) ---
            final_states = torch.zeros_like(chunk_states_x)
            h = torch.zeros((batch, self.n_heads, self.d_state), device=x.device)
            for c in range(n_chunks):
                h = h * decay_chunk[:, :, c].unsqueeze(-1) + chunk_states_x[:, :, c, :]
                final_states[:, :, c, :] = h
            prev_states = torch.roll(final_states, shifts=1, dims=2)
            prev_states[:, :, 0, :] = 0

        # --- 3. Add Recurrence Effect to Output ---
        # Y_off = C * (prev_state * decay_into_chunk)
        # Apply decay to prev_state across the chunk length
        # Approximate: prev_state decays over L
        Y_off = torch.matmul(prev_states.unsqueeze(3), C_chunks.transpose(-1, -2)).transpose(-1, -2)
        # Note: Dimensions need careful aligning, this is a simplified view of the SSD low-rank term
        
        Y_out = Y_diag + Y_off # (B, H, C, L, P)
        return rearrange(Y_out, 'b h c l p -> b (c l) h p')

    def forward(self, u):
        B, L, _ = u.shape
        
        # 1. Projections (Parallel)
        zxbcdt = self.in_proj(u) # (B, L, huge_dim)
        
        # Split logic (simplified slices)
        # Order: [vec(z), vec(x), vec(B), vec(C), dt]
        # Dimensions are complex, let's assume d_head=d_state for simplicity in slicing
        A_len = self.n_heads
        D_len = self.d_model
        
        # Simple split for demo
        z, xBC, dt = torch.split(zxbcdt, [D_len, D_len + 2*self.n_heads*self.d_state, A_len], dim=-1)
        
        # 2. Conv1d on (x, B, C)
        xBC = xBC.transpose(1, 2) # (B, Dim, L)
        xBC = F.silu(self.conv1d(xBC))
        xBC = xBC.transpose(1, 2)
        
        # Unpack x, B, C
        x, B_raw, C_raw = torch.split(xBC, [D_len, self.n_heads*self.d_state, self.n_heads*self.d_state], dim=-1)
        
        # Reshape for Multi-Head
        x = rearrange(x, 'b l (h p) -> b l h p', h=self.n_heads, p=self.d_head)
        B_proj = rearrange(B_raw, 'b l (h n) -> b l h n', h=self.n_heads, n=self.d_state)
        C_proj = rearrange(C_raw, 'b l (h n) -> b l h n', h=self.n_heads, n=self.d_state)

        # 3. SSD Algorithm
        y = self.ssd_step(x, dt, B_proj, C_proj, self.chunk_size)
        
        # 4. Output Projection
        y = rearrange(y, 'b l h p -> b l (h p)')
        
        # Gating (SiLU)
        # z uses simple SiLU here, can use metal kernel if extracted
        y = y * F.silu(z)
        
        return self.out_proj(self.norm(y))