
import math
import mlx.core as mx
import mlx.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class Mamba2Mixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        A_init_range: tuple = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size

        # Input projection
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # Conv1d
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            stride=1,
            padding=d_conv - 1,
            groups=conv_dim,
            bias=True
        )

        # SSM Parameters
        # A: (nheads)
        self.A_log = mx.log(mx.random.uniform(low=A_init_range[0], high=A_init_range[1], shape=(self.nheads,)))
        
        # D: (nheads)
        self.D = mx.ones((self.nheads,))
        
        # dt bias
        dt = mx.exp(
            mx.random.uniform(low=math.log(dt_min), high=math.log(dt_max), shape=(self.nheads,))
        )
        dt = mx.maximum(dt, dt_init_floor)
        self.dt_bias = dt + mx.log(-mx.expm1(-dt))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Norm
        self.norm = RMSNorm(self.d_inner)

    def __call__(self, x, cache=None):
        # x: (B, L, D)
        B, L, D = x.shape
        
        zxbcdt = self.in_proj(x)
        
        # Split
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.ngroups * self.d_state - self.nheads) // 2
        
        z = zxbcdt[..., :self.d_inner]
        xBC = zxbcdt[..., self.d_inner : self.d_inner + self.d_inner + 2 * self.ngroups * self.d_state]
        dt = zxbcdt[..., -self.nheads:]
        
        if L > 1:
            # Prefill mode
            xBC_conv = self.conv1d(xBC)
            xBC_conv = xBC_conv[:, :L, :]
            xBC_conv = nn.silu(xBC_conv)
            
            # Update cache if present
            if cache is not None:
                conv_state = cache[0]
                K = self.d_conv
                
                xBC_t = xBC.transpose(0, 2, 1) # (B, C, L)
                
                combined = mx.concatenate([conv_state, xBC_t], axis=2)
                new_conv_state = combined[:, :, -K:]
                cache[0] = new_conv_state
                
        else:
            # Step mode (L=1)
            if cache is not None:
                # Update conv state
                conv_state = cache[0] # (B, C, K)
                xBC_step = xBC[:, 0, :]
                conv_state = mx.concatenate([conv_state[:, :, 1:], xBC_step[:, :, None]], axis=2)
                cache[0] = conv_state
                
                # Conv
                w = self.conv1d.weight.squeeze(-1) # (C, K)
                xBC_conv = mx.sum(conv_state * w, axis=2) + self.conv1d.bias
                xBC_conv = nn.silu(xBC_conv)
                
                # Expand to (B, 1, C) for consistency
                xBC_conv = xBC_conv[:, None, :]
            else:
                # Fallback for L=1 without cache
                xBC_conv = self.conv1d(xBC)
                xBC_conv = xBC_conv[:, :L, :]
                xBC_conv = nn.silu(xBC_conv)
            
        # Split xBC_conv
        x = xBC_conv[..., :self.d_inner]
        B_param = xBC_conv[..., self.d_inner : self.d_inner + self.ngroups * self.d_state]
        C_param = xBC_conv[..., self.d_inner + self.ngroups * self.d_state :]
        
        # SSM
        A = -mx.exp(self.A_log) # (H,)
        dt = nn.softplus(dt + self.dt_bias) # (B, L, H)
        
        # Reshape
        x = x.reshape(B, L, self.nheads, self.headdim)
        B_param = B_param.reshape(B, L, self.ngroups, self.d_state)
        C_param = C_param.reshape(B, L, self.ngroups, self.d_state)
        
        # Scan / Recurrence
        if L > 1:
            # Chunk Scan (Simplified)
            dA = mx.exp(dt * A)
            
            B_param = mx.repeat(B_param, self.nheads // self.ngroups, axis=2)
            C_param = mx.repeat(C_param, self.nheads // self.ngroups, axis=2)
            
            ys = []
            
            if cache is not None:
                h = cache[1]
            else:
                h = mx.zeros((B, self.nheads, self.d_state, self.headdim))
            
            for t in range(L):
                dt_t = dt[:, t, :, None, None] # (B, H, 1, 1)
                A_t = A[None, :, None, None] # (1, H, 1, 1)
                dA_t = mx.exp(dt_t[:, :, 0, 0] * A_t[:, :, 0, 0])[:, :, None, None] # (B, H, 1, 1)
                
                B_t = B_param[:, t, :, :, None] # (B, H, N, 1)
                x_t = x[:, t, :, None, :] # (B, H, 1, P)
                
                Bx_t = mx.matmul(B_t, x_t) # (B, H, N, P)
                
                h = h * dA_t + Bx_t * dt_t
                
                C_t = C_param[:, t, :, :, None] # (B, H, N, 1)
                y_t = mx.sum(C_t * h, axis=2) # (B, H, P)
                ys.append(y_t)
            
            if cache is not None:
                cache[1] = h
                
            y = mx.stack(ys, axis=1) # (B, L, H, P)
            
        else:
            # Step mode
            if cache is not None:
                ssm_state = cache[1] # (B, H, N, P)
                
                dt_t = dt[:, 0, :, None, None]
                A_t = A[None, :, None, None]
                dA_t = mx.exp(dt_t[:, :, 0, 0] * A_t[:, :, 0, 0])[:, :, None, None]
                
                B_param = mx.repeat(B_param, self.nheads // self.ngroups, axis=2)
                B_t = B_param[:, 0, :, :, None]
                
                x_t = x[:, 0, :, None, :]
                
                Bx_t = mx.matmul(B_t, x_t)
                
                h = ssm_state * dA_t + Bx_t * dt_t
                cache[1] = h
                
                C_param = mx.repeat(C_param, self.nheads // self.ngroups, axis=2)
                C_t = C_param[:, 0, :, :, None]
                
                y = mx.sum(C_t * h, axis=2) # (B, H, P)
                y = y[:, None, :, :] # (B, 1, H, P)
            else:
                 y = mx.zeros((B, L, self.nheads, self.headdim)) # Placeholder
            
        # D skip
        y = y + self.D[None, None, :, None] * x
        
        # Reshape
        y = y.reshape(B, L, self.d_inner)
        
        # Norm
        y = self.norm(y)
        
        # Gating
        y = y * nn.silu(z)
        
        # Out Proj
        out = self.out_proj(y)
        
        return out

class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mixer = Mamba2Mixer(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
        )
        
    def __call__(self, x, cache=None):
        residual = x
        x = self.norm(x)
        x = self.mixer(x, cache=cache)
        return residual + x
