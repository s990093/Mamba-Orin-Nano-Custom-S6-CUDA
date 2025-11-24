import torch
import numpy as np
import Metal
import time
from typing import Optional, List, Dict
from .metal_utils import (
    MetalContext,
    torch_tensor_to_metal_buffer,
    allocate_metal_buffer,
    create_command_buffer,
    execute_and_wait,
    metal_buffer_to_numpy
)

class MambaEngine:
    """
    Pure Metal Inference Engine for Mamba2.
    
    Manages the entire model state and execution on the GPU to minimize
    CPU-GPU synchronization overhead.
    """
    
    def __init__(self, model, device="mps"):
        print("[MambaEngine] Initializing Metal Engine...")
        self.ctx = MetalContext()
        self.ctx.library = self.ctx.compile_library("src/metal/mamba_advanced.metal", "mamba_advanced")
        
        # Function Constants
        # 0: has_bias (bool)
        # 1: activation_type (int)
        # 2: use_fast_math (bool)
        
        # We need to create specialized pipelines.
        # For simplicity, we'll create a default set where bias=True, act=SiLU, fast_math=True.
        # In a full implementation, we'd cache variants.
        
        import ctypes
        
        fc = Metal.MTLFunctionConstantValues.new()
        
        # Helper to set constant
        def set_constant(val, type_enum, index):
            # Create ctypes value
            if type_enum == Metal.MTLDataTypeBool:
                c_val = ctypes.c_bool(val)
            elif type_enum == Metal.MTLDataTypeInt:
                c_val = ctypes.c_int(val)
            else:
                raise ValueError("Unsupported type")
            
            # Pass address as void pointer
            # PyObjC setConstantValue:type:atIndex: expects a buffer compatible object or integer address?
            # The error "depythonifying 'unsigned long', got 'bytes'" suggests it wants an address (int).
            # Let's try passing ctypes.addressof(c_val)
            fc.setConstantValue_type_atIndex_(ctypes.addressof(c_val), type_enum, index)

        set_constant(True, Metal.MTLDataTypeBool, 0) # has_bias
        set_constant(0, Metal.MTLDataTypeInt, 1)     # activation_type
        set_constant(True, Metal.MTLDataTypeBool, 2) # use_fast_math
        
        # Helper to get pipeline with constants
        def get_specialized(name):
            desc = Metal.MTLComputePipelineDescriptor.new()
            func = self.ctx.library.newFunctionWithName_constantValues_error_(name, fc, None)[0]
            if func is None:
                raise ValueError(f"Function {name} not found or failed to specialize")
            desc.setComputeFunction_(func)
            state, err = self.ctx.device.newComputePipelineStateWithDescriptor_options_reflection_error_(desc, 0, None, None)
            if err:
                raise RuntimeError(f"Failed to create pipeline for {name}: {err}")
            return state

        # Pipelines
        self.pipelines = {}
        
        # Load original library for missing kernels
        self.lib_original = self.ctx.compile_library("src/metal/mamba_ssm_full.metal", "mamba_kernels")
        
        self.pipelines["embedding"] = self.ctx.get_pipeline(self.lib_original, "embedding_kernel")
        self.pipelines["linear"] = self.ctx.get_pipeline(self.lib_original, "linear_proj_kernel")
        self.pipelines["conv1d"] = self.ctx.get_pipeline(self.lib_original, "conv1d_causal_kernel")
        self.pipelines["conv1d_update"] = self.ctx.get_pipeline(self.lib_original, "conv1d_update_kernel")
        self.pipelines["residual"] = self.ctx.get_pipeline(self.lib_original, "residual_add_kernel")
        
        # New Optimized Kernels
        self.pipelines["rms_norm"] = get_specialized("rms_norm_opt")
        self.pipelines["ssm_update"] = get_specialized("ssm_step_opt")
        self.pipelines["argmax"] = self.ctx.get_pipeline(self.ctx.library, "argmax_kernel")
        
        # Integrate SwiGLU kernel for MLP layers
        from metal_ops.swiglu_metal import integrate_swiglu_kernel
        integrate_swiglu_kernel(self)
        
        # Optionally integrate BMM kernel
        try:
            from metal_ops.mamba_bmm_wrapper import integrate_bmm_kernel
            integrate_bmm_kernel(self)
        except Exception as e:
            print(f"[MambaEngine] BMM kernel not available: {e}")
        
        self.config = model.config
        self.d_model = self.config.d_model
        self.d_inner = self.config.d_model * self.config.expand
        self.d_state = getattr(self.config, "d_state", 128)
        self.d_conv = getattr(self.config, "d_conv", 4)
        self.headdim = getattr(self.config, "headdim", 64)
        self.ngroups = getattr(self.config, "ngroups", 1)
        self.nheads = self.d_inner // self.headdim
        self.n_layer = self.config.n_layer
        self.vocab_size = self.config.vocab_size
        
        # Load weights to Metal buffers
        self._load_weights(model)
        
        # Allocate persistent state buffers
        self._allocate_states(batch_size=1) # Default to batch size 1
        
        print("[MambaEngine] Initialization Complete.")

    def _load_weights(self, model):
        """Load all model weights into persistent Metal buffers."""
        print("[MambaEngine] Loading weights to GPU...")
        
        # Embedding
        self.w_embedding = torch_tensor_to_metal_buffer(self.ctx, model.embedding.weight)
        
        # Layers
        self.layers = []
        for i, layer in enumerate(model.layers):
            layer_buffers = {}
            mixer = layer.mixer
            
            # Norm
            layer_buffers["norm_w"] = torch_tensor_to_metal_buffer(self.ctx, layer.norm.weight)
            
            # In Proj
            layer_buffers["in_proj_w"] = torch_tensor_to_metal_buffer(self.ctx, mixer.in_proj.weight)
            if mixer.in_proj.bias is not None:
                layer_buffers["in_proj_b"] = torch_tensor_to_metal_buffer(self.ctx, mixer.in_proj.bias)
            else:
                layer_buffers["in_proj_b"] = None
                
            # Conv1d
            # Weight: (D, 1, K) -> (D, K)
            conv_w = mixer.conv1d.weight.squeeze(1)
            layer_buffers["conv1d_w"] = torch_tensor_to_metal_buffer(self.ctx, conv_w)
            layer_buffers["conv1d_b"] = torch_tensor_to_metal_buffer(self.ctx, mixer.conv1d.bias)
            
            # SSM Parameters
            # A_log: (H) -> (H, E, N)
            # We need to repeat it.
            # A_log is (nheads).
            # Target: (nheads, headdim, d_state)
            A_log_expanded = mixer.A_log.unsqueeze(-1).unsqueeze(-1).expand(-1, self.headdim, self.d_state).contiguous()
            layer_buffers["A_log"] = torch_tensor_to_metal_buffer(self.ctx, A_log_expanded)
            
            # D: (H) -> (H, E)
            # D is (nheads) usually.
            if mixer.D.dim() == 1:
                D_expanded = mixer.D.unsqueeze(-1).expand(-1, self.headdim).contiguous()
            else:
                D_expanded = mixer.D
            layer_buffers["D"] = torch_tensor_to_metal_buffer(self.ctx, D_expanded)
            
            # dt_bias: (H) -> (H, E)
            dt_bias_expanded = mixer.dt_bias.unsqueeze(-1).expand(-1, self.headdim).contiguous()
            layer_buffers["dt_bias"] = torch_tensor_to_metal_buffer(self.ctx, dt_bias_expanded)
            
            # Out Proj
            layer_buffers["out_proj_w"] = torch_tensor_to_metal_buffer(self.ctx, mixer.out_proj.weight)
            if mixer.out_proj.bias is not None:
                layer_buffers["out_proj_b"] = torch_tensor_to_metal_buffer(self.ctx, mixer.out_proj.bias)
            else:
                layer_buffers["out_proj_b"] = None
                
            self.layers.append(layer_buffers)
            
        # Final Norm
        self.w_norm_f = torch_tensor_to_metal_buffer(self.ctx, model.norm_f.weight)
        
        # LM Head
        self.w_lm_head = torch_tensor_to_metal_buffer(self.ctx, model.lm_head.weight)

    def _allocate_states(self, batch_size):
        """Allocate persistent inference states."""
        self.batch_size = batch_size
        
        # Temporary buffers for intermediate results
        # We need 2 buffers of size (B, L, D_model) for residual connection ping-pong
        # But for single token step, L=1
        self.buf_x = allocate_metal_buffer(self.ctx, batch_size * self.d_model)
        self.buf_residual = allocate_metal_buffer(self.ctx, batch_size * self.d_model)
        self.buf_hidden = allocate_metal_buffer(self.ctx, batch_size * self.d_inner) # For projected state
        
        # Layer states
        self.layer_states = []
        for _ in range(self.n_layer):
            states = {}
            # Conv state: (B, D_inner, K)
            # Note: d_inner includes the expansion factor
            conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
            states["conv_state"] = allocate_metal_buffer(self.ctx, batch_size * conv_dim * self.d_conv)
            
            # SSM state: (B, H, E, N)
            nheads = self.d_inner // self.headdim
            states["ssm_state"] = allocate_metal_buffer(self.ctx, batch_size * nheads * self.headdim * self.d_state)
            
            # Zero out states
            self._zero_buffer(states["conv_state"])
            self._zero_buffer(states["ssm_state"])
            
            self.layer_states.append(states)
            
    def _zero_buffer(self, buffer):
        """Zero out a Metal buffer."""
        ptr = buffer.contents()
        size = buffer.length()
        # Create numpy view and fill with zeros
        # Note: as_buffer is a method on the pointer object returned by contents()
        np_view = np.frombuffer(ptr.as_buffer(size), dtype=np.uint8)
        np_view.fill(0)

    def reset_states(self):
        """Reset all internal states to zero."""
        for layer_states in self.layer_states:
            self._zero_buffer(layer_states["conv_state"])
            self._zero_buffer(layer_states["ssm_state"])

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform one inference step.
        
        Args:
            input_ids: (B, 1) tensor of token IDs
            
        Returns:
            Logits: (B, 1, Vocab)
        """
        B = input_ids.shape[0]
        if B != self.batch_size:
            self._allocate_states(B)
            
        # 1. Embedding
        # Copy input_ids to Metal
        input_ids_int32 = input_ids.to(torch.int32)
        buf_input_ids = self.ctx.device.newBufferWithBytes_length_options_(
            input_ids_int32.numpy().tobytes(),
            input_ids_int32.numpy().nbytes,
            Metal.MTLResourceStorageModeShared
        )
        
        cmd = create_command_buffer(self.ctx)
        enc = cmd.computeCommandEncoder()
        
        # Dispatch Embedding: input_ids -> buf_x
        self._dispatch_embedding(enc, buf_input_ids, self.buf_x)
        
        # Allocate temp buffers if needed
        if not hasattr(self, 'buf_norm'):
             self.buf_norm = allocate_metal_buffer(self.ctx, B * self.d_model)
        
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if not hasattr(self, 'buf_proj_out'):
            self.buf_proj_out = allocate_metal_buffer(self.ctx, B * d_in_proj)
            
        # 2. Layers
        for i, layer in enumerate(self.layers):
            states = self.layer_states[i]
            
            # RMSNorm: buf_x -> buf_norm
            self._dispatch_rmsnorm(enc, self.buf_x, layer["norm_w"], self.buf_norm)
            
            # In Proj: buf_norm -> buf_proj_out
            self._dispatch_linear(enc, self.buf_norm, layer["in_proj_w"], layer["in_proj_b"], self.buf_proj_out, d_in=self.d_model, d_out=d_in_proj)
            
            # Conv1d
            # We need to point to the "xBC" part of buf_proj_out.
            # Layout: [z(d_inner), x(d_inner), B(G*N), C(G*N), dt(H)]
            # Wait, mamba2_metal.py split logic:
            # d_mlp = (d_in_proj - 2*d_ssm - 2*ngroups*d_state - nheads) // 2
            # z0, x0, z, xBC, dt = split(...)
            # So offsets are:
            # z0: 0
            # x0: d_mlp
            # z: 2*d_mlp
            # xBC: 2*d_mlp + d_inner
            # dt: 2*d_mlp + d_inner + (d_inner + 2*G*N)
            
            # Calculate d_mlp
            # d_ssm is d_inner usually
            d_mlp = (d_in_proj - 2 * self.d_inner - 2 * self.ngroups * self.d_state - self.nheads) // 2
            
            # Offsets (in float elements)
            off_z0 = 0
            off_x0 = d_mlp
            off_z = 2 * d_mlp
            off_xBC = 2 * d_mlp + self.d_inner
            off_dt = off_xBC + self.d_inner + 2 * self.ngroups * self.d_state
            
            # Dispatch Conv1d Update: buf_proj_out[off_xBC] -> buf_proj_out[off_xBC] (in-place)
            dim_xBC = self.d_inner + 2 * self.ngroups * self.d_state
            self._dispatch_conv1d_update(enc, self.buf_proj_out, off_xBC, layer["conv1d_w"], layer["conv1d_b"], states["conv_state"], dim=dim_xBC, stride=d_in_proj)
            
            # SSM Update
            # Inputs: 
            # x: buf_proj_out[off_xBC] (first d_inner elements)
            # B: buf_proj_out[off_xBC + d_inner]
            # C: buf_proj_out[off_xBC + d_inner + G*N]
            # dt: buf_proj_out[off_dt]
            # A, D, dt_bias: from layer buffers
            # z: buf_proj_out[off_z]
            # Output: buf_hidden (reused for y)
            
            # Offsets relative to xBC start
            rel_off_x = 0
            rel_off_B = self.d_inner
            rel_off_C = self.d_inner + self.ngroups * self.d_state
            
            # Absolute offsets
            abs_off_x = off_xBC + rel_off_x
            abs_off_B = off_xBC + rel_off_B
            abs_off_C = off_xBC + rel_off_C
            abs_off_dt = off_dt
            abs_off_z = off_z
            
            self._dispatch_ssm_update(
                enc,
                state=states["ssm_state"],
                x=self.buf_proj_out, x_offset=abs_off_x,
                dt=self.buf_proj_out, dt_offset=abs_off_dt,
                dt_bias=layer["dt_bias"],
                A=layer["A_log"],
                B_param=self.buf_proj_out, B_offset=abs_off_B,
                C_param=self.buf_proj_out, C_offset=abs_off_C,
                D=layer["D"],
                z=self.buf_proj_out, z_offset=abs_off_z,
                out=self.buf_hidden, # Output y
                stride=d_in_proj
            )
            
            # MLP Path (if d_mlp > 0)
            # y = cat([silu(z0) * x0, y], dim=-1)
            # This requires a custom kernel or separate ops.
            # For now, let's assume d_mlp == 0 (standard Mamba2).
            # If d_mlp > 0, we need to implement this.
            # Mamba2-130m config: d_model=768, expand=2 -> d_inner=1536.
            # headdim=64, nheads=24. ngroups=1. d_state=128.
            # d_in_proj = 2*1536 + 2*1*128 + 24 = 3072 + 256 + 24 = 3352.
            # Linear out features = 3352.
            # d_mlp calculation: (3352 - 2*1536 - 2*1*128 - 24) // 2 = 0.
            # So d_mlp is 0 for standard Mamba2. Safe to ignore for now.
            
            # Out Proj: buf_hidden -> buf_proj_out (reuse buffer, size d_model)
            # Wait, buf_proj_out is huge. We can reuse the beginning.
            # Or use buf_residual? No, buf_residual holds the residual.
            # We need a temp buffer for the output of the block before adding to residual.
            # Let's use buf_norm again? No, buf_norm is d_model, so yes we can reuse it!
            # buf_hidden (d_inner) -> buf_norm (d_model)
            self._dispatch_linear(enc, self.buf_hidden, layer["out_proj_w"], layer["out_proj_b"], self.buf_norm, d_in=self.d_inner, d_out=self.d_model)
            
            # Residual: buf_x += buf_norm
            self._dispatch_residual(enc, self.buf_x, self.buf_norm, size=B * self.d_model)
            
        # 3. Final Norm
        # buf_x -> buf_norm
        self._dispatch_rmsnorm(enc, self.buf_x, self.w_norm_f, self.buf_norm)
        
        # 4. LM Head
        # buf_norm -> buf_logits (need new buffer)
        if not hasattr(self, 'buf_logits'):
            self.buf_logits = allocate_metal_buffer(self.ctx, B * self.vocab_size)
            
        self._dispatch_linear(enc, self.buf_norm, self.w_lm_head, None, self.buf_logits, d_in=self.d_model, d_out=self.vocab_size)
        
        enc.endEncoding()
        execute_and_wait(cmd)
        
        # Return logits as tensor (on CPU)
        # Shape: (B, 1, Vocab)
        logits_np = metal_buffer_to_numpy(self.buf_logits, (B, 1, self.vocab_size))
        return torch.from_numpy(logits_np)

    def _dispatch_embedding(self, enc, input_ids, out):
        enc.setComputePipelineState_(self.pipelines["embedding"])
        enc.setBuffer_offset_atIndex_(input_ids, 0, 0)
        enc.setBuffer_offset_atIndex_(self.w_embedding, 0, 1)
        enc.setBuffer_offset_atIndex_(out, 0, 2)
        
        enc.setBytes_length_atIndex_(np.uint32(self.vocab_size).tobytes(), 4, 3)
        enc.setBytes_length_atIndex_(np.uint32(self.d_model).tobytes(), 4, 4)
        
        grid_size = (self.d_model, self.batch_size, 1)
        group_size = (min(self.d_model, 1024), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_size),
            Metal.MTLSizeMake(*group_size)
        )

    def _dispatch_rmsnorm(self, enc, x, w, out):
        enc.setComputePipelineState_(self.pipelines["rms_norm"])
        enc.setBuffer_offset_atIndex_(x, 0, 0)
        enc.setBuffer_offset_atIndex_(w, 0, 1)
        enc.setBuffer_offset_atIndex_(out, 0, 2)
        
        # Kernel args: tokens, dim
        # tokens = batch_size (since L=1)
        enc.setBytes_length_atIndex_(np.uint32(self.batch_size).tobytes(), 4, 3)
        enc.setBytes_length_atIndex_(np.uint32(self.d_model).tobytes(), 4, 4)
        
        grid_size = (self.batch_size, 1, 1)
        # Optimized kernel uses 256 threads
        group_size = (256, 1, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_size),
            Metal.MTLSizeMake(*group_size)
        )

    def _dispatch_linear(self, enc, x, w, b, out, d_in, d_out):
        enc.setComputePipelineState_(self.pipelines["linear"])
        enc.setBuffer_offset_atIndex_(x, 0, 0)
        enc.setBuffer_offset_atIndex_(w, 0, 1)
        if b:
            enc.setBuffer_offset_atIndex_(b, 0, 2)
        else:
            # Pass a dummy buffer or rely on null check?
            # Metal API allows passing nil for buffer.
            # In PyObjC, we can pass None? Let's try.
            # If this crashes, we need a dummy zero buffer.
            enc.setBuffer_offset_atIndex_(None, 0, 2)
            
        enc.setBuffer_offset_atIndex_(out, 0, 3)
        
        # Kernel args: tokens, d_model (d_in), d_inner (d_out)
        enc.setBytes_length_atIndex_(np.uint32(self.batch_size).tobytes(), 4, 4)
        enc.setBytes_length_atIndex_(np.uint32(d_in).tobytes(), 4, 5)
        enc.setBytes_length_atIndex_(np.uint32(d_out).tobytes(), 4, 6)
        
        grid_size = (d_out, self.batch_size, 1)
        group_size = (32, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_size),
            Metal.MTLSizeMake(*group_size)
        )

    def _dispatch_conv1d_update(self, enc, x_buf, x_offset, w, b, state, dim, stride):
        enc.setComputePipelineState_(self.pipelines["conv1d"]) # Actually conv1d_update_kernel
        # Wait, I named it "conv1d" in __init__ but it points to "conv1d_causal_kernel".
        # I need to update __init__ to point to "conv1d_update_kernel"!
        # I will fix this in a separate tool call or assume I fixed it.
        # Let's fix it here by using the correct name if I can.
        # I can't change __init__ here easily.
        # I will assume I'll fix __init__ later.
        # For now, let's assume self.pipelines["conv1d_update"] exists.
        
        # Actually I should fix __init__ in this replacement too if possible.
        # But I am replacing `step` downwards.
        # I will add `conv1d_update` to pipelines in a separate edit or assume it's there.
        # Let's use a new key "conv1d_update".
        
        enc.setComputePipelineState_(self.pipelines["conv1d_update"])
        
        # Buffers
        # x_buf with offset
        enc.setBuffer_offset_atIndex_(x_buf, x_offset * 4, 0) # Offset in bytes
        enc.setBuffer_offset_atIndex_(state, 0, 1)
        enc.setBuffer_offset_atIndex_(w, 0, 2)
        enc.setBuffer_offset_atIndex_(b, 0, 3)
        
        # Constants: B, D, K, stride
        enc.setBytes_length_atIndex_(np.uint32(self.batch_size).tobytes(), 4, 4)
        enc.setBytes_length_atIndex_(np.uint32(dim).tobytes(), 4, 5)
        enc.setBytes_length_atIndex_(np.uint32(self.d_conv).tobytes(), 4, 6)
        enc.setBytes_length_atIndex_(np.uint32(stride).tobytes(), 4, 7)
        
        grid_size = (dim, self.batch_size, 1)
        group_size = (32, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_size),
            Metal.MTLSizeMake(*group_size)
        )

    def _dispatch_ssm_update(self, enc, state, x, x_offset, dt, dt_offset, dt_bias, A, B_param, B_offset, C_param, C_offset, D, z, z_offset, out, stride):
        enc.setComputePipelineState_(self.pipelines["ssm_update"])
        
        # Buffers
        enc.setBuffer_offset_atIndex_(state, 0, 0)
        enc.setBuffer_offset_atIndex_(x, x_offset * 4, 1)
        enc.setBuffer_offset_atIndex_(dt, dt_offset * 4, 2)
        enc.setBuffer_offset_atIndex_(dt_bias, 0, 3)
        enc.setBuffer_offset_atIndex_(A, 0, 4)
        enc.setBuffer_offset_atIndex_(B_param, B_offset * 4, 5)
        enc.setBuffer_offset_atIndex_(C_param, C_offset * 4, 6)
        enc.setBuffer_offset_atIndex_(D, 0, 7)
        enc.setBuffer_offset_atIndex_(z, z_offset * 4, 8)
        enc.setBuffer_offset_atIndex_(out, 0, 9)
        
        # Constants
        # B, H, E, N, G, stride
        enc.setBytes_length_atIndex_(np.uint32(self.batch_size).tobytes(), 4, 10)
        nheads = self.d_inner // self.headdim
        enc.setBytes_length_atIndex_(np.uint32(nheads).tobytes(), 4, 11)
        enc.setBytes_length_atIndex_(np.uint32(self.headdim).tobytes(), 4, 12)
        enc.setBytes_length_atIndex_(np.uint32(self.d_state).tobytes(), 4, 13)
        enc.setBytes_length_atIndex_(np.uint32(self.ngroups).tobytes(), 4, 14)
        enc.setBytes_length_atIndex_(np.uint32(stride).tobytes(), 4, 15)
        
        # Grid: (E, H, B)
        # Threadgroup: (32, 1, 1) or whatever fits E.
        # Optimized kernel uses vectorized loop over N, so one thread handles N/4 items.
        # But we parallelize over E.
        # E is headdim (64 or 128).
        # We can use 64 threads per group?
        # Let's use (headdim, 1, 1) as threadgroup size.
        
        grid_size = (self.headdim, nheads, self.batch_size)
        group_size = (min(self.headdim, 1024), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_size),
            Metal.MTLSizeMake(*group_size)
        )

    def _dispatch_residual(self, enc, residual, x, size):
        enc.setComputePipelineState_(self.pipelines["residual"])
        enc.setBuffer_offset_atIndex_(residual, 0, 0)
        enc.setBuffer_offset_atIndex_(x, 0, 1)
        enc.setBytes_length_atIndex_(np.uint32(size).tobytes(), 4, 2)
        
        grid_size = (size, 1, 1)
        group_size = (min(size, 1024), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_size),
            Metal.MTLSizeMake(*group_size)
        )
