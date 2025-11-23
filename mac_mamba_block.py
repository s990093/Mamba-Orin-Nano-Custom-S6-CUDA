# mac_mamba_block.py  ← 更新版，新增 selective_state_update 支援單步前向傳播加速
import os
import numpy as np
import Metal

class MambaBlockMetal:
    def __init__(
        self,
        d_model=768,
        expand=2,
        headdim=64,
        d_state=128,
        ngroups=8,
        d_conv=4,
        rmsnorm=True,
        metal_path="src/metal/mamba_ssm_full.metal"
    ):
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        self.d_conv = d_conv
        self.rmsnorm = rmsnorm

        self.nheads = self.d_inner // headdim
        assert self.nheads % ngroups == 0

        self.device = Metal.MTLCreateSystemDefaultDevice()
        print(f"[MambaBlockMetal] Using device: {self.device.name()}")
        print(f"[MambaBlockMetal] d_model={d_model}, d_inner={self.d_inner}, nheads={self.nheads}, ngroups={ngroups}")

        # Compile Metal
        with open(os.path.abspath(metal_path)) as f:
            source = f.read()
        opts = Metal.MTLCompileOptions.new()
        library, err = self.device.newLibraryWithSource_options_error_(source, opts, None)
        if err: raise RuntimeError(err.localizedDescription())
        self.library = library
        self.queue = self.device.newCommandQueue()

        # Pipelines
        self.pipes = {}
        for name in ["rms_norm_kernel", "linear_proj_kernel", "conv1d_causal_kernel",
                     "ssm_scan_kernel", "gating_kernel", "selective_scan_update_kernel"]:
            fn = library.newFunctionWithName_(name)
            pipe, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err.localizedDescription())
            self.pipes[name] = pipe

        self._init_weights()
        
        # Inference cache
        self.inference_cache = None

    def _init_weights(self):
        Di = self.d_inner
        H, E, N, G = self.nheads, self.headdim, self.d_state, self.ngroups

        self.A_log = np.log(np.random.uniform(1, 16, (H, E, N))).astype('f4')
        self.x_proj_w = np.random.randn(Di, self.d_model).astype('f4') * 0.02
        self.z_proj_w = np.random.randn(Di, self.d_model).astype('f4') * 0.02
        self.out_proj_w = np.random.randn(self.d_model, Di).astype('f4') * 0.02
        self.conv_w = np.random.randn(Di, self.d_conv).astype('f4') * 0.02
        self.conv_b = np.random.randn(Di).astype('f4') * 0.01
        self.dt_proj_w = np.random.randn(Di, Di).astype('f4') * 0.02  # simplified
        self.dt_proj_b = np.random.randn(Di).astype('f4') * 0.01
        self.B_proj_w = np.random.randn(G*N, Di).astype('f4') * 0.02
        self.C_proj_w = np.random.randn(G*N, Di).astype('f4') * 0.02
        self.D = np.zeros((H, E), 'f4')
        self.norm_w = np.ones(self.d_model, 'f4')

    def _buf(self, arr):
        if arr is None: return None
        arr = np.ascontiguousarray(arr, 'f4')
        return self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared)

    def forward(self, x_np):
        B, L, _ = x_np.shape
        Di = self.d_inner
        H, E, N, G = self.nheads, self.headdim, self.d_state, self.ngroups

        x = self._buf(x_np)

        # Parameters
        A_log = self._buf(self.A_log)
        D = self._buf(self.D)
        x_w = self._buf(self.x_proj_w.T)
        z_w = self._buf(self.z_proj_w.T)
        out_w = self._buf(self.out_proj_w.T)
        conv_w = self._buf(self.conv_w)
        conv_b = self._buf(self.conv_b)
        dt_w = self._buf(self.dt_proj_w.T)
        dt_b = self._buf(self.dt_proj_b)
        B_w = self._buf(self.B_proj_w.T)
        C_w = self._buf(self.C_proj_w.T)
        norm_w = self._buf(self.norm_w)

        # Buffers (關鍵：數字在前！)
        normed    = self.device.newBufferWithLength_options_(B*L*self.d_model*4,  Metal.MTLResourceStorageModeShared)
        xb        = self.device.newBufferWithLength_options_(B*L*Di*4,            Metal.MTLResourceStorageModeShared)
        zb        = self.device.newBufferWithLength_options_(B*L*Di*4,            Metal.MTLResourceStorageModeShared)
        conv_out  = self.device.newBufferWithLength_options_(B*L*Di*4,            Metal.MTLResourceStorageModeShared)
        dt        = self.device.newBufferWithLength_options_(B*L*Di*4,            Metal.MTLResourceStorageModeShared)
        B_proj    = self.device.newBufferWithLength_options_(B*L*G*N*4,           Metal.MTLResourceStorageModeShared)
        C_proj    = self.device.newBufferWithLength_options_(B*L*G*N*4,           Metal.MTLResourceStorageModeShared)
        ssm_out   = self.device.newBufferWithLength_options_(B*L*Di*4,            Metal.MTLResourceStorageModeShared)
        state     = self.device.newBufferWithLength_options_(B*H*E*N*4,           Metal.MTLResourceStorageModeShared)
        out_buf   = self.device.newBufferWithLength_options_(B*L*self.d_model*4, Metal.MTLResourceStorageModeShared)

        cmd = self.queue.commandBuffer()

        # 1. RMSNorm
        if self.rmsnorm:
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipes["rms_norm_kernel"])
            enc.setBuffer_offset_atIndex_(x, 0, 0)
            enc.setBuffer_offset_atIndex_(norm_w, 0, 1)
            enc.setBuffer_offset_atIndex_(normed, 0, 2)
            enc.setBytes_length_atIndex_(np.uint32(B*L).tobytes(), 4, 3)
            enc.setBytes_length_atIndex_(np.uint32(self.d_model).tobytes(), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(B*L,1,1), Metal.MTLSizeMake(256,1,1))
            enc.endEncoding()
            x_src = normed
        else:
            x_src = x

        # 2. Linear proj
        def linear(src, w, dst):
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipes["linear_proj_kernel"])
            enc.setBuffer_offset_atIndex_(src, 0, 0)
            enc.setBuffer_offset_atIndex_(w, 0, 1)
            enc.setBuffer_offset_atIndex_(None, 0, 2)
            enc.setBuffer_offset_atIndex_(dst, 0, 3)
            enc.setBytes_length_atIndex_(np.uint32(B*L).tobytes(), 4, 4)
            enc.setBytes_length_atIndex_(np.uint32(self.d_model).tobytes(), 4, 5)
            enc.setBytes_length_atIndex_(np.uint32(Di).tobytes(), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(Di, B*L, 1), Metal.MTLSizeMake(64,1,1))
            enc.endEncoding()

        linear(x_src, x_w, xb)
        linear(x_src, z_w, zb)

        # 3. Conv
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipes["conv1d_causal_kernel"])
        enc.setBuffer_offset_atIndex_(xb, 0, 0)
        enc.setBuffer_offset_atIndex_(conv_w, 0, 1)
        enc.setBuffer_offset_atIndex_(conv_b, 0, 2)
        enc.setBuffer_offset_atIndex_(conv_out, 0, 3)
        for i, v in enumerate([B, L, Di, self.d_conv], 4):
            enc.setBytes_length_atIndex_(np.uint32(v).tobytes(), 4, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(Di, L, B), Metal.MTLSizeMake(32,4,1))
        enc.endEncoding()

        # 4. Project dt/B/C (simplified)
        linear(conv_out, dt_w, dt)
        linear(conv_out, B_w, B_proj)
        linear(conv_out, C_w, C_proj)

        # 5. SSM
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipes["ssm_scan_kernel"])
        for i, buf in enumerate([state, conv_out, dt, dt_b, A_log, B_proj, C_proj, D, ssm_out]):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        for i, val in enumerate([B, L, H, E, N, G, 1], 10):
            enc.setBytes_length_atIndex_(np.uint32(val).tobytes(), 4, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(E, H, B), Metal.MTLSizeMake(8,8,1))
        enc.endEncoding()

        # 6. Gating
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipes["gating_kernel"])
        enc.setBuffer_offset_atIndex_(ssm_out, 0, 0)
        enc.setBuffer_offset_atIndex_(zb, 0, 1)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((B*L*Di+511)//512,1,1), Metal.MTLSizeMake(512,1,1))
        enc.endEncoding()

        # 7. Out proj
        linear(out_buf, out_w, out_buf)

        cmd.commit()
        cmd.waitUntilCompleted()

        result = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), np.float32)
        return result.reshape(B, L, self.d_model).copy() + x_np

    def allocate_inference_cache(self, batch_size):
        """
        Pre-allocate GPU buffers for inference to avoid re-allocation overhead.
        """
        B = batch_size
        H, E, N, G = self.nheads, self.headdim, self.d_state, self.ngroups
        
        self.inference_cache = {
            "state": self.device.newBufferWithLength_options_(B*H*E*N*4, Metal.MTLResourceStorageModeShared),
            "x": self.device.newBufferWithLength_options_(B*H*E*4, Metal.MTLResourceStorageModeShared),
            "dt": self.device.newBufferWithLength_options_(B*H*E*4, Metal.MTLResourceStorageModeShared),
            "B": self.device.newBufferWithLength_options_(B*G*N*4, Metal.MTLResourceStorageModeShared),
            "C": self.device.newBufferWithLength_options_(B*G*N*4, Metal.MTLResourceStorageModeShared),
            "z": self.device.newBufferWithLength_options_(B*H*E*4, Metal.MTLResourceStorageModeShared),
            "out": self.device.newBufferWithLength_options_(B*H*E*4, Metal.MTLResourceStorageModeShared),
            # Pre-load constant weights
            "A_log": self._buf(self.A_log),
            "D": self._buf(self.D),
            "dt_bias": None # We handle bias in python or kernel? Kernel has dt_bias input.
        }
        
        # Initialize state to zeros
        ptr = self.inference_cache["state"].contents()
        np_ptr = np.frombuffer(ptr.as_buffer(self.inference_cache["state"].length()), dtype='f4')
        np_ptr[:] = 0

    def step_inference(self, x_np, dt_np, B_np, C_np, z_np=None, dt_softplus=True):
        """
        Fast inference step using pre-allocated buffers.
        Does NOT wait for completion.
        Returns the output buffer.
        """
        if self.inference_cache is None:
            raise RuntimeError("Inference cache not initialized. Call allocate_inference_cache first.")
            
        cache = self.inference_cache
        B, H, E = x_np.shape
        N = self.d_state
        G = self.ngroups
        
        # Copy inputs to GPU buffers
        # This is still a sync copy (CPU->Shared Buffer), but avoids allocation
        # x
        ptr = cache["x"].contents()
        np.frombuffer(ptr.as_buffer(cache["x"].length()), dtype='f4')[:] = x_np.flatten()
        
        # dt
        ptr = cache["dt"].contents()
        np.frombuffer(ptr.as_buffer(cache["dt"].length()), dtype='f4')[:] = dt_np.flatten()
        
        # B
        ptr = cache["B"].contents()
        np.frombuffer(ptr.as_buffer(cache["B"].length()), dtype='f4')[:] = B_np.flatten()
        
        # C
        ptr = cache["C"].contents()
        np.frombuffer(ptr.as_buffer(cache["C"].length()), dtype='f4')[:] = C_np.flatten()
        
        # z
        if z_np is not None:
            ptr = cache["z"].contents()
            np.frombuffer(ptr.as_buffer(cache["z"].length()), dtype='f4')[:] = z_np.flatten()
            
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipes["selective_scan_update_kernel"])
        
        # Set buffers
        enc.setBuffer_offset_atIndex_(cache["state"], 0, 0)
        enc.setBuffer_offset_atIndex_(cache["x"], 0, 1)
        enc.setBuffer_offset_atIndex_(cache["dt"], 0, 2)
        enc.setBuffer_offset_atIndex_(None, 0, 3) # dt_bias (assumed baked in or handled)
        enc.setBuffer_offset_atIndex_(cache["A_log"], 0, 4)
        enc.setBuffer_offset_atIndex_(cache["B"], 0, 5)
        enc.setBuffer_offset_atIndex_(cache["C"], 0, 6)
        enc.setBuffer_offset_atIndex_(cache["D"], 0, 7)
        enc.setBuffer_offset_atIndex_(cache["z"] if z_np is not None else None, 0, 8)
        enc.setBuffer_offset_atIndex_(cache["out"], 0, 9)
        
        # Set constants
        for i, val in enumerate([B, H, E, N, G, dt_softplus, False], 10):
            enc.setBytes_length_atIndex_(np.uint32(val).tobytes(), 4, i)
            
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(E, H, B), Metal.MTLSizeMake(8, 8, 1))
        enc.endEncoding()
        
        cmd.commit()
        # NO waitUntilCompleted()
        
        return cache["out"]

    def selective_state_update(self, state_np, x_np, dt_np, B_np, C_np, z_np=None, dt_bias_np=None, dt_softplus=True):
        """
        Legacy slow method for reference or testing.
        """
        B, H, E = x_np.shape
        N = state_np.shape[3]
        G = self.ngroups
        
        state = self._buf(state_np)
        x = self._buf(x_np)
        dt = self._buf(dt_np)
        dt_bias = self._buf(dt_bias_np)
        A_log = self._buf(self.A_log)
        B_proj = self._buf(B_np)
        C_proj = self._buf(C_np)
        D = self._buf(self.D)
        z = self._buf(z_np)
        out_buf = self.device.newBufferWithLength_options_(B * H * E * 4, Metal.MTLResourceStorageModeShared)

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipes["selective_scan_update_kernel"])
        for i, buf in enumerate([state, x, dt, dt_bias, A_log, B_proj, C_proj, D, z, out_buf]):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        for i, val in enumerate([B, H, E, N, G, dt_softplus, False], 10):
            enc.setBytes_length_atIndex_(np.uint32(val).tobytes(), 4, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(E, H, B), Metal.MTLSizeMake(8, 8, 1))
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        result = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), np.float32)
        new_state = np.frombuffer(state.contents().as_buffer(state.length()), np.float32)
        return result.reshape(B, H, E).copy(), new_state.reshape(B, H, E, N).copy()

# ================ TEST ================
if __name__ == "__main__":
    block = MambaBlockMetal()
    x = np.random.randn(2, 1024, 768).astype('f4')
    y = block.forward(x)
    print("FINAL SUCCESS! Output shape:", y.shape)
    
    # Test Fast Inference
    print("\nTesting Fast Inference...")
    block.allocate_inference_cache(batch_size=2)
    
    B, H, E, N, G = 2, block.nheads, block.headdim, block.d_state, block.ngroups
    x_step = np.random.randn(B, H, E).astype('f4')
    dt_step = np.random.randn(B, H, E).astype('f4')
    B_step = np.random.randn(B, G, N).astype('f4')
    C_step = np.random.randn(B, G, N).astype('f4')
    z_step = np.random.randn(B, H, E).astype('f4')
    
    out_buf = block.step_inference(x_step, dt_step, B_step, C_step, z_step)
    
    # Wait for result just for testing
    block.queue.commandBuffer().waitUntilCompleted() # Wait for previous commands? No, need to wait on the specific command buffer.
    # But step_inference doesn't return the command buffer.
    # In real usage we don't wait.
    # To check result, we can just read the buffer (it might be stale if we don't wait, but for test we can wait on device)
    
    # Hack to wait: submit empty command buffer and wait
    cmd = block.queue.commandBuffer()
    cmd.commit()
    cmd.waitUntilCompleted()
    
    result = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype='f4')
    print("Fast Inference Output shape:", result.shape)
    print("Output sample:", result[:5])