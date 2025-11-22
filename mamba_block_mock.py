# mamba_mock.py
import os
import time
import numpy as np

try:
    import Metal
except Exception as e:
    Metal = None
    # We'll still allow pure-Python fallback behavior

class MambaBlockMock:
    """
    MambaBlockMock:
    - Tries to compile and use real Metal kernels if Metal is available and the .metal path is valid.
    - Otherwise falls back to a CPU-simulated pipeline (still follows same stages and shapes).
    """
    def __init__(self,
                 metal_path="src/metal/mamba_ssd.metal",
                 device=None,
                 pure_mock_if_no_metal=True):
        self.metal_path = os.path.abspath(metal_path)
        self.pure_mock = pure_mock_if_no_metal

        # Attempt to get Metal device
        if Metal is not None:
            try:
                self.device = device or Metal.MTLCreateSystemDefaultDevice()
            except Exception:
                self.device = None
        else:
            self.device = None

        if self.device is None:
            print("[MambaBlockMock] Metal not available or device not found.")
            if not self.pure_mock:
                raise RuntimeError("Metal required but not available.")
            print("[MambaBlockMock] Falling back to pure CPU mock mode.")
            self._metal_ready = False
        else:
            print(f"[MambaBlockMock] Using Metal device: {self.device.name()}")
            self._metal_ready = self._try_load_and_create_pipelines()

        # shape defaults (you can change before forward)
        self.d_model = 64
        self.d_state = 16
        self.n_groups = 32

    # --------------------------
    # Metal library / pipeline
    # --------------------------
    def _try_load_and_create_pipelines(self):
        try:
            with open(self.metal_path, "r") as f:
                source = f.read()
        except Exception as e:
            print(f"[MambaBlockMock] Warning: can't read metal file: {e}")
            return False

        try:
            opts = Metal.MTLCompileOptions.new()
            library, err = self.device.newLibraryWithSource_options_error_(source, opts, None)
            if err:
                print("[MambaBlockMock] Metal compile returned error:", err)
                return False
            self.lib = library
        except Exception as e:
            print("[MambaBlockMock] Exception compiling metal:", e)
            return False

        # helper to get pipeline (safe)
        def make_pipeline(name):
            try:
                fn = self.lib.newFunctionWithName_(name)
                if fn is None:
                    print(f"[MambaBlockMock] Kernel not found in lib: {name}")
                    return None
                p, perr = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if perr:
                    print(f"[MambaBlockMock] Pipeline error for {name}: {perr}")
                    return None
                return p
            except Exception as e:
                print(f"[MambaBlockMock] Exception creating pipeline {name}: {e}")
                return None

        # create pipelines
        self.p_rms = make_pipeline("rms_norm_kernel")
        self.p_conv = make_pipeline("depthwise_conv1d_kernel")
        self.p_scan = make_pipeline("selective_scan_update_kernel")
        self.p_gate = make_pipeline("gating_kernel")

        ok = any([self.p_rms, self.p_conv, self.p_scan, self.p_gate])
        if not ok:
            print("[MambaBlockMock] No usable pipeline created. Will run in CPU-mock mode.")
            return False

        print("[MambaBlockMock] Metal pipelines ready (some may be mocked if missing).")
        self.command_queue = self.device.newCommandQueue()
        return True

    # --------------------------
    # Helpers: buffer creation
    # --------------------------
    def _np_to_buffer(self, arr):
        """
        Convert numpy array to Metal buffer (shared storage) if Metal is available,
        otherwise return numpy array (for pure mock).
        """
        if self._metal_ready and self.device is not None:
            data = arr.astype(np.float32, copy=False, order='C')
            try:
                return self.device.newBufferWithBytes_length_options_(
                    data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
                )
            except Exception as e:
                print("[MambaBlockMock] newBuffer failed:", e)
                # fallback to CPU
                return data
        else:
            return arr.astype(np.float32, copy=True)

    def _read_buffer_to_numpy(self, buf, shape):
        """
        Read Metal buffer back to numpy. If pure mock, buf is already numpy.
        """
        if not self._metal_ready or not hasattr(buf, "contents"):
            # assume numpy
            return np.array(buf, dtype=np.float32).reshape(shape)
        # Metal buffer path: use contents() -> as_buffer(length)
        try:
            ptr = buf.contents().as_buffer(buf.length())
            arr = np.frombuffer(ptr, dtype=np.float32).copy()
            return arr.reshape(shape)
        except Exception as e:
            print("[MambaBlockMock] read buffer failed:", e)
            # return zeros to not crash
            return np.zeros(shape, dtype=np.float32)

    # --------------------------
    # Core forward
    # --------------------------
    def forward(self, x_np,
                dt=None, dt_bias=None, A=None, B=None, C=None, D=None, z=None,
                n_groups=None, do_rms=True, do_conv=True, do_ssm=True, do_gate=True):
        """
        x_np: (B, L, D_model)
        dt: (B, NHeads, Dim) or None (we'll broadcast/zero)
        A: (NHeads, Dim, DState) or None
        B, C: (B, NGroups, DState)
        D: (NHeads, Dim)
        z: (B, NHeads, Dim) gating branch (flattened layout expected in kernels)
        """
        BATCH, L, D_model = x_np.shape
        self.d_model = D_model
        n_groups = n_groups or self.n_groups

        # simple head dimension assumption for kernel shapes:
        # For the selective_scan kernel we expect layout (B, NHeads, Dim)
        # To avoid deep reshaping complexity here, assume NHeads = 1 for mock usage
        # If user needs multi-head, caller should supply properly shaped buffers.

        # ---- Prepare buffers ----
        # Flatten where kernel expects (Batch, NHeads, Dim) as contiguous array: size = B * nheads * dim
        # For simplicity we set nheads=1 and reshape accordingly if necessary
        nheads = 1
        if D_model % 1 != 0:
            raise ValueError("D_model must be int multiple for mock (nheads assumed 1).")

        # reshape x for selective_scan expected shape (B, NHeads, Dim) -> we take first timestep features
        # The selective_scan kernel in your code expects per-dim per-head scalar x (no sequence dimension).
        # Since earlier design applied SSM per timestep, if user passes sequence, we'll call scan for each time-step
        # For simplicity here we collapse sequence by taking mean along time for the selective_scan input (mock).
        x_collapsed = np.mean(x_np, axis=1).astype(np.float32)  # shape (B, D)

        # Prepare other params defaults if None
        if dt is None:
            dt = np.zeros_like(x_collapsed, dtype=np.float32)  # (B, D)
        else:
            dt = np.array(dt, dtype=np.float32)

        if dt_bias is None:
            dt_bias = np.zeros((nheads, D_model), dtype=np.float32)

        if A is None:
            # make small negative A for stability (A shape (nheads, D, dstate))
            A = -np.abs(np.random.randn(nheads, D_model, self.d_state).astype(np.float32))
        if B is None:
            B = np.random.randn(BATCH, n_groups, self.d_state).astype(np.float32) * 0.01
        if C is None:
            C = np.random.randn(BATCH, n_groups, self.d_state).astype(np.float32) * 0.01
        if D is None:
            D = np.ones((nheads, D_model), dtype=np.float32) * 0.0
        if z is None:
            z = np.random.randn(BATCH, nheads, D_model).astype(np.float32) * 0.01

        # Cast into buffers (or cpu arrays if no metal)
        buf_x = self._np_to_buffer(x_collapsed.flatten())
        buf_dt = self._np_to_buffer(dt.flatten())
        buf_dt_bias = self._np_to_buffer(dt_bias.flatten())
        buf_A = self._np_to_buffer(A.flatten())
        buf_B = self._np_to_buffer(B.flatten())
        buf_C = self._np_to_buffer(C.flatten())
        buf_D = self._np_to_buffer(D.flatten())
        buf_z = self._np_to_buffer(z.flatten())
        # out buffer for selective_scan: shape B * nheads * D_model
        out_shape = (BATCH, nheads, D_model)
        out_len = BATCH * nheads * D_model
        if self._metal_ready:
            buf_out = self.device.newBufferWithLength_options_(out_len * 4, Metal.MTLResourceStorageModeShared)
        else:
            buf_out = np.zeros((out_len,), dtype=np.float32)

        # ---- If Metal ready, encode commands ----
        start_t = time.time()
        if self._metal_ready:
            try:
                cmd_buf = self.command_queue.commandBuffer()
                # RMSNorm (if available and requested) -- note kernel expects shape (B, D) per batch row
                if do_rms and self.p_rms:
                    enc = cmd_buf.computeCommandEncoder()
                    enc.setComputePipelineState_(self.p_rms)
                    enc.setBuffer_offset_atIndex_(buf_x, 0, 0)       # input
                    # weight (use ones) -> create simple weight buffer
                    w = np.ones((D_model,), dtype=np.float32)
                    buf_w = self._np_to_buffer(w)
                    enc.setBuffer_offset_atIndex_(buf_w, 0, 1)
                    enc.setBuffer_offset_atIndex_(buf_out, 0, 2)
                    # set B, D as uint
                    b_val = np.uint32(BATCH)
                    d_val = np.uint32(D_model)
                    enc.setBytes_length_atIndex_(b_val.tobytes(), 4, 3)
                    enc.setBytes_length_atIndex_(d_val.tobytes(), 4, 4)
                    # dispatch: we use 1D grid over batch rows
                    grid = Metal.MTLSize(BATCH, 1, 1)
                    group = Metal.MTLSize(min(BATCH, 64), 1, 1)
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, group)
                    enc.endEncoding()

                # Depthwise conv (if available)
                if do_conv and self.p_conv:
                    enc = cmd_buf.computeCommandEncoder()
                    enc.setComputePipelineState_(self.p_conv)
                    # We need conv input layout (B, L, D). Use original x_np for conv.
                    buf_conv_in = self._np_to_buffer(x_np.flatten())
                    # conv weights: (D, K) where K=4
                    K = 4
                    conv_w = np.random.randn(D_model, K).astype(np.float32) * 0.01
                    buf_conv_w = self._np_to_buffer(conv_w.flatten())
                    buf_conv_b = self._np_to_buffer(np.zeros((D_model,), dtype=np.float32))
                    # output buffer for conv (B*L*D)
                    conv_out_len = BATCH * L * D_model
                    buf_conv_out = self.device.newBufferWithLength_options_(conv_out_len * 4, Metal.MTLResourceStorageModeShared)
                    enc.setBuffer_offset_atIndex_(buf_conv_in, 0, 0)
                    enc.setBuffer_offset_atIndex_(buf_conv_w, 0, 1)
                    enc.setBuffer_offset_atIndex_(buf_conv_b, 0, 2)
                    enc.setBuffer_offset_atIndex_(buf_conv_out, 0, 3)
                    # params: B, L, D, K
                    enc.setBytes_length_atIndex_(np.uint32(BATCH).tobytes(), 4, 4)
                    enc.setBytes_length_atIndex_(np.uint32(L).tobytes(), 4, 5)
                    enc.setBytes_length_atIndex_(np.uint32(D_model).tobytes(), 4, 6)
                    enc.setBytes_length_atIndex_(np.uint32(K).tobytes(), 4, 7)
                    # grid: (D, L, B)
                    grid = Metal.MTLSize(D_model, L, BATCH)
                    # choose a reasonable threadgroup
                    wth = self.p_conv.threadExecutionWidth() if hasattr(self.p_conv, "threadExecutionWidth") else 64
                    group = Metal.MTLSize(min(wth, 256), 1, 1)
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, group)
                    enc.endEncoding()
                else:
                    buf_conv_out = None

                # Selective scan (SSM) kernel
                if do_ssm and self.p_scan:
                    enc = cmd_buf.computeCommandEncoder()
                    enc.setComputePipelineState_(self.p_scan)
                    # state buffer: initialize small state array
                    state_len = BATCH * nheads * D_model * self.d_state
                    buf_state = self.device.newBufferWithLength_options_(state_len * 4, Metal.MTLResourceStorageModeShared)
                    # set required buffers per your kernel signature
                    enc.setBuffer_offset_atIndex_(buf_state, 0, 0)
                    enc.setBuffer_offset_atIndex_(buf_x, 0, 1)  # x collapsed
                    enc.setBuffer_offset_atIndex_(buf_dt, 0, 2)
                    enc.setBuffer_offset_atIndex_(buf_dt_bias, 0, 3)
                    enc.setBuffer_offset_atIndex_(buf_A, 0, 4)
                    enc.setBuffer_offset_atIndex_(buf_B, 0, 5)
                    enc.setBuffer_offset_atIndex_(buf_C, 0, 6)
                    enc.setBuffer_offset_atIndex_(buf_D, 0, 7)
                    enc.setBuffer_offset_atIndex_(buf_z, 0, 8)
                    enc.setBuffer_offset_atIndex_(buf_out, 0, 9)
                    # constant params (batch, nheads, dim, dstate, n_groups, has_z, has_dt_bias, has_D)
                    enc.setBytes_length_atIndex_(np.uint32(BATCH).tobytes(), 4, 10)
                    enc.setBytes_length_atIndex_(np.uint32(nheads).tobytes(), 4, 11)
                    enc.setBytes_length_atIndex_(np.uint32(D_model).tobytes(), 4, 12)
                    enc.setBytes_length_atIndex_(np.uint32(self.d_state).tobytes(), 4, 13)
                    enc.setBytes_length_atIndex_(np.uint32(n_groups).tobytes(), 4, 14)
                    enc.setBytes_length_atIndex_(np.uint32(1 if z is not None else 0).tobytes(), 4, 15)
                    enc.setBytes_length_atIndex_(np.uint32(1 if dt_bias is not None else 0).tobytes(), 4, 16)
                    enc.setBytes_length_atIndex_(np.uint32(1 if D is not None else 0).tobytes(), 4, 17)
                    # grid: (dim, nheads, batch)
                    grid = Metal.MTLSize(D_model, nheads, BATCH)
                    group = Metal.MTLSize(min(64, D_model), 1, 1)
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, group)
                    enc.endEncoding()

                # Gating
                if do_gate and self.p_gate:
                    enc = cmd_buf.computeCommandEncoder()
                    enc.setComputePipelineState_(self.p_gate)
                    # gating expects flattened y and z; we'll use buf_out and buf_z
                    enc.setBuffer_offset_atIndex_(buf_out, 0, 0)
                    enc.setBuffer_offset_atIndex_(buf_z, 0, 1)
                    # out buffer for gating (reuse buf_out)
                    enc.setBuffer_offset_atIndex_(buf_out, 0, 2)
                    # dispatch (1D over all elements)
                    total_elems = out_len
                    grid = Metal.MTLSize(total_elems, 1, 1)
                    group = Metal.MTLSize(min(512, total_elems), 1, 1)
                    # Some Metal versions require dispatchThreads vs dispatchThreadgroups; try dispatchThreadgroups
                    try:
                        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, group)
                    except Exception:
                        enc.dispatchThreads_threadsPerThreadgroup_(grid, group)
                    enc.endEncoding()

                # commit and wait
                cmd_buf.commit()
                cmd_buf.waitUntilCompleted()
                elapsed = (time.time() - start_t)
                # read out final data
                out_np = self._read_buffer_to_numpy(buf_out, out_shape)
                return out_np.reshape(BATCH, nheads * D_model) if out_np is not None else None
            except Exception as e:
                print("[MambaBlockMock] Exception during metal forward:", e)
                # fallback to CPU mock below
        # ---- Pure CPU fallback simulation (keeps same stages & shapes) ----
        elapsed = time.time() - start_t
        # 1) RMSNorm simulation: normalize per-batch/time-collapsed vector
        rms = np.sqrt(np.mean(x_collapsed**2, axis=-1, keepdims=True) + 1e-6)
        x_rms = (x_collapsed / rms) * 1.0  # weight = 1

        # 2) Depthwise conv simulation (causal) — operate on sequence
        conv_out = np.zeros_like(x_np)
        K = 4
        conv_w = np.random.randn(D_model, K).astype(np.float32) * 0.01
        for b in range(BATCH):
            for l in range(L):
                for d_idx in range(D_model):
                    acc = 0.0
                    for k in range(K):
                        t_idx = l - K + 1 + k
                        if t_idx >= 0:
                            acc += x_np[b, t_idx, d_idx] * conv_w[d_idx, k]
                    # SiLU
                    s = 1.0 / (1.0 + np.exp(-acc))
                    conv_out[b, l, d_idx] = acc * s

        # 3) Selective scan (SSM) simulation: simple exponential moving accumulate per collapsed x
        state = np.zeros((BATCH, nheads, D_model, self.d_state), dtype=np.float32)
        out_ssm = np.zeros((BATCH, nheads, D_model), dtype=np.float32)
        for b in range(BATCH):
            for h in range(nheads):
                for d_idx in range(D_model):
                    out_acc = 0.0
                    for s in range(self.d_state):
                        a = -abs(A[h, d_idx, s])
                        db = B[b, 0, s] if B.shape[1] > 0 else 0.0
                        dc = C[b, 0, s] if C.shape[1] > 0 else 0.0
                        prev = state[b, h, d_idx, s]
                        # simple decayed update
                        new_st = prev * np.exp(a * 0.1) + db * x_rms[b, d_idx]
                        state[b, h, d_idx, s] = new_st
                        out_acc += new_st * dc
                    # add D contribution
                    out_acc += x_rms[b, d_idx] * (D[h, d_idx] if D is not None else 0.0)
                    # gating z
                    if z is not None:
                        zv = z[b, h, d_idx]
                        s_z = 1.0 / (1.0 + np.exp(-zv))
                        out_acc *= (zv * s_z)
                    out_ssm[b, h, d_idx] = out_acc

        # 4) Gate (elementwise multiply) — already applied above in ssm step if z present
        final = out_ssm  # shape (B, nheads, D_model)
        elapsed = time.time() - start_t
        # return shape consistent with Metal path: (B, nheads, D_model)
        return final

# quick unit test when run directly
# if __name__ == "__main__":
#     x = np.random.randn(2, 8, 64).astype(np.float32)
#     mock = MambaBlockMock(metal_path="src/metal/mamba_ssd.metal")
#     out = mock.forward(x)
#     print("Out shape:", np.array(out).shape)
