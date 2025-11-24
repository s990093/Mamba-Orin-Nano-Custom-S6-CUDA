# Mamba2 Metal Performance Optimization Tasks

## Overview
This document outlines concrete tasks to optimize Mamba2 Metal inference by adopting chunked/tiled/fused kernel strategies from Triton implementations. The goal is to reduce per-token dispatch overhead and maximize GPU utilization through larger tile-based kernels.

**Target Performance**: 2-5x throughput improvement over current baseline (~50 tokens/s for 130M, ~11 tokens/s for 780M).

## Priority Levels
- **P0 (Critical)**: Must-have for significant performance gains
- **P1 (High)**: High impact, should be done soon
- **P2 (Medium)**: Nice to have, polish

---

## P0 Tasks

### Task 1: Command Buffer Reuse & Offset Precomputation
**Priority**: P0  
**Files**: `metal_ops/mamba_engine.py`  
**Estimated Impact**: 10-20% latency reduction, reduced jitter

#### Changes
- **Location**: `MambaEngine.__init__`
  - Add `self.persistent_cmd_pool` or reusable command buffer
  - Precompute all layer offsets (`off_z0`, `off_x0`, `off_z`, `off_xBC`, `off_dt`) 
  - Store in `self.layer_meta[layer_idx]` dict

- **Location**: `MambaEngine.step`
  - Remove per-token `create_command_buffer()` calls
  - Reuse command buffer/encoder across dispatches
  - Batch all kernel dispatches, commit once at end

#### Success Criteria
- [ ] Command buffer commits reduced from N (per-kernel) to 1-2 (per-token)
- [ ] Latency std deviation drops by 50%+
- [ ] No visible allocation spikes in Xcode GPU capture

---

### Task 2: Eliminate Per-Token Buffer Allocations
**Priority**: P0  
**Files**: `metal_ops/mamba_engine.py`  
**Estimated Impact**: 15-25% throughput improvement

#### Changes
- **Location**: `MambaEngine._allocate_states`
  - Move ALL buffer allocations here (called once in `__init__`)
  - Allocate: `buf_logits`, `buf_proj_out`, `buf_norm`, `buf_hidden`, `buf_residual`
  - Use page-aligned sizes (multiple of 4096 bytes)

- **Location**: `MambaEngine.step`
  - Remove all `if not hasattr(self, 'buf_*'):` checks
  - Remove all `device.newBufferWithBytes_length_options_` calls
  - Only exception: `input_ids` buffer (small, per-batch)

#### Success Criteria
- [ ] Zero buffer allocations after warm-up (verify in profiler)
- [ ] Memory usage stable across 1000+ tokens
- [ ] No `hasattr` checks in hot path

---

### Task 3: Tiled Block Matrix Multiply (Chunked BMM)
**Priority**: P0  
**Files**: `src/metal/mamba_bmm.metal` (new), `metal_ops/mamba_engine.py`  
**Estimated Impact**: 2-3x speedup for matmul-heavy layers

#### Kernel Requirements
Create `_bmm_chunk_fwd_kernel` in `mamba_bmm.metal`:
- **Tile sizes**: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 (function constants)
- **Support**: Grouped heads (ngroups parameter)
- **Support**: Optional causal masking (IS_CAUSAL constant)
- **Optimization**: Use `threadgroup` memory for tile caching
- **Layout**: Assume row-major for A, B; write row-major C

#### Python Integration
```python
# In MambaEngine.__init__
self.lib_bmm = self.ctx.compile_library("src/metal/mamba_bmm.metal", "mamba_bmm")
self.pipelines["bmm_chunk_fwd"] = self.ctx.get_pipeline(self.lib_bmm, "bmm_chunk_fwd_kernel")

# New method
def _dispatch_bmm_chunk(self, enc, a_buf, b_buf, out_buf, M, N, K, 
                         batch, ngroups, stride_a, stride_b, stride_out):
    # Set buffers and dispatch tiled BMM
    pass
```

#### Success Criteria
- [ ] Microbenchmark: 2x faster than naive matmul for 1024x1024 @ 512
- [ ] Numerical correctness: max abs error < 1e-3 vs reference
- [ ] Integrated into `step()` for in_proj/out_proj linear layers

---

## P1 Tasks

### Task 4: Chunked Cumsum / DT Accumulation Kernel
**Priority**: P1  
**Files**: `src/metal/mamba_chunk_cumsum.metal` (new)  
**Estimated Impact**: 20-30% SSM computation speedup

#### Kernel Specification
Create `_chunk_cumsum_fwd_kernel`:
- **Input**: `dt` (B, L, nheads), `A_log` (nheads, headdim, d_state), `dt_bias` (nheads, headdim)
- **Output**: `dt_out` (processed dt), `dA_cumsum` (cumulative dA for SSM)
- **Operations**: 
  - Apply softplus to dt if `DT_SOFTPLUS` constant is true
  - Add dt_bias if `HAS_DT_BIAS` is true
  - Compute cumulative sum: `dA[i] = sum(exp(-A * dt[j]) for j <= i)`
  
#### Integration
- Replace manual dt processing in Python
- Feed `dA_cumsum` directly to SSM kernel

#### Success Criteria
- [ ] Matches CPU reference implementation (tolerance 1e-4)
- [ ] Faster than sequential CPU loop by 10x+
- [ ] No numerical stability issues (check with long sequences)

---

### Task 5: Fused Conv1d + SSM + Gating Kernel
**Priority**: P1  
**Files**: `src/metal/mamba_fused_block.metal` (new)  
**Estimated Impact**: 30-50% per-layer speedup

#### Kernel Fusion Strategy
Combine these operations into ONE kernel:
1. `conv1d_update`: Causal convolution state update
2. `ssm_update`: Selective state space update
3. `silu(z) * y`: Gating operation
4. `out_proj`: Output linear projection
5. `residual_add`: Add skip connection

#### Implementation Plan
- Use `threadgroup` memory to cache xBC tile
- Load conv weights, SSM params (A, B, C, D) once per tile
- Compute sequentially within threadgroup
- Write final output directly to `buf_x` (next layer input)

#### Success Criteria
- [ ] 30%+ faster than separate kernel dispatches
- [ ] Correctness: outputs match reference within 1e-3
- [ ] Memory traffic reduced (verify in GPU profiler)

---

## P2 Tasks

### Task 6: Autotune Parameters
**Priority**: P2  
**Files**: `metal_ops/mamba_engine.py`, `configs/metal_tune.json` (new)

#### Approach
- Create multiple pipeline variants with different BLOCK sizes
- Run microbenchmarks on target hardware
- Store optimal configs in JSON
- Load appropriate variant at runtime based on model size

#### Success Criteria
- [ ] 5-10% additional speedup from optimal tile sizes
- [ ] Config selection automated

---

### Task 7: MTLHeap Buffer Management
**Priority**: P2  
**Files**: `metal_ops/metal_utils.py`

#### Changes
- Allocate single large `MTLHeap` for all weights + states
- Sub-allocate buffers from heap with proper alignment
- Reduce memory fragmentation

#### Success Criteria
- [ ] Total allocated memory reduced by 10%+
- [ ] Allocation time negligible

---

### Task 8: Benchmark & Testing Infrastructure
**Priority**: P2  
**Files**: `benchmarks/metal/` (new), `tests/metal/` (new)

#### Deliverables
- `bench_infer.py`: Automated throughput/latency measurement
- `test_kernels.py`: Unit tests for each kernel vs CPU reference
- CI integration for regression testing

#### Success Criteria
- [ ] All kernels have unit tests
- [ ] Benchmarks run in CI
- [ ] Performance regression alerts

---

## Implementation Roadmap

### Week 1
- [ ] Task 1: Command buffer reuse
- [ ] Task 2: Eliminate allocations
- [ ] Task 3: Tiled BMM kernel (skeleton)

### Week 2
- [ ] Task 3: BMM integration & tuning
- [ ] Task 4: Chunked cumsum kernel

### Week 3
- [ ] Task 5: Fused block kernel
- [ ] Task 8: Testing infrastructure

### Week 4
- [ ] Task 6: Autotuning
- [ ] Task 7: Heap management
- [ ] Polish & documentation

---

## Code Modification Map

### `MambaEngine.__init__`
```python
# Add persistent resources
self.persistent_cmd = self.ctx.queue.commandBuffer()
self.layer_meta = []  # Precomputed offsets per layer

# Precompute ALL offsets once
for layer_idx in range(self.n_layer):
    d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
    d_mlp = (d_in_proj - 2 * self.d_inner - 2 * self.ngroups * self.d_state - self.nheads) // 2
    
    self.layer_meta.append({
        'off_z0': 0,
        'off_x0': d_mlp,
        'off_z': 2 * d_mlp,
        'off_xBC': 2 * d_mlp + self.d_inner,
        'off_dt': 2 * d_mlp + self.d_inner + self.d_inner + 2 * self.ngroups * self.d_state,
        'd_in_proj': d_in_proj
    })
```

### `MambaEngine.step`
```python
def step(self, input_ids):
    # Single command buffer for ALL operations
    cmd = self.persistent_cmd
    enc = cmd.computeCommandEncoder()
    
    # Embedding
    self._dispatch_embedding(enc, ...)
    
    # Process all layers
    for i, layer in enumerate(self.layers):
        meta = self.layer_meta[i]
        
        # Option A: Use fused kernel
        self._dispatch_fused_block(enc, layer, meta, ...)
        
        # Option B: Use separate optimized kernels
        # self._dispatch_bmm_chunk(enc, ...)
        # self._dispatch_chunk_cumsum(enc, ...)
        # ...
    
    # Final norm + LM head
    self._dispatch_rmsnorm(enc, ...)
    self._dispatch_linear(enc, ...)
    
    enc.endEncoding()
    execute_and_wait(self.ctx, cmd)
    
    # Return logits (zero-copy view)
    return self.buf_logits
```

---

## Testing Checklist (Per Task)

- [ ] Unit test: Numerical correctness vs CPU reference
- [ ] Microbenchmark: Kernel speed vs naive implementation  
- [ ] Integration test: End-to-end inference produces valid output
- [ ] Profiling: GPU timeline shows reduced idle/fewer dispatches
- [ ] Memory: No leaks or unexpected allocations

---

## Performance Targets

| Model       | Baseline  | Target   | Metric        |
|-------------|-----------|----------|---------------|
| mamba2-130m | 50 tok/s  |  100+    | Throughput    |
| mamba2-780m | 11 tok/s  |  30+     | Throughput    |
| mamba2-130m | 20ms      |  <10ms   | Latency/token |

---

## References
- Triton BMM kernel: `mamba/mamba_ssm/ops/triton/ssd_bmm.py`
- Triton cumsum: `mamba/mamba_ssm/ops/triton/ssd_chunk_state.py`
- Metal Performance Shaders: https://developer.apple.com/metal/
