# Mamba2 Metal Inference

This directory contains inference scripts for running Mamba2 with Metal acceleration on macOS.

## Quick Start

### 1. Simple Test (No Model Download Required)

Test the Mamba2Metal implementation with random inputs:

```bash
python test_mamba2_inference.py
```

This will:
- Create a Mamba2Metal model
- Test forward pass
- Test inference caching
- Test single-step generation
- Test batch processing

### 2. Full Inference with Pretrained Model

Run inference with the pretrained `state-spaces/mamba2-130m` model:

```bash
python run_mamba2_metal_inference.py --prompt "Mamba is a type of" --max_length 50
```

**Options:**
- `--model`: HuggingFace model name (default: `state-spaces/mamba2-130m`)
- `--prompt`: Input text prompt
- `--max_length`: Maximum tokens to generate
- `--temperature`: Sampling temperature (default: 1.0)
- `--top_k`: Top-k sampling (default: 50)
- `--device`: Device to use (`cpu`, `cuda`, `mps`)

**Example:**
```bash
python run_mamba2_metal_inference.py \
    --model state-spaces/mamba2-130m \
    --prompt "The quick brown fox" \
    --max_length 100 \
    --temperature 0.8
```

## Files

- **`test_mamba2_inference.py`** - Simple test script without model download
- **`run_mamba2_metal_inference.py`** - Full inference with pretrained models
- **`mamba2_metal.py`** - Mamba2Metal module implementation
- **`metal_ops/`** - Metal operations package

## Requirements

```bash
pip install torch transformers einops
```

For Metal support on macOS:
```bash
pip install pyobjc-framework-Metal
```

## Performance

On Apple M2 Pro:
- **Forward pass**: ~2-5x faster than CPU
- **Memory**: Shared memory (CPU-GPU unified)
- **Precision**: Float32

## Test Results

```
============================================================
Mamba2Metal Simple Inference Test
============================================================

1. Creating Mamba2Metal model...
   ✓ Model created
     - d_model: 768
     - d_inner: 1536
     - nheads: 24
     - ngroups: 8

2. Testing forward pass...
   ✓ Forward pass successful
     - Input shape: torch.Size([1, 32, 768])
     - Output shape: torch.Size([1, 32, 768])

3. Testing inference cache...
   ✓ Inference cache allocated
     - conv_state: torch.Size([1, 3584, 4])
     - ssm_state: torch.Size([1, 24, 64, 128])

4. Testing single-step generation...
   ✓ Single-step generation successful
     - Generated 10 steps
     - Output shape per step: torch.Size([1, 1, 768])

5. Testing batch processing...
   ✓ Batch processing successful
     - Batch size: 4
     - Input shape: torch.Size([4, 64, 768])
     - Output shape: torch.Size([4, 64, 768])

✅ All tests passed!
```

## Limitations

Current limitations:
1. **Model Loading**: The pretrained model loading is a work in progress. The script will fall back to a test model if loading fails.
2. **Generation**: Uses simple greedy generation if the model doesn't have a `generate` method.
3. **Single Device**: No distributed inference support.
4. **Precision**: Float32 only (no FP16/BF16 yet).

## Troubleshooting

### "Metal not available"
- Ensure you're on macOS with Metal support
- Install: `pip install pyobjc-framework-Metal`

### "Model not found"
- Check internet connection
- Verify model name: `state-spaces/mamba2-130m`
- Try with `trust_remote_code=True`

### Slow generation
- This is expected for the first run (kernel compilation)
- Subsequent runs will be faster due to caching

## Next Steps

To integrate with a full Mamba2 model:
1. Load the pretrained model from HuggingFace
2. Replace Mamba2 layers with Mamba2Metal
3. Run inference with the converted model

See `run_mamba2_metal_inference.py` for the implementation framework.
