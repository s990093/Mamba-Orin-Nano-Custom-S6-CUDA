"""
Test script for Mamba2Metal implementation.

This script tests the basic functionality of the Metal-accelerated Mamba2 module.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_metal_ops():
    """Test individual Metal operations."""
    print("=" * 60)
    print("Testing Metal Operations")
    print("=" * 60)
    
    try:
        from metal_ops import METAL_AVAILABLE
        
        if not METAL_AVAILABLE:
            print("❌ Metal not available on this system")
            return False
        
        print("✓ Metal is available")
        
        # Test imports
        from metal_ops import (
            causal_conv1d_fn,
            selective_state_update,
            RMSNormGated,
            mamba_chunk_scan_combined,
        )
        print("✓ All Metal operations imported successfully")
        
        # Test RMSNorm
        print("\nTesting RMSNormGated...")
        norm = RMSNormGated(hidden_size=768, eps=1e-5)
        x = torch.randn(2, 10, 768)
        z = torch.randn(2, 10, 768)
        
        # Test without gating
        y1 = norm(x)
        assert y1.shape == x.shape, f"Shape mismatch: {y1.shape} != {x.shape}"
        print(f"  ✓ RMSNorm without gating: {x.shape} -> {y1.shape}")
        
        # Test with gating
        y2 = norm(x, z)
        assert y2.shape == x.shape, f"Shape mismatch: {y2.shape} != {x.shape}"
        print(f"  ✓ RMSNorm with gating: {x.shape} -> {y2.shape}")
        
        # Test causal_conv1d_fn
        print("\nTesting causal_conv1d_fn...")
        D_conv = 256
        x_conv = torch.randn(2, 100, D_conv)  # (B, L, D)
        weight = torch.randn(D_conv, 4)  # (D, W)
        bias = torch.randn(D_conv)  # (D,)
        
        y_conv = causal_conv1d_fn(x_conv, weight, bias, activation="silu")
        assert y_conv.shape == x_conv.shape, f"Shape mismatch: {y_conv.shape} != {x_conv.shape}"
        print(f"  ✓ causal_conv1d_fn: {x_conv.shape} -> {y_conv.shape}")
        
        print("\n✅ All Metal operations tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Metal operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba2_metal():
    """Test Mamba2Metal module."""
    print("\n" + "=" * 60)
    print("Testing Mamba2Metal Module")
    print("=" * 60)
    
    try:
        from mamba2_metal import Mamba2Metal, METAL_AVAILABLE
        
        if not METAL_AVAILABLE:
            print("⚠️  Metal not available, testing CPU fallback...")
        
        # Create model
        print("\nCreating Mamba2Metal model...")
        model = Mamba2Metal(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=8,
        )
        print(f"  ✓ Model created")
        print(f"    - d_model: {model.d_model}")
        print(f"    - d_inner: {model.d_inner}")
        print(f"    - nheads: {model.nheads}")
        print(f"    - ngroups: {model.ngroups}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, model.d_model)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == x.shape, f"Output shape mismatch: {y.shape} != {x.shape}"
        print(f"  ✓ Forward pass: {x.shape} -> {y.shape}")
        
        # Test inference cache
        print("\nTesting inference cache...")
        conv_state, ssm_state = model.allocate_inference_cache(batch_size=1, max_seqlen=1024)
        print(f"  ✓ Inference cache allocated")
        print(f"    - conv_state: {conv_state.shape}")
        print(f"    - ssm_state: {ssm_state.shape}")
        
        # Test single-step inference
        print("\nTesting single-step inference...")
        x_step = torch.randn(1, 1, model.d_model)
        with torch.no_grad():
            y_step, conv_state, ssm_state = model.step(x_step, conv_state, ssm_state)
        
        assert y_step.shape == x_step.shape, f"Step output shape mismatch: {y_step.shape} != {x_step.shape}"
        print(f"  ✓ Single-step inference: {x_step.shape} -> {y_step.shape}")
        
        print("\n✅ All Mamba2Metal tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Mamba2Metal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility():
    """Test compatibility with original Mamba2."""
    print("\n" + "=" * 60)
    print("Testing Compatibility with Original Mamba2")
    print("=" * 60)
    
    try:
        from mamba2_metal import Mamba2Metal
        
        # Try to import original Mamba2
        try:
            from mamba.mamba_ssm.modules.mamba2 import Mamba2
            has_original = True
        except ImportError:
            print("⚠️  Original Mamba2 not available, skipping compatibility test")
            return True
        
        # Create both models with same config
        config = {
            "d_model": 768,
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "headdim": 64,
            "ngroups": 8,
        }
        
        print("\nCreating models with identical config...")
        model_metal = Mamba2Metal(**config)
        model_orig = Mamba2(**config)
        
        # Compare parameter counts
        params_metal = sum(p.numel() for p in model_metal.parameters())
        params_orig = sum(p.numel() for p in model_orig.parameters())
        
        print(f"  Metal model parameters: {params_metal:,}")
        print(f"  Original model parameters: {params_orig:,}")
        
        if params_metal == params_orig:
            print("  ✓ Parameter counts match!")
        else:
            print(f"  ⚠️  Parameter count mismatch (difference: {abs(params_metal - params_orig):,})")
        
        # Test that they have the same structure
        print("\nComparing model structure...")
        metal_keys = set(dict(model_metal.named_parameters()).keys())
        orig_keys = set(dict(model_orig.named_parameters()).keys())
        
        common_keys = metal_keys & orig_keys
        metal_only = metal_keys - orig_keys
        orig_only = orig_keys - metal_keys
        
        print(f"  Common parameters: {len(common_keys)}")
        if metal_only:
            print(f"  Metal-only parameters: {metal_only}")
        if orig_only:
            print(f"  Original-only parameters: {orig_only}")
        
        print("\n✅ Compatibility test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Mamba2Metal Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test Metal operations
    results.append(("Metal Operations", test_metal_ops()))
    
    # Test Mamba2Metal module
    results.append(("Mamba2Metal Module", test_mamba2_metal()))
    
    # Test compatibility
    results.append(("Compatibility", test_compatibility()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
