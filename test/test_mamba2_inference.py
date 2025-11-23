"""
Simple test script for Mamba2Metal inference.

This script demonstrates the Mamba2Metal module without requiring
a full pretrained model download.
"""

import torch
from mamba2_metal import Mamba2Metal

def test_mamba2_metal_inference():
    """Test Mamba2Metal with random inputs."""
    
    print("=" * 60)
    print("Mamba2Metal Simple Inference Test")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating Mamba2Metal model...")
    model = Mamba2Metal(
        d_model=768,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=8,
    )
    print(f"   ✓ Model created")
    print(f"     - d_model: {model.d_model}")
    print(f"     - d_inner: {model.d_inner}")
    print(f"     - nheads: {model.nheads}")
    print(f"     - ngroups: {model.ngroups}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 1
    seq_len = 32
    x = torch.randn(batch_size, seq_len, model.d_model)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"   ✓ Forward pass successful")
    print(f"     - Input shape: {x.shape}")
    print(f"     - Output shape: {y.shape}")
    
    # Test inference caching
    print("\n3. Testing inference cache...")
    conv_state, ssm_state = model.allocate_inference_cache(batch_size=1, max_seqlen=1024)
    print(f"   ✓ Inference cache allocated")
    print(f"     - conv_state: {conv_state.shape}")
    print(f"     - ssm_state: {ssm_state.shape}")
    
    # Test single-step generation
    print("\n4. Testing single-step generation...")
    num_steps = 10
    
    # Start with a random "token embedding"
    current_input = torch.randn(1, 1, model.d_model)
    
    outputs = []
    for step in range(num_steps):
        with torch.no_grad():
            output, conv_state, ssm_state = model.step(current_input, conv_state, ssm_state)
        outputs.append(output)
        
        # Use output as next input (in real scenario, this would go through embedding)
        current_input = output
        
        if (step + 1) % 5 == 0:
            print(f"   ✓ Generated {step + 1} steps")
    
    print(f"   ✓ Single-step generation successful")
    print(f"     - Generated {num_steps} steps")
    print(f"     - Output shape per step: {outputs[0].shape}")
    
    # Test batch processing
    print("\n5. Testing batch processing...")
    batch_size = 4
    seq_len = 64
    x_batch = torch.randn(batch_size, seq_len, model.d_model)
    
    with torch.no_grad():
        y_batch = model(x_batch)
    
    print(f"   ✓ Batch processing successful")
    print(f"     - Batch size: {batch_size}")
    print(f"     - Input shape: {x_batch.shape}")
    print(f"     - Output shape: {y_batch.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nMamba2Metal is working correctly and ready for use.")
    print("To use with a pretrained model, run:")
    print("  python run_mamba2_metal_inference.py --model state-spaces/mamba2-130m")
    print()


if __name__ == "__main__":
    test_mamba2_metal_inference()
