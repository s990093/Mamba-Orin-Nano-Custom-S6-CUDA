#!/usr/bin/env python3
"""
Debug script to check if weights are loading correctly
"""

from safetensors.torch import load_file
import torch
import os

model_path = "model/mamba-130m" if os.path.exists("model/mamba-130m") else "state-spaces/mamba-130m"
print(f"Checking model: {model_path}")

if os.path.isdir(model_path):
    try:
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    except:
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
else:
    from huggingface_hub import hf_hub_download
    try:
        model_file = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state_dict = load_file(model_file)
    except:
        model_file = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")

print("\nKey weight shapes:")
for key in sorted(state_dict.keys()):
    if 'embedding' in key or 'lm_head' in key:
        print(f"{key}: {state_dict[key].shape}")

print("\n\nFirst layer weights:")
for key in sorted(state_dict.keys()):
    if 'layers.0' in key:
        print(f"{key}: {state_dict[key].shape}")
