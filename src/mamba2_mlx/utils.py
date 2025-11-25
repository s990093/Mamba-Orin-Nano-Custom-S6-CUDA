
import os
import torch
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

def load_weights(model, hf_model_name, dtype=mx.float32):
    print(f"Loading weights from {hf_model_name}...")
    
    state_dict = None
    
    # Check if local directory
    if os.path.isdir(hf_model_name):
        safetensors_path = os.path.join(hf_model_name, "model.safetensors")
        bin_path = os.path.join(hf_model_name, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            print(f"Found local safetensors: {safetensors_path}")
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"Found local pytorch_model.bin: {bin_path}")
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in local directory: {hf_model_name}")
    elif os.path.isfile(hf_model_name):
        print(f"Loading weights from local file: {hf_model_name}")
        if hf_model_name.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(hf_model_name)
        else:
            state_dict = torch.load(hf_model_name, map_location="cpu")
    else:
        # HuggingFace Hub
        from huggingface_hub import hf_hub_download
        
        try:
            file_path = hf_hub_download(repo_id=hf_model_name, filename="model.safetensors")
            from safetensors.torch import load_file
            state_dict = load_file(file_path)
        except:
            file_path = hf_hub_download(repo_id=hf_model_name, filename="pytorch_model.bin")
            state_dict = torch.load(file_path, map_location="cpu")
        
    # Convert to MLX
    mlx_weights = {}
    
    print("Converting weights to MLX...")
    for k, v in tqdm(state_dict.items(), desc="Processing Weights"):
        # Map keys
        new_k = k
        if k.startswith("backbone.embeddings."):
            new_k = k.replace("backbone.embeddings.", "embedding.")
        elif k.startswith("backbone.embedding."):
            new_k = k.replace("backbone.embedding.", "embedding.")
        elif k.startswith("backbone.layers."):
            new_k = k.replace("backbone.layers.", "layers.")
        elif k.startswith("backbone.norm_f."):
            new_k = k.replace("backbone.norm_f.", "norm_f.")
            
        # Convert tensor to numpy then mx.array
        if isinstance(v, torch.Tensor):
            v = v.float().cpu().numpy()
            
        if "conv1d.weight" in new_k:
            # PyTorch: (D, 1, K) -> MLX: (D, K, 1)
            v = v.transpose(0, 2, 1) # (D, K, 1)
            
        mlx_weights[new_k] = mx.array(v).astype(dtype)
        
    # Check for vocab size mismatch
    if "embedding.weight" in mlx_weights:
        loaded_vocab_size = mlx_weights["embedding.weight"].shape[0]
        if loaded_vocab_size != model.config.vocab_size:
            print(f"Vocab size mismatch: config={model.config.vocab_size}, weights={loaded_vocab_size}. Resizing...")
            model.config.vocab_size = loaded_vocab_size
            model.embedding = nn.Embedding(loaded_vocab_size, model.config.d_model)
            model.lm_head = nn.Linear(model.config.d_model, loaded_vocab_size, bias=False)
            if model.config.tie_word_embeddings:
                model.lm_head.weight = model.embedding.weight # Re-tie weights
            
    # Load into model
    model.load_weights(list(mlx_weights.items()))
    print(f"Weights loaded successfully with dtype={dtype}.")
