"""
Run Mamba 2 inference using MLX.
"""

import argparse
import mlx.core as mx
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerFast
import os

# Import from our new package
from src.mamba2_mlx import Mamba2Model, Mamba2Config, load_weights, generate

# Set default device to GPU for better performance
mx.set_default_device(mx.gpu)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json or tokenizer directory")
    parser.add_argument("--prompt", type=str, default="Mamba is")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for weights")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (limit to top k tokens)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p/nucleus sampling (cumulative probability)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (>1.0 discourages repetition)")
    args = parser.parse_args()
    
    # Map string to mlx dtype
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16
    }
    target_dtype = dtype_map[args.dtype]
    
    tokenizer = None
    if args.tokenizer:
        try:
            if args.tokenizer.endswith(".json"):
                print(f"Loading tokenizer from file: {args.tokenizer}")
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
            else:
                print(f"Loading tokenizer from directory: {args.tokenizer}")
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"Error loading tokenizer from {args.tokenizer}: {e}")
            
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer from {args.model}: {e}")
            print("Fallback to EleutherAI/gpt-neox-20b")
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config
    if os.path.isdir(args.model):
        config_path = os.path.join(args.model, "config.json")
        if os.path.exists(config_path):
            hf_config = AutoConfig.from_pretrained(args.model)
        else:
            parent_dir = os.path.dirname(args.model)
            if os.path.exists(os.path.join(parent_dir, "config.json")):
                 hf_config = AutoConfig.from_pretrained(parent_dir)
            else:
                 print("Warning: Could not find config.json, using default Mamba2-130m config")
                 hf_config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
    elif os.path.isfile(args.model):
        parent_dir = os.path.dirname(args.model)
        if os.path.exists(os.path.join(parent_dir, "config.json")):
             hf_config = AutoConfig.from_pretrained(parent_dir)
        else:
             print("Warning: Could not find config.json, using default Mamba2-130m config")
             hf_config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
    else:
        hf_config = AutoConfig.from_pretrained(args.model)
    
    # Helper to get attribute with fallbacks
    def get_config_attr(config, keys, default):
        for k in keys:
            if hasattr(config, k):
                return getattr(config, k)
        return default

    config = Mamba2Config(
        d_model=get_config_attr(hf_config, ["d_model", "hidden_size"], 768),
        n_layer=get_config_attr(hf_config, ["n_layer", "num_hidden_layers"], 24),
        vocab_size=get_config_attr(hf_config, ["vocab_size"], 50277),
        d_state=get_config_attr(hf_config, ["d_state", "state_size"], 128),
        d_conv=get_config_attr(hf_config, ["d_conv", "conv_kernel"], 4),
        expand=get_config_attr(hf_config, ["expand"], 2),
        headdim=get_config_attr(hf_config, ["headdim", "head_dim"], 64),
        ngroups=get_config_attr(hf_config, ["ngroups", "n_groups"], 1),
        tie_word_embeddings=get_config_attr(hf_config, ["tie_word_embeddings"], True),
        rms_norm_eps=get_config_attr(hf_config, ["rms_norm_eps", "layer_norm_epsilon"], 1e-5),
    )
    
    model = Mamba2Model(config)
    
    mx.eval(model.parameters())
    
    load_weights(model, args.model, dtype=target_dtype)
    
    generate(
        model, 
        tokenizer, 
        args.prompt, 
        max_tokens=args.max_tokens, 
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )

if __name__ == "__main__":
    main()
