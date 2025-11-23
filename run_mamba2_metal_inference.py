"""
Inference script for Mamba2 with Metal acceleration.

This script loads the pretrained state-spaces/mamba2-130m model and runs
text generation using the Metal-accelerated backend.
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from mamba2_metal import Mamba2Metal

class Mamba2ModelMetal:
    """Wrapper for Mamba2 model with Metal acceleration."""
    
    def __init__(self, model_name="state-spaces/mamba2-130m", device="cpu"):
        """
        Initialize Mamba2 model with Metal acceleration.
        
        Args:
            model_name: HuggingFace model name (used for config reference mostly)
            device: Device to use (cpu/cuda/mps)
        """

        self.device = device
        
        # --- 修改 1: 直接使用 GPTNeoXTokenizer ---
        print(f"Loading GPTNeoXTokenizer (EleutherAI/gpt-neox-20b) directly...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Error loading GPTNeoXTokenizer: {e}")
            raise e
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded (vocab_size: {len(self.tokenizer)})")
        
        print(f"Loading Mamba2 model with Metal acceleration...")
        
        # 由於你的環境似乎無法直接載入 pretrained model，這裡保留你原本的 "From Scratch" 測試邏輯
        # 並依據 SimpleConfig 初始化
        print("Creating model from scratch using SimpleMambaLM...")
        
        class SimpleMambaLM(nn.Module):
            def __init__(self, config):
                super().__init__()
                # 1. 增加 Embedding 層：把 (B, L) 轉成 (B, L, D)
                self.embedding = nn.Embedding(config.vocab_size, config.d_model)
                # 2. 這是原本的 Mamba 層
                self.backbone = Mamba2Metal(
                    d_model=config.d_model,
                    d_state=128,
                    d_conv=4,
                    expand=2,
                    headdim=64,
                    ngroups=8,
                )
                # 3. 增加輸出層：把 (B, L, D) 轉回 (B, L, V)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids) # 變成 3 維
                x = self.backbone(x)
                logits = self.lm_head(x)
                return logits

        # 設定 Config
        class SimpleConfig:
            vocab_size = 50280 # GPT-NeoX vocab size
            d_model = 768
            n_layer = 1
        
        self.config = SimpleConfig()
        
        # 初始化這個包含 Embedding 的完整模型
        self.model = SimpleMambaLM(self.config)
        self.model.to(device) # 記得移動到 device
        
        print(f"✓ Model loaded successfully (From Scratch Test Mode)")

    def _convert_to_metal(self, model):
        """Placeholder for layer replacement."""
        return model
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        """
        Generate text from a prompt.
        """
        print(f"\nTokenizing prompt...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        print(f"Input tokens: {input_ids.shape[1]}")
        print(f"Generating up to {max_length} tokens...")
        
        # --- 修改 2: 直接使用自定義的 _simple_generate 推論 ---
        # 移除 hasattr(self.model, 'generate') 的檢查，強制跑你的迴圈
        print("Using custom greedy generation loop...")
        output_ids = self._simple_generate(input_ids, max_length)
        
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


    def _simple_generate(self, input_ids, max_length):
        """
        Basic greedy generation loop for raw PyTorch models.
        """
        generated = input_ids
        for i in range(max_length):
            # 取得模型輸出
            logits = self.model(generated)
            # 取最後一個 token 的 logits (B, L, V) -> (B, V)
            last_token_logits = logits[:, -1, :]
            
            # Greedy search: 取機率最大的
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)
            
            # 串接
            generated = torch.cat([generated, next_token], dim=1)
            
            # 簡單進度顯示
            if i % 10 == 0:
                print(f".", end="", flush=True)

            # 檢查是否是 EOS
            if self.tokenizer.eos_token_id and next_token.item() == self.tokenizer.eos_token_id:
                break
                
        print() # Newline after dots
        return generated
    

def main():
    parser = argparse.ArgumentParser(description="Run Mamba2 Inference with Metal Acceleration")
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba2-130m",
        help="HuggingFace model name (ignored for tokenizer now)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Mamba is a type of",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (Not used in greedy search currently)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (Not used in greedy search currently)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mamba2 Metal Inference")
    print("=" * 60)
    
    # Initialize model
    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.model}")
    print("=" * 60)
    try:
        model = Mamba2ModelMetal(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate text
    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    
    try:
        output = model.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        print(f"\n{'=' * 60}")
        print("Generated Output:")
        print("=" * 60)
        print(output)
        print()
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()