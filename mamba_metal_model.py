import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import math
import time
import psutil
import os

# 假設這是你已經寫好的 Metal 模組
from mac_mamba_block import MambaBlockMetal

class MambaModelMetal:
    def __init__(self, model_name="state-spaces/mamba-130m"):
        self.process = psutil.Process(os.getpid())
        mem_start = self.process.memory_info().rss / 1024**3
        print(f"Initial Memory: {mem_start:.2f} GB")

        print(f"Loading Mamba-1 Model: {model_name}")
        
        # 1. Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 2. Load Config
        self.config = AutoConfig.from_pretrained(model_name)
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layer
        self.vocab_size = self.config.vocab_size
        self.d_state = 16 
        self.dt_rank = math.ceil(self.d_model / 16)
        
        # 3. Initialize Metal Blocks
        self.layers = []
        for i in range(self.n_layers):
            block = MambaBlockMetal(
                d_model=self.d_model,
                expand=2, 
                d_state=self.d_state, 
                d_conv=4,
                ngroups=1
            )
            self.layers.append(block)
        
        # 4. Load Weights
        self._load_weights(model_name)
        
        mem_end = self.process.memory_info().rss / 1024**3
        print(f"Model Loaded. Current Memory: {mem_end:.2f} GB (Increased by {mem_end - mem_start:.2f} GB)")
        
    def _load_weights(self, model_name):

        try:
            model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
            state_dict = load_file(model_path)
        except:
            model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu")

        def get_w(key): return state_dict[key].float().numpy()
        
        # Embeddings
        emb_key = "backbone.embeddings.weight" if "backbone.embeddings.weight" in state_dict else "backbone.embedding.weight"
        self.embedding = get_w(emb_key)

        for i in range(self.n_layers):
            block = self.layers[i]
            prefix = f"backbone.layers.{i}"
            mixer = f"{prefix}.mixer"
            
            block.norm_w = get_w(f"{prefix}.norm.weight")
            in_proj = get_w(f"{mixer}.in_proj.weight")
            block.x_proj_w = in_proj[:block.d_inner, :]
            block.z_proj_w = in_proj[block.d_inner:, :]
            block.conv_w = get_w(f"{mixer}.conv1d.weight").squeeze(1) 
            block.conv_b = get_w(f"{mixer}.conv1d.bias")
            x_proj = get_w(f"{mixer}.x_proj.weight")
            block.x_proj_dt_w = x_proj[:self.dt_rank, :]
            block.x_proj_B_w = x_proj[self.dt_rank : self.dt_rank + self.d_state, :]
            block.x_proj_C_w = x_proj[self.dt_rank + self.d_state :, :]
            block.dt_proj_w = get_w(f"{mixer}.dt_proj.weight")
            block.dt_proj_b = get_w(f"{mixer}.dt_proj.bias")
            block.A_log = get_w(f"{mixer}.A_log") 
            block.D = get_w(f"{mixer}.D")
            block.out_proj_w = get_w(f"{mixer}.out_proj.weight")

        self.norm_f_w = get_w("backbone.norm_f.weight")
        self.lm_head_w = get_w("lm_head.weight")

    def _rms_norm_np(self, x, w, eps=1e-5):
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        return x * w / np.sqrt(mean_sq + eps)

    def generate(self, prompt, max_length=50):
        print("-" * 40)
        print(f"Starting Generation Task")
        print(f"Prompt: '{prompt}'")
        
        # Memory Check Before
        mem_before = self.process.memory_info().rss / 1024**3
        
        input_ids = self.tokenizer.encode(prompt)
        prompt_len = len(input_ids)
        
        # Init States
        conv_states = [np.zeros((1, layer.d_inner, layer.d_conv), dtype='f4') for layer in self.layers]
        for block in self.layers:
            block.allocate_inference_cache(batch_size=1)

        # --- Phase 1: Prefill (Prompt Processing) ---
        t0 = time.perf_counter()
        
        # Process prompt (except last token)
        for token in input_ids[:-1]:
            _, conv_states = self.step(token, conv_states)
            
        t1 = time.perf_counter()
        prefill_time = t1 - t0
        prefill_tokens = prompt_len - 1
        
        # --- Phase 2: Decoding (Generation) ---
        curr_token = input_ids[-1]
        generated_tokens = []
        
        print("Output: ", end="", flush=True)
        
        t_gen_start = time.perf_counter()
        
        for _ in range(max_length):
            logits, conv_states = self.step(curr_token, conv_states)
            
            next_token = np.argmax(logits[0, 0])
            generated_tokens.append(next_token)
            
            print(self.tokenizer.decode([next_token]), end="", flush=True)
            curr_token = next_token
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        t_gen_end = time.perf_counter()
        gen_time = t_gen_end - t_gen_start
        gen_count = len(generated_tokens)
        
        print("\n" + "-" * 40)
        
        # --- Metrics Report ---
        mem_after = self.process.memory_info().rss / 1024**3
        peak_mem = mem_after # Approximate
        
        print(f"📊 Performance Metrics:")
        print(f"-----------------------")
        print(f"Memory Usage      : {mem_after:.2f} GB (Peak/Current)")
        print(f"Prompt Length     : {prompt_len} tokens")
        print(f"Generated Length  : {gen_count} tokens")
        
        if prefill_tokens > 0:
            print(f"Prefill Latency   : {prefill_time*1000:.2f} ms")
            print(f"Prefill Throughput: {prefill_tokens / prefill_time:.2f} tokens/sec")
        
        print(f"Decode Latency    : {gen_time*1000:.2f} ms")
        print(f"Decode Throughput : {gen_count / gen_time:.2f} tokens/sec")
        print("-" * 40)
        
        return self.tokenizer.decode(generated_tokens)

    def step(self, token, conv_states):
        # 1. Embedding
        x = self.embedding[np.array([[token]])] 
        
        new_conv_states = []
        
        for i, block in enumerate(self.layers):
            residual = x
            x = self._rms_norm_np(x, block.norm_w)
            
            # Projections
            x_signal = x @ block.x_proj_w.T
            z = x @ block.z_proj_w.T
            
            # Conv1d
            c_state = conv_states[i]
            c_state = np.roll(c_state, -1, axis=-1)
            c_state[:, :, -1] = x_signal.squeeze(1)
            new_conv_states.append(c_state)
            
            conv_out = np.sum(c_state * block.conv_w, axis=-1) + block.conv_b
            conv_out = conv_out[:, np.newaxis, :]
            conv_out = conv_out * (1 / (1 + np.exp(-conv_out))) # SiLU

            # SSM Params
            dt_hidden = conv_out @ block.x_proj_dt_w.T
            B = conv_out @ block.x_proj_B_w.T
            C = conv_out @ block.x_proj_C_w.T
            dt = dt_hidden @ block.dt_proj_w.T + block.dt_proj_b
            
            # Metal Kernel Prep
            H, E = block.nheads, block.headdim
            x_ssm_in = conv_out.reshape(1, H, E)
            dt_in = dt.reshape(1, H, E)
            z_in = z.reshape(1, H, E)
            B_in = B.reshape(1, 1, block.d_state)
            C_in = C.reshape(1, 1, block.d_state)
            
            # Metal Execution
            out_buf = block.step_inference(x_ssm_in, dt_in, B_in, C_in, z_in)
            
            # Readback
            y_ssm = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype='f4').reshape(1, H, E)
            y_out = y_ssm.reshape(1, 1, block.d_inner)
            
            # Out Proj & Res
            out = y_out @ block.out_proj_w.T
            x = out + residual

        x = self._rms_norm_np(x, self.norm_f_w)
        logits = x @ self.lm_head_w.T
        
        return logits, new_conv_states

