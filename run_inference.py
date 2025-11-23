import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import math
import time
import psutil
import os
import sys

# Rich library for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box

console = Console()

# 假設這是你已經寫好的 Metal 模組
# 如果你在本地測試，請確保此檔案存在
try:
    from mac_mamba_block import MambaBlockMetal
except ImportError:
    print("⚠️ Warning: 'mac_mamba_block' not found. Using mock for structure check.")
    class MambaBlockMetal: # Mock Class for syntax check if file missing
        def __init__(self, **kwargs): self.d_inner=kwargs['d_model']*kwargs['expand']; self.d_conv=4; self.nheads=1; self.headdim=64; self.d_state=16
        def allocate_inference_cache(self, batch_size): pass
        def step_inference(self, *args): 
            # Return a dummy pointer-like object just to pass execution flow
            class MockBuffer:
                def contents(self): return self
                def as_buffer(self, l): return b'\x00'*l
                def length(self): return 4*1*64 # float32 * nheads * headdim
            return MockBuffer()

class MambaMonitorMetal:
    def __init__(self, model_name="state-spaces/mamba-130m"):
        self.process = psutil.Process(os.getpid())
        
        # Welcome banner
        console.print(Panel.fit(
            f"[bold cyan]Mamba Metal Model Initialization[/bold cyan]\n"
            f"Process ID: {os.getpid()}\n"
            f"Model: [yellow]{model_name}[/yellow]",
            border_style="cyan"
        ))
        
        self.tokenizer = self._load_tokenizer(model_name)
        
        with console.status("[bold green]Loading model config...") as status:
            try:
                self.config = AutoConfig.from_pretrained(model_name)
            except:
                self.config = AutoConfig.from_pretrained("state-spaces/mamba-130m")
        
        # Model Params
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layer
        self.vocab_size = self.config.vocab_size  # Will be updated from actual weights
        self.d_state = 16 
        self.dt_rank = math.ceil(self.d_model / 16)
        
        # Display model info
        info_table = Table(show_header=False, box=box.SIMPLE)
        info_table.add_row("[cyan]d_model[/cyan]", str(self.d_model))
        info_table.add_row("[cyan]n_layers[/cyan]", str(self.n_layers))
        info_table.add_row("[cyan]vocab_size[/cyan]", str(self.vocab_size))
        console.print(Panel(info_table, title="[bold]Model Configuration[/bold]", border_style="blue"))
        
        # Initialize Metal Blocks with progress bar
        console.print(f"\n[bold green]Initializing {self.n_layers} Metal Blocks...[/bold green]")
        self.layers = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Creating blocks...", total=self.n_layers)
            for i in range(self.n_layers):
                block = MambaBlockMetal(
                    d_model=self.d_model, expand=2, d_state=self.d_state, d_conv=4, ngroups=1
                )
                self.layers.append(block)
                progress.update(task, advance=1)
        
        # Load Weights
        self._load_weights(model_name)
        
        mem_usage = self._get_mem()
        console.print(Panel(
            f"[bold green]✓[/bold green] Model Ready\n"
            f"Memory Usage: [yellow]{mem_usage:.2f} GB[/yellow]",
            border_style="green"
        ))

    def _get_time(self):
        return time.strftime("%H:%M:%S", time.localtime())

    def _get_mem(self):
        return self.process.memory_info().rss / 1024**3

    def _load_tokenizer(self, model_name):
        # Try to use the model's own tokenizer first
        console.print("[cyan]Loading tokenizer...[/cyan]")
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            console.print(f"[green]✓[/green] Using model tokenizer: {model_name}")
        except Exception as e:
            console.print(f"[yellow]⚠ Model tokenizer not found, trying EleutherAI fallback...[/yellow]")
            try:
                tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
                console.print(f"[yellow]⚠ Using EleutherAI tokenizer (vocab may not match!)[/yellow]")
            except:
                console.print(f"[red]✗ Failed to load tokenizer[/red]")
                raise
        
        tok.pad_token_id = tok.eos_token_id
        return tok

    def _load_weights(self, model_name):
        console.print("\n[bold cyan]Loading Weights & Converting Parameters...[/bold cyan]")

        if os.path.isdir(model_name):
            # Local model loading with progress
            console.print(f"[yellow]📁 Loading local model:[/yellow] {model_name}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                load_task = progress.add_task("[cyan]Reading model file...", total=100)
                
                try:
                    model_file = os.path.join(model_name, "model.safetensors")
                    file_size = os.path.getsize(model_file) if os.path.exists(model_file) else 0
                    progress.update(load_task, advance=30, description="[cyan]Loading safetensors...")
                    state_dict = load_file(model_file)
                    progress.update(load_task, advance=70, description="[green]✓ Loaded safetensors")
                except:
                    progress.update(load_task, advance=30, description="[cyan]Loading pytorch_model.bin...")
                    state_dict = torch.load(os.path.join(model_name, "pytorch_model.bin"), map_location="cpu")
                    progress.update(load_task, advance=70, description="[green]✓ Loaded pytorch_model.bin")
        else:
            # HuggingFace download
            console.print(f"[yellow]☁️  Downloading from HuggingFace:[/yellow] {model_name}")
            try:
                with console.status("[bold green]Downloading model.safetensors..."):
                    model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                    state_dict = load_file(model_path)
            except:
                with console.status("[bold green]Downloading pytorch_model.bin..."):
                    model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                    state_dict = torch.load(model_path, map_location="cpu")

        def get_w(key): return state_dict[key].float().numpy()
        
        # Load embedding and check vocab size
        with console.status("[bold green]Loading embeddings..."):
            emb_key = "backbone.embeddings.weight" if "backbone.embeddings.weight" in state_dict else "backbone.embedding.weight"
            self.embedding = get_w(emb_key)
            
            # Check if actual vocab size matches config
            actual_vocab_size = self.embedding.shape[0]
            if actual_vocab_size != self.vocab_size:
                console.print(f"[yellow]⚠ Vocab size mismatch![/yellow]")
                console.print(f"  Config vocab_size: {self.vocab_size}")
                console.print(f"  Actual embedding size: {actual_vocab_size}")
                console.print(f"  [cyan]Using actual size: {actual_vocab_size}[/cyan]")
                self.vocab_size = actual_vocab_size

        # Load layer weights with progress bar
        console.print("\n[bold green]Processing layer weights...[/bold green]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Converting weights...", total=self.n_layers)
            
            for i in range(self.n_layers):
                block = self.layers[i]
                prefix = f"backbone.layers.{i}"
                mixer = f"{prefix}.mixer"
                
                progress.update(task, description=f"[cyan]Layer {i+1}/{self.n_layers}...")
                
                block.norm_w = get_w(f"{prefix}.norm.weight")
                in_proj = get_w(f"{mixer}.in_proj.weight")
                
                # Pre-transpose and make contiguous for faster matmul
                block.x_proj_w_T = np.ascontiguousarray(in_proj[:block.d_inner, :].T)
                block.z_proj_w_T = np.ascontiguousarray(in_proj[block.d_inner:, :].T)
                
                block.conv_w = get_w(f"{mixer}.conv1d.weight").squeeze(1) 
                block.conv_b = get_w(f"{mixer}.conv1d.bias")
                
                x_proj = get_w(f"{mixer}.x_proj.weight")
                block.x_proj_dt_w_T = np.ascontiguousarray(x_proj[:self.dt_rank, :].T)
                block.x_proj_B_w_T = np.ascontiguousarray(x_proj[self.dt_rank : self.dt_rank + self.d_state, :].T)
                block.x_proj_C_w_T = np.ascontiguousarray(x_proj[self.dt_rank + self.d_state :, :].T)
                
                block.dt_proj_w_T = np.ascontiguousarray(get_w(f"{mixer}.dt_proj.weight").T)
                block.dt_proj_b = get_w(f"{mixer}.dt_proj.bias")
                
                # CRITICAL: Keep A_log as-is, don't convert to A
                # Metal kernel expects A_log and does exp(-exp(A_log)*dt) itself
                a_log = get_w(f"{mixer}.A_log") 
                block.A_log = a_log  # Store as A_log, not A
                
                block.D = get_w(f"{mixer}.D")
                block.out_proj_w_T = np.ascontiguousarray(get_w(f"{mixer}.out_proj.weight").T)
                
                progress.update(task, advance=1)

        # Load final layers
        with console.status("[bold green]Loading final layers..."):
            self.norm_f_w = get_w("backbone.norm_f.weight")
            self.lm_head_w_T = np.ascontiguousarray(get_w("lm_head.weight").T)
        
        console.print("[bold green]✓ All weights loaded successfully[/bold green]")

    def _rms_norm_np(self, x, w, eps=1e-5):
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        return x * w / np.sqrt(mean_sq + eps)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # --- 關鍵修正 2: Softplus 函數 ---
    def _softplus_np(self, x):
        return np.logaddexp(0, x)
    # -------------------------------

    def sample_token(self, logits, temperature=1.0, top_k=50, top_p=0.9):
        logits = logits[0, 0]
        logits = logits / temperature
        probs = self._softmax(logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        if top_k > 0:
            indices_to_remove = probs < np.sort(probs)[-top_k]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        confidence = probs[next_token]
        return next_token, confidence, entropy

    def generate(self, prompt, max_length=100, 
                 temperature=0.8, top_k=40, top_p=0.9, 
                 memory_limit_gb=4.0):
        
        print("\n" + "="*60)
        print(f"🚀 GENERATION START")
        print(f"Prompt: '{prompt}'")
        print(f"Config: Temp={temperature}, TopK={top_k}, TopP={top_p}, MemLimit={memory_limit_gb}GB")
        print("="*60)
        print(f"{'Token':<15} | {'Conf.':<8} | {'Ent.':<8} | {'Mem (GB)':<8}")
        print("-" * 60)

        input_ids = self.tokenizer.encode(prompt)
        
        # Initialize conv states
        conv_states = [np.zeros((1, layer.d_inner, layer.d_conv), dtype='f4') for layer in self.layers]
        
        for block in self.layers:
            block.allocate_inference_cache(batch_size=1)

        t0 = time.time()
        # Prefill
        for token in input_ids[:-1]:
            _, conv_states = self.step(token, conv_states)
        
        curr_token = input_ids[-1]
        generated_tokens = []
        nll_sum = 0.0
        oom_triggered = False
        
        try:
            for i in range(max_length):
                curr_mem = self._get_mem()
                if curr_mem > memory_limit_gb:
                    print(f"\n\n🛑 EMERGENCY STOP: Memory usage ({curr_mem:.2f} GB) exceeded limit ({memory_limit_gb} GB).")
                    oom_triggered = True
                    raise MemoryError("OOM")

                logits, conv_states = self.step(curr_token, conv_states)
                
                next_token, confidence, entropy = self.sample_token(
                    logits, temperature, top_k, top_p
                )
                
                nll_sum += -np.log(confidence + 1e-10)
                token_str = self.tokenizer.decode([next_token]).replace("\n", "\\n")
                
                # ANSI colors
                conf_color = "\033[92m" if confidence > 0.7 else "\033[93m" 
                reset = "\033[0m"
                
                print(f"{token_str:<15} | {conf_color}{confidence:.2f}{reset:<8} | {entropy:.2f}     | {curr_mem:.2f}")

                generated_tokens.append(next_token)
                curr_token = next_token
                
                if next_token == self.tokenizer.eos_token_id:
                    print("\n[EOS Reached]")
                    break
                    
        except KeyboardInterrupt:
            print("\n[User Interrupted]")
        except MemoryError:
            pass # Loop terminated by OOM

        # Statistics Calculation
        total_time = time.time() - t0
        generated_text = self.tokenizer.decode(generated_tokens)
        ppl = np.exp(nll_sum / len(generated_tokens)) if generated_tokens else 0.0
        final_mem = self._get_mem()
        
        if not oom_triggered:
            print("\n" + "="*60)
            print("📊 FINAL STATISTICS")
            print(f"Total Time      : {total_time:.2f}s")
            print(f"Throughput      : {len(generated_tokens)/total_time:.2f} tokens/s")
            print(f"Generation PPL  : {ppl:.4f} (Lower is better)")
            print(f"Final Memory    : {final_mem:.2f} GB")
            print("="*60)
            print(f"Result:\n{prompt}{generated_text}")
        
        return generated_text
    
    def sync_gpu(self):
        """Explicit GPU synchronization - wait for all pending Metal commands."""
        for block in self.layers:
            if hasattr(block, '_last_cmd') and block._last_cmd:
                block._last_cmd.waitUntilCompleted()
    
    def step(self, token, conv_states, conv_indices=None):
        # Optimized embedding lookup - direct indexing
        x = self.embedding[token:token+1]
        new_conv_states = []
        
        for i, block in enumerate(self.layers):
            residual = x
            x = self._rms_norm_np(x, block.norm_w)
            
            # Use pre-transposed weights (no .T needed)
            x_signal = x @ block.x_proj_w_T
            z = x @ block.z_proj_w_T
            
            # Simple Conv1d using np.roll (revert from circular buffer for debugging)
            c_state = conv_states[i]
            c_state = np.roll(c_state, -1, axis=-1)
            c_state[:, :, -1] = x_signal[0]
            new_conv_states.append(c_state)
            
            conv_out = np.sum(c_state * block.conv_w, axis=-1) + block.conv_b
            conv_out = conv_out[:, np.newaxis, :]
            # SiLU activation
            conv_out = conv_out * (1 / (1 + np.exp(-conv_out)))

            # SSM Params - use pre-transposed weights
            dt_hidden = conv_out @ block.x_proj_dt_w_T
            B = conv_out @ block.x_proj_B_w_T
            C = conv_out @ block.x_proj_C_w_T
            
            dt = dt_hidden @ block.dt_proj_w_T + block.dt_proj_b
            
            # Apply Softplus to dt
            dt = self._softplus_np(dt)
            
            H, E = block.nheads, block.headdim
            x_ssm_in = conv_out.reshape(1, H, E)
            dt_in = dt.reshape(1, H, E)
            z_in = z.reshape(1, H, E)
            B_in = B.reshape(1, 1, block.d_state)
            C_in = C.reshape(1, 1, block.d_state)
            
            # Metal Inference Call (async - dispatches but doesn't wait)
            out_buf = block.step_inference(x_ssm_in, dt_in, B_in, C_in, z_in)
            
            # Sync before reading (necessary for correctness with shared memory)
            if hasattr(block, '_last_cmd') and block._last_cmd:
                block._last_cmd.waitUntilCompleted()
            
            # Read result using persistent view (zero-copy)
            y_ssm = block.inference_cache["out_view"].reshape(1, 1, block.d_inner)
            
            # Use pre-transposed weight
            out = y_ssm @ block.out_proj_w_T
            x = out + residual

        x = self._rms_norm_np(x, self.norm_f_w)
        logits = x @ self.lm_head_w_T
        return logits, new_conv_states

if __name__ == "__main__":
    # 注意: 如果你有本地模型，請修改路徑，否則會嘗試從 HF 下載
 
    # model_path = "model/mamba-1.4b"
    model_path = "state-spaces/mamba-130m"
        
    model = MambaMonitorMetal(model_path)
    prompt = "Deep learning on Mac is"
    
    # # 1. 測試正常生成
    # model.generate(
    #     prompt, 
    #     max_length=50, 
    #     temperature=0.7, 
    #     top_k=50, 
    #     top_p=0.9,
    #     memory_limit_gb=16.0 # Set high enough to pass
    # )

    # print("\n\n")
    # print("-" * 20 + " Testing Memory Kill Switch (Limit: 0.1GB) " + "-" * 20)

    # 2. 測試記憶體熔斷
    model.generate(
        prompt, 
        max_length=100  , 
        temperature=0.8, 
        top_k=40, 
        top_p=0.9,
        memory_limit_gb=6 # Set low to trigger OOM
    )