import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from mac_mamba_block import MambaBlockMetal

class MambaModelMetal:
    def __init__(self, model_name="state-spaces/mamba-130m"):
        print(f"Loading model: {model_name}")
        self.is_mamba2 = "mamba2" in model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        except Exception:
            print("Tokenizer load failed, trying EleutherAI/gpt-neox-20b")
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layer
        self.vocab_size = self.config.vocab_size
        
        # Mamba 2 defaults
        self.d_state = 128 if self.is_mamba2 else 16
        
        print(f"Model config: d_model={self.d_model}, n_layers={self.n_layers}, vocab_size={self.vocab_size}, is_mamba2={self.is_mamba2}")
        
        # Initialize Metal Blocks
        self.layers = []
        for i in range(self.n_layers):
            print(f"Initializing Metal Block {i+1}/{self.n_layers}...")
            block = MambaBlockMetal(
                d_model=self.d_model,
                expand=2, 
                d_state=self.d_state, 
                d_conv=4,
                ngroups=1, 
                rmsnorm=True
            )
            self.layers.append(block)
            
        self._load_weights(model_name)
        # print("Skipping weight loading for stability test. Initializing random weights.")
        # self.embedding = np.random.randn(self.vocab_size, self.d_model).astype('f4') * 0.02
        # self.norm_f_w = np.ones(self.d_model, dtype='f4')
        # self.lm_head_w = np.random.randn(self.vocab_size, self.d_model).astype('f4') * 0.02
        
    def _load_weights(self, model_name):
        print("Downloading and loading weights...")
        # Download model.safetensors
        try:
            model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
            state_dict = load_file(model_path)
        except Exception:
            # Fallback to pytorch_model.bin if safetensors not found
            print("safetensors not found, trying pytorch_model.bin")
            model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu")
            
        print("Copying weights to Metal blocks...")
        
        # keys = list(state_dict.keys())
        # print("All keys in state_dict:")
        # for k in keys:
        #     print(k)
            
        # Embeddings
        # Handle vocab size mismatch by slicing if necessary
        if "backbone.embeddings.weight" in state_dict:
            emb_key = "backbone.embeddings.weight"
        elif "backbone.embedding.weight" in state_dict:
            emb_key = "backbone.embedding.weight"
        elif "embedding.weight" in state_dict:
            emb_key = "embedding.weight"
        else:
            # Try to find embedding key
            emb_key = [k for k in state_dict.keys() if "embed" in k and "weight" in k][0]
            print(f"Guessed embedding key: {emb_key}")

        emb_w = state_dict[emb_key].float().numpy()
        # Use actual vocab size from checkpoint to avoid cutting off special tokens
        actual_vocab_size = emb_w.shape[0]
        if actual_vocab_size != self.vocab_size:
            print(f"Warning: Config vocab_size={self.vocab_size}, but checkpoint has {actual_vocab_size}. Using checkpoint size.")
            self.vocab_size = actual_vocab_size
        self.embedding = emb_w
        
        # Layers
        for i in range(self.n_layers):
            block = self.layers[i]
            prefix = f"backbone.layers.{i}"
            
            # Norm
            block.norm_w = state_dict[f"{prefix}.norm.weight"].float().numpy()
            
            # Mixer
            mixer_prefix = f"{prefix}.mixer"
            
            # Conv1d
            conv_w = state_dict[f"{mixer_prefix}.conv1d.weight"].float().numpy()
            # Mamba 2: conv operates on [x, v] concatenated, shape is (d_inner + ngroups*d_state, 1, d_conv)
            # We only need the first d_inner channels for x
            if self.is_mamba2:
                conv_w = conv_w[:block.d_inner, :, :]
            block.conv_w = conv_w.squeeze(1)
            
            conv_b = state_dict[f"{mixer_prefix}.conv1d.bias"].float().numpy()
            if self.is_mamba2:
                conv_b = conv_b[:block.d_inner]
            block.conv_b = conv_b
            
            # Mamba 2 has an additional norm inside mixer (for SSM input)
            if self.is_mamba2 and f"{mixer_prefix}.norm.weight" in state_dict:
                block.mixer_norm_w = state_dict[f"{mixer_prefix}.norm.weight"].float().numpy()
            else:
                block.mixer_norm_w = None
            
            # Projections
            if f"{mixer_prefix}.in_proj.weight" in state_dict and self.is_mamba2:
                # Mamba 2
                in_proj_w = state_dict[f"{mixer_prefix}.in_proj.weight"].float().numpy()
                # Split: z, x, B, C, dt
                # z: d_inner (1536)
                # x: d_inner (1536)
                # B: ngroups * d_state (128)
                # C: ngroups * d_state (128)
                # dt: nheads (24)
                
                d_inner = block.d_inner
                d_state = block.d_state
                nheads = block.nheads
                
                # Order: z, x, B, C, dt
                z_end = d_inner
                x_end = z_end + d_inner
                B_end = x_end + d_state # ngroups=1
                C_end = B_end + d_state
                
                block.z_proj_w = in_proj_w[:z_end, :]
                block.x_proj_w = in_proj_w[z_end:x_end, :]
                block.B_proj_w = in_proj_w[x_end:B_end, :]
                block.C_proj_w = in_proj_w[B_end:C_end, :]
                block.dt_proj_w = in_proj_w[C_end:, :] # (nheads, d_model)
                
                block.dt_proj_b = state_dict[f"{mixer_prefix}.dt_bias"].float().numpy()
                
                # A_log and D in Mamba 2 are (nheads,) and need to be broadcast
                A_log_raw = state_dict[f"{mixer_prefix}.A_log"].float().numpy()  # (nheads,)
                D_raw = state_dict[f"{mixer_prefix}.D"].float().numpy()  # (nheads,)
                
                # Broadcast to (H, E, N) and (H, E)
                H, E, N = block.nheads, block.headdim, block.d_state
                # A_log: repeat for each headdim and state_dim
                block.A_log = np.tile(A_log_raw[:, None, None], (1, E, N))  # (H, E, N)
                # D: repeat for each headdim
                block.D = np.tile(D_raw[:, None], (1, E))  # (H, E)
                
                block.out_proj_w = state_dict[f"{mixer_prefix}.out_proj.weight"].float().numpy()
                
            else:
                # Mamba 1
                # in_proj for x and z
                in_proj_w = state_dict[f"{mixer_prefix}.in_proj.weight"].float().numpy()
                d_inner = block.d_inner
                block.x_proj_w = in_proj_w[:d_inner, :]
                block.z_proj_w = in_proj_w[d_inner:, :]

                # x_proj for dt, B, C
                x_proj_w = state_dict[f"{mixer_prefix}.x_proj.weight"].float().numpy()
                dt_proj_w = state_dict[f"{mixer_prefix}.dt_proj.weight"].float().numpy()
                dt_proj_b = state_dict[f"{mixer_prefix}.dt_proj.bias"].float().numpy()
                
                dt_rank = 48 # ceil(768/16) for d_model=768, d_state=16
                d_state = 16 # Mamba 1 d_state
                
                x_proj_dt = x_proj_w[:dt_rank, :]
                x_proj_B = x_proj_w[dt_rank:dt_rank+d_state, :]
                x_proj_C = x_proj_w[dt_rank+d_state:, :]
                
                block.dt_proj_w = dt_proj_w @ x_proj_dt
                block.dt_proj_b = dt_proj_b
                block.B_proj_w = x_proj_B
                block.C_proj_w = x_proj_C
                
                block.A_log = state_dict[f"{mixer_prefix}.A_log"].float().numpy()
                block.D = state_dict[f"{mixer_prefix}.D"].float().numpy()
                block.out_proj_w = state_dict[f"{mixer_prefix}.out_proj.weight"].float().numpy()
            
        # Final Norm
        self.norm_f_w = state_dict["backbone.norm_f.weight"].float().numpy()
        
        # LM Head - use same vocab size as embedding
        lm_head_w = state_dict["lm_head.weight"].float().numpy()
        self.lm_head_w = lm_head_w
        
        print("Weights loaded successfully.")

    def forward(self, input_ids):
        # input_ids: (B, L)
        # Embeddings
        x = self.embedding[input_ids] # (B, L, D)
        
        # Layers
        for block in self.layers:
            x = block.forward(x)
            
        # Final Norm
        # Need to implement RMSNorm manually or use a block's kernel?
        # The block has a pipe for rms_norm. We can reuse one block's pipe or implement numpy/metal one.
        # Let's just use numpy for the final norm for simplicity, or reuse the first block's kernel.
        # Using numpy for now to avoid managing another metal buffer context here.
        x = self._rms_norm_np(x, self.norm_f_w)
        
        # LM Head
        logits = x @ self.lm_head_w.T
        return logits

    def _rms_norm_np(self, x, w, eps=1e-5):
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        return x * w / np.sqrt(mean_sq + eps)

    def generate(self, prompt, max_length=50, temperature=1.0):
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        # (1, L)
        
        # Prefill
        # We can run the full forward on the prompt
        # But for generation we need the state.
        # The current block.forward() does NOT return state.
        # It re-calculates state from scratch.
        # To do autoregressive generation efficiently, we need `step` function or `selective_state_update`.
        
        # The user added `selective_state_update` to `mac_mamba_block.py`.
        # We should use that.
        
        # Strategy:
        # 1. Run full forward on prompt to get the initial state?
        #    Wait, `forward` in `mac_mamba_block.py` uses `ssm_scan_kernel`. It doesn't output the final state.
        #    So we can't easily get the state after the prompt.
        #    We might need to modify `mac_mamba_block.py` to return the final state, 
        #    OR we just run step-by-step for the whole prompt (slow but works),
        #    OR we implement a "forward_with_state" in the block.
        
        # Given I can't modify the block easily without asking (though I can), 
        # let's look at `ssm_scan_kernel`. It writes to `state` buffer but that buffer is local to forward.
        # "state = self.device.newBufferWithLength..."
        
        # I should probably modify `mac_mamba_block.py` to allow passing in/out state.
        # But for now, let's just implement a naive generation that re-runs forward? 
        # No, that's O(L^2).
        # The user specifically asked for "inference" and "selective_state_update".
        # So I must use `selective_state_update`.
        
        # To use `selective_state_update`, I need to maintain the state (B, H, E, N).
        # Initial state is zeros.
        # I can process the prompt token-by-token using `step`.
        
        print(f"Generating from prompt: '{prompt}'")
        tokens = input_ids[0].tolist()
        
        # Init states for all layers
        # SSM state is now managed by block.inference_cache
        # We only need to manage conv_state in python
        B = 1
        for block in self.layers:
            block.allocate_inference_cache(B)
            
        states = [None for _ in range(self.n_layers)] # conv_state initialized in step
        
        # Process prompt
        for i, token in enumerate(tokens):
            next_token, states = self.step(token, states)
            # We discard the output for the prompt tokens, except the last one if we were doing next-token prediction immediately.
            # But here we just want to warm up the state.
        
        # Generate
        generated = []
        curr_token = tokens[-1] # The last token of prompt is the input for the first generated token
        
        # Wait, `step` logic:
        # Input: token `t`. Output: logits for `t+1`.
        # So for the prompt [t1, t2, t3], we feed t1 -> get state1, (ignoring out), feed t2 -> state2, feed t3 -> state3, logits3.
        # logits3 gives us t4.
        
        # So we need to feed all tokens.
        
        # Feed prompt
        curr_token = tokens[0]
        for t in tokens[1:]:
            _, states = self.step(curr_token, states)
            curr_token = t
            
        # Now curr_token is the last token of the prompt.
        # We are ready to generate.
        
        for _ in range(max_length):
            logits, states = self.step(curr_token, states)
            # logits: (1, 1, vocab_size)
            next_token_id = np.argmax(logits[0, 0]) # Greedy
            generated.append(next_token_id)
            curr_token = next_token_id
            print(self.tokenizer.decode([curr_token]), end="", flush=True)
            
        print("\nGeneration done.")
        return self.tokenizer.decode(generated)

    def step(self, token, states):
        # token: int or (1,)
        # states: list of (B, H, E, N)
        
        x = self.embedding[np.array([[token]])] # (1, 1, D)
        
        # We need to handle the residual connection and normalization carefully.
        # Block forward:
        # 1. Norm
        # 2. Linear (x, z)
        # 3. Conv
        # 4. SSM
        # 5. Gating
        # 6. Out proj
        # 7. Residual
        
        # For step-wise, we need to manually implement the block logic using `selective_state_update`.
        # AND we need to handle the Conv1d state!
        # The `selective_state_update` only handles the SSM state.
        # What about the Conv1d state?
        # The user's `mac_mamba_block.py` does NOT seem to have a `conv_state_update` or similar.
        # It only has `selective_state_update`.
        # Standard Mamba step:
        # 1. Update conv state (shift and add new input).
        # 2. Compute conv output.
        # 3. ...
        
        new_states = []
        
        for i, block in enumerate(self.layers):
            # SSM state is managed by block.inference_cache
            # We only need to manage conv_state here (or move it to block too? For now keep here)
            conv_state = states[i] if states[i] is not None else np.zeros((1, block.d_inner, block.d_conv), dtype='f4')
            
            # 1. RMSNorm
            residual = x
            x = self._rms_norm_np(x, block.norm_w)
            
            if self.is_mamba2:
                # Mamba 2 Flow
                # Projections from Input (x)
                z = x @ block.z_proj_w.T
                x_ssm_in = x @ block.x_proj_w.T
                B_proj = x @ block.B_proj_w.T
                C_proj = x @ block.C_proj_w.T
                dt = x @ block.dt_proj_w.T + block.dt_proj_b
                
                # Conv1d on x_ssm_in
                conv_state = np.roll(conv_state, -1, axis=-1)
                conv_state[:, :, -1] = x_ssm_in.squeeze(1)
                
                conv_out = np.sum(conv_state * block.conv_w, axis=-1) + block.conv_b
                conv_out = conv_out[:, np.newaxis, :]
                conv_out = conv_out * (1 / (1 + np.exp(-conv_out))) # SiLU
                
                # Apply mixer norm if exists (Mamba 2 specific)
                if block.mixer_norm_w is not None:
                    conv_out = self._rms_norm_np(conv_out, block.mixer_norm_w)
                
                # SSM
                H = block.nheads
                E = block.headdim
                
                x_ssm = conv_out.reshape(1, H, E)
                
                # dt is (1, 1, nheads). Need to broadcast to (1, H, E)
                # Metal kernel expects (B, H, E). We repeat per headdim.
                dt_ssm = np.repeat(dt, E, axis=-1).reshape(1, H, E)
                
                B_ssm = B_proj.reshape(1, 1, block.d_state)
                C_ssm = C_proj.reshape(1, 1, block.d_state)
                z_ssm = z.reshape(1, H, E)
                
            else:
                # Mamba 1 Flow
                xz = np.concatenate([
                    x @ block.x_proj_w.T,
                    x @ block.z_proj_w.T
                ], axis=-1)
                
                x_curr = xz[:, :, :block.d_inner]
                z_curr = xz[:, :, block.d_inner:]
                
                conv_state = np.roll(conv_state, -1, axis=-1)
                conv_state[:, :, -1] = x_curr.squeeze(1)
                
                conv_out = np.sum(conv_state * block.conv_w, axis=-1) + block.conv_b
                conv_out = conv_out[:, np.newaxis, :]
                conv_out = conv_out * (1 / (1 + np.exp(-conv_out))) # SiLU
                
                dt = conv_out @ block.dt_proj_w.T + block.dt_proj_b
                B_proj = conv_out @ block.B_proj_w.T
                C_proj = conv_out @ block.C_proj_w.T
                
                H = block.nheads
                E = block.headdim
                
                x_ssm = conv_out.reshape(1, H, E)
                dt_ssm = dt.reshape(1, H, E)
                B_ssm = B_proj.reshape(1, 1, block.d_state)
                C_ssm = C_proj.reshape(1, 1, block.d_state)
                z_ssm = z_curr.reshape(1, H, E)
            
            # Use fast inference step
            # This returns a Metal buffer
            out_buf = block.step_inference(x_ssm, dt_ssm, B_ssm, C_ssm, z_ssm)
            
            # Read back result (Sync point for now, as next layer needs it on CPU)
            # Optimization: We could chain layers on GPU if we moved everything there.
            # For now, we accept this sync.
            y_ssm = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype='f4').reshape(1, H, E)
            
            y_out = y_ssm.reshape(1, 1, block.d_inner)
            
            # 7. Out Proj
            out = y_out @ block.out_proj_w.T
            
            # Residual
            x = out + residual
            
            new_states.append(conv_state)
            
        # Final Norm
        x = self._rms_norm_np(x, self.norm_f_w)
        
        # Head
        logits = x @ self.lm_head_w.T
        
        return logits, new_states

