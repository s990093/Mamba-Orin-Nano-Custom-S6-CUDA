
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
from .config import Mamba2Config
from .modules import Mamba2Block, RMSNorm

class Mamba2Model(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        print(f"Creating {config.n_layer} layers...")
        self.layers = []
        for i in tqdm(range(config.n_layer), desc="Initializing Layers"):
            self.layers.append(Mamba2Block(config, i))
            
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight # Tie weights

    def __call__(self, x, cache=None):
        x = self.embedding(x)
        
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, cache=layer_cache)
            
        x = self.norm_f(x)
        return self.lm_head(x)
        
    def make_cache(self, batch_size):
        cache = []
        for _ in range(self.config.n_layer):
            # Conv state: (B, C, K)
            # SSM state: (B, H, N, P)
            d_inner = self.config.d_model * self.config.expand
            nheads = d_inner // self.config.headdim
            
            conv_dim = d_inner + 2 * self.config.ngroups * self.config.d_state
            
            conv_state = mx.zeros((batch_size, conv_dim, self.config.d_conv))
            ssm_state = mx.zeros((batch_size, nheads, self.config.d_state, self.config.headdim))
            cache.append([conv_state, ssm_state])
        return cache
