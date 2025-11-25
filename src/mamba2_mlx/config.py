
class Mamba2Config:
    def __init__(self, d_model=768, n_layer=24, vocab_size=50277, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, tie_word_embeddings=True, rms_norm_eps=1e-5):
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.tie_word_embeddings = tie_word_embeddings
        self.rms_norm_eps = rms_norm_eps
