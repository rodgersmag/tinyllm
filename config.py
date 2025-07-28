
#### 2. `config.py`

class GPTConfig:
    def __init__(self):
        self.ctx_len = 64          # Context length (reduced for memory)
        self.n_emb = 64            # Embedding size
        self.n_heads = 4           # Number of attention heads
        self.n_layers = 2          # Number of transformer layers
        self.head_size = 64        # Head size (n_emb / n_heads)
        self.dropout = 0.1         # Dropout rate
        self.batch_size = 8        # Batch size (reduced for memory)
        self.lr = 1e-3             # Learning rate
        self.num_epochs = 10       # Number of epochs
        self.vocab_size = None     # Set dynamically by tokenizer