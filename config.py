class GPTConfig:
    def __init__(self):
        self.ctx_len = 128          # Increased context length for better learning
        self.n_emb = 128            # Increased embedding size
        self.n_heads = 8            # More attention heads
        self.n_layers = 4           # More transformer layers
        self.head_size = self.n_emb // self.n_heads  # Correctly calculate head size
        self.dropout = 0.1          # Dropout rate
        self.batch_size = 16        # Increased batch size (will be auto-optimized)
        self.lr = 5e-4              # Slightly lower learning rate for stability
        self.num_epochs = 5         # Fewer epochs since we're using more data per batch
        self.vocab_size = None      # Set dynamically by tokenizer