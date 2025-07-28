import mlx.core as mx
import mlx.nn as nn

class CharTokenizer:
    def __init__(self, data):
        self.chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return "".join([self.itos[t] for t in tokens])

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.n_emb = config.n_emb
        self.qkv = nn.Linear(config.n_emb, 3 * config.n_emb)
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        B, T, C = x.shape
        
        # Create causal mask for current sequence length
        causal_mask = mx.tril(mx.ones((T, T)))
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_size)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # (B, 3, n_heads, T, head_size)

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, n_heads, T, head_size)

        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * (self.head_size ** -0.5)  # (B, n_heads, T, T)
        
        # Apply causal mask
        attn = mx.where(causal_mask, attn, -mx.inf)
        attn = nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, n_heads, T, head_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)
        
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.fc2 = nn.Linear(4 * config.n_emb, config.n_emb)
        self.dropout = nn.Dropout(config.dropout)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_emb)
        self.pos_emb = nn.Embedding(config.ctx_len, config.n_emb)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.ln_f = nn.LayerNorm(config.n_emb)
        self.head = nn.Linear(config.n_emb, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        B, T = x.shape
        
        # Token and position embeddings
        tok_emb = self.token_emb(x)  # (B, T, n_emb)
        pos_emb = self.pos_emb(mx.arange(T))  # (T, n_emb)
        x = self.dropout(tok_emb + pos_emb)  # Broadcasting: (B, T, n_emb) + (T, n_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits