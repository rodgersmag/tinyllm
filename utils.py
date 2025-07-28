import mlx.core as mx
import mlx.nn as nn

class CharTokenizer:
    def __init__(self, data):
        self.chars = sorted(list(set("".join(data))))
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
        self.qkv = nn.Linear(config.n_emb, 3 * config.n_emb)  # Input: C, Output: 3 * C
        self.proj = nn.Linear(config.n_emb, config.n_emb)    # Projection back to C
        self.dropout = nn.Dropout(config.dropout)
        self.causal_mask = mx.tril(mx.ones((config.ctx_len, config.ctx_len)))

    def forward(self, x):
        B, T, C = x.shape
        # Compute Q, K, V in one go
        qkv = self.qkv(x)  # Shape: (B, T, 3 * C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_size)  # Shape: (B, T, 3, n_heads, head_size)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # Shape: (B, 3, n_heads, T, head_size)

        # Split into Q, K, V
        q = qkv[:, 0]  # (B, n_heads, T, head_size)
        k = qkv[:, 1]  # (B, n_heads, T, head_size)
        v = qkv[:, 2]  # (B, n_heads, T, head_size)

        # Compute attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) / (self.head_size ** 0.5)  # (B, n_heads, T, T)
        attn = attn * self.causal_mask[:T, :T]  # Apply causal mask
        attn = nn.softmax(attn, axis=-1)        # Softmax over last dimension
        attn = self.dropout(attn)               # Apply dropout

        # Compute output
        out = attn @ v  # (B, n_heads, T, head_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)  # Back to (B, T, C)
        out = self.proj(out)  # Project back to embedding dimension
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.fc2 = nn.Linear(4 * config.n_emb, config.n_emb)
        self.dropout = nn.Dropout(config.dropout)
    
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
    
    def forward(self, x):
        x = x + self.attn.forward(self.ln1(x))  # Fixed: Use forward method
        x = x + self.ff.forward(self.ln2(x))    # Fixed: Use forward method
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
    
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(mx.arange(T))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block.forward(x)  # Fixed: Use block.forward(x) instead of block(x)
        x = self.ln_f(x)
        return self.head(x)