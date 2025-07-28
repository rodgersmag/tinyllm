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
        self.qkv = nn.Linear(config.n_emb, 3 * config.n_emb)
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", mx.tril(mx.ones((config.ctx_len, config.ctx_len))))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_size).transpose(0, 3, 1, 2)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q @ k.transpose(0, -1)) / (self.head_size ** 0.5)
        attn = attn * self.causal_mask[:T, :T]
        attn = nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(0, 2, 1).reshape(B, T, C)
        return self.proj(out)

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
    
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(mx.arange(T))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)