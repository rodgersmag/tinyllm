import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import numpy as np
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os

### hyper params (must match training)
ctx_len = 8  # Must match the ctx_len used in train.py line 38
n_emb = 128
dropout = 0.1
head_size = 128
n_heads = 4
n_layers = 3

### Load tokenizer
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'input.txt')

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
itos = {i:c for i,c in enumerate(vocab)}
stoi = {c:i for i,c in enumerate(vocab)}

def encode(x):
    """Encode string to token ids, handling unknown characters"""
    return [stoi.get(c, 0) for c in x]  # Default to 0 for unknown chars

decode = lambda x: ''.join([itos[i] for i in x])

### Model Definition (same as training)
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.wpe = nn.Embedding(8, n_emb)  # Use 8 to match training
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)],
        )
        self.ln_f = nn.LayerNorm(dims=n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
    
    def __call__(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(mx.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= ctx_len else idx[:, -ctx_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = mx.softmax(logits, axis=-1)
            idx_next = mx.random.categorical(mx.log(probs))
            idx_next = mx.expand_dims(idx_next, axis=1)
            idx = mx.concatenate([idx, idx_next], axis=1)
        return idx

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size, bias=False)
        indices = mx.arange(8)  # Use 8 to match training
        mask = indices[:, None] < indices[None]
        self._causal_mask = mask * -1e9
        self.c_proj = nn.Linear(head_size, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def __call__(self, x):
        B, T, C = x.shape
        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        mha_shape = (B, T, n_heads, head_size//n_heads)
        K = mx.as_strided(K, (mha_shape)).transpose([0, 2, 1, 3])
        Q = mx.as_strided(Q, (mha_shape)).transpose([0, 2, 1, 3])
        V = mx.as_strided(V, (mha_shape)).transpose([0, 2, 1, 3])
        attn_weights = (Q @ K.transpose([0, 1, 3, 2])) / math.sqrt(Q.shape[-1])
        attn_weights = attn_weights + self._causal_mask[:T, :T]
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        o = (attn_weights @ V)
        o = o.transpose([0, 2, 1, 3]).reshape((B, T, head_size))
        o = self.c_proj(self.resid_dropout(o))
        return o

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.mha = MultiHeadAttention()
        self.ln_1 = nn.LayerNorm(dims=n_emb)
        self.ln_2 = nn.LayerNorm(dims=n_emb)
    
    def __call__(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Initialize FastAPI app
app = FastAPI(title="MLX GPT-2 Inference API")

# Load model at startup
model_path = os.path.join(script_dir, 'model.npz')
model = GPT()
model.load_weights(model_path)
model.eval()
mx.eval(model.parameters())
print(f"Model loaded from {model_path}")

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: Optional[str] = Field(default="", description="Starting text for generation")
    max_tokens: int = Field(default=100, ge=1, le=1000, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int

@app.get("/")
async def root():
    return {
        "message": "MLX GPT-2 Inference API",
        "endpoints": {
            "/generate": "POST - Generate text from prompt",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        # Prepare prompt
        if request.prompt:
            # Filter prompt to only include known characters
            filtered_prompt = ''.join(c for c in request.prompt if c in stoi)
            if filtered_prompt != request.prompt:
                print(f"Warning: Some characters were filtered from prompt")
            
            context = mx.array(encode(filtered_prompt)).reshape(1, -1)
        else:
            context = mx.zeros((1, 1), dtype=mx.int32)
        
        # Generate text
        generated = model.generate(context, request.max_tokens, temperature=request.temperature)
        output = decode(generated[0].tolist())
        
        # Calculate actual tokens generated (excluding prompt)
        prompt_tokens = len(encode(request.prompt)) if request.prompt else 0
        total_tokens = generated.shape[1]
        tokens_generated = total_tokens - prompt_tokens
        
        return GenerationResponse(
            generated_text=output,
            prompt=request.prompt,
            tokens_generated=tokens_generated
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)