import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np

# --- Configuration ---
CONFIG = {
    'BLOCK_SIZE': 256,
    'VOCAB_SIZE': 50257,
    'N_EMBD': 384,
    'N_HEAD': 6,
    'N_LAYER': 6,
    'DROPOUT': 0.1,
    'MODEL_PATH': 'model.pth',
}

# --- Model Definition (must match train.py and inference.py) ---

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd, self.head_dim = n_head, n_embd, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(CONFIG['BLOCK_SIZE'], CONFIG['BLOCK_SIZE'])))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['VOCAB_SIZE'], config['N_EMBD'])
        self.position_embedding = nn.Embedding(config['BLOCK_SIZE'], config['N_EMBD'])
        self.blocks = nn.ModuleList([
            TransformerBlock(config['N_EMBD'], config['N_HEAD'], config['DROPOUT'])
            for _ in range(config['N_LAYER'])
        ])
        self.ln_f = nn.LayerNorm(config['N_EMBD'])
        self.head = nn.Linear(config['N_EMBD'], config['VOCAB_SIZE'], bias=False)
        self.dropout = nn.Dropout(config['DROPOUT'])

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global model, tokenizer, device
    
    # Startup: Load the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"API using device: {device}")
    
    tokenizer = tiktoken.encoding_for_model("gpt2")
    CONFIG['VOCAB_SIZE'] = tokenizer.n_vocab
    
    model = TinyGPT(CONFIG).to(device)
    
    model_path = CONFIG['MODEL_PATH']
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}. Please run train.py first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    yield  # Application is running
    
    # Shutdown: Cleanup resources (if needed)
    print("Shutting down API...")

# --- API Setup ---
app = FastAPI(title="TinyStories GPT Inference API", lifespan=lifespan)

# Global variables for model, tokenizer, and device
model = None
tokenizer = None
device = None

# --- Request/Response Models ---
class GenerationRequest(BaseModel):
    prompt: str = Field(default="Once upon a time", description="Starting text for generation.")
    max_tokens: int = Field(default=150, ge=10, le=500, description="Maximum number of new tokens to generate.")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature for creativity.")
    top_k: int = Field(default=50, ge=0, description="Top-k filtering for sampling.")

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int

# --- Text Generation Logic ---
def generate_text_api(req: GenerationRequest) -> str:
    """Generate text using the loaded model for the API."""
    model.eval()
    allowed_special = {"<|endoftext|>"}
    prompt_tokens = tokenizer.encode(req.prompt, allowed_special=allowed_special)
    generated = prompt_tokens.copy()
    end_of_text_token_id = tokenizer.encode("<|endoftext|>", allowed_special=allowed_special)[0]

    with torch.no_grad():
        for _ in range(req.max_tokens):
            context = generated[-CONFIG['BLOCK_SIZE']:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            logits = model(x)
            next_logits = logits[0, -1, :] / req.temperature
            
            if req.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, req.top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()
                next_token = top_k_indices[next_token_idx].item()
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            if next_token == end_of_text_token_id:
                break
    
    return tokenizer.decode(generated)

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the TinyStories GPT Inference API!"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_story(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please wait or check server logs.")
    
    try:
        full_text = generate_text_api(request)
        
        # Exclude the prompt from the final generated text for token count
        allowed_special = {"<|endoftext|>"}
        prompt_tokens_len = len(tokenizer.encode(request.prompt, allowed_special=allowed_special))
        generated_tokens_len = len(tokenizer.encode(full_text, allowed_special=allowed_special))
        
        return GenerationResponse(
            generated_text=full_text,
            prompt=request.prompt,
            tokens_generated=generated_tokens_len - prompt_tokens_len
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
