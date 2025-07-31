import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np

# --- Configuration ---
# This should match the configuration used for training in train.py
CONFIG = {
    'BLOCK_SIZE': 256,
    'VOCAB_SIZE': 50257,  # Default for gpt2, ensure it matches tokenizer
    'N_EMBD': 384,
    'N_HEAD': 6,
    'N_LAYER': 6,
    'DROPOUT': 0.1,
    'MODEL_PATH': 'model.pth',
}

# --- Model Definition (must match train.py) ---

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

# --- Text Generation ---

def generate_text(model: nn.Module, tokenizer, device: torch.device, prompt: str, max_length: int, temperature: float, top_k: int) -> str:
    """Generate text from the loaded model"""
    model.eval()
    
    try:
        prompt_tokens = tokenizer.encode(prompt)
        generated = prompt_tokens.copy()
        end_of_text_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        with torch.no_grad():
            for _ in range(max_length):
                context = generated[-CONFIG['BLOCK_SIZE']:]
                x = torch.tensor([context], dtype=torch.long, device=device)
                
                logits = model(x)
                next_logits = logits[0, -1, :] / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
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
        
    except Exception as e:
        return f"Error generating text: {e}"

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained TinyGPT model.")
    parser.add_argument('--prompt', type=str, nargs='+', default=[
        "Once upon a time",
        "A brave knight",
        "In a magical forest",
        "The friendly dragon",
        "There was a little girl",
        "A boy and a girl in Africa"
    ], help='One or more starting prompts for generation.')
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum number of tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling.')
    parser.add_argument('--model_path', type=str, default=CONFIG['MODEL_PATH'], help='Path to the saved model weights.')
    parser.add_argument('--output_file', type=str, default='output.txt', help='File to save the generated stories.')
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = tiktoken.encoding_for_model("gpt2")
    CONFIG['VOCAB_SIZE'] = tokenizer.n_vocab

    # Load model
    model = TinyGPT(CONFIG).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}. Please run train.py first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate text for each prompt and save to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(args.prompt):
            print(f"\nGenerating for prompt {i+1}/{len(args.prompt)}: '{prompt}'")
            print("-" * 40)
            
            story = generate_text(
                model, tokenizer, device,
                prompt=prompt,
                max_length=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            print(story)
            f.write(f"--- PROMPT: {prompt} ---\n")
            f.write(story)
            f.write("\n\n" + "=" * 60 + "\n\n")
    
    print(f"\nAll stories saved to {args.output_file}")

if __name__ == "__main__":
    main()
