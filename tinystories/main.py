# Gemini-generated script for Google Colab.
# This script combines all steps into a single cell for easy execution.

# --- Cell 1: Environment and GPU check ---
print("--- Cell 1: Environment and GPU check ---")
import subprocess
import sys

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

# Check GPU availability
print("Checking for GPU...")
success, stdout, stderr = run_command("nvidia-smi")
if success:
    print("GPU detected:")
    print(stdout)
else:
    print("nvidia-smi not available or no GPU detected. This is fine on CPU.")

success, stdout, stderr = run_command("nvcc --version")
if success:
    print("NVCC version:")
    print(stdout)
else:
    print("nvcc not available. This is fine on CPU.")
print("-" * 20)


# --- Cell 2: Install packages and handle kernel restart ---
print("\n--- Cell 2: Install packages and handle kernel restart ---")
def install_and_restart_if_needed():
    """Install packages and force a kernel restart if numpy version is incorrect."""
    
    try:
        import numpy
        # If numpy is the correct version, we don't need to do anything.
        if numpy.__version__ == '1.26.4':
            print("Correct NumPy version (1.26.4) is already installed. Proceeding.")
            return
    except (ImportError, AttributeError):
        # If numpy is not installed or something is wrong, we proceed to install.
        pass

    print("Incorrect NumPy version detected or packages missing. Installing...")
    
    # Use uv to install packages
    run_command("pip install -q uv")
    
    packages = [
        "numpy==1.26.4", "torch", "torchvision", "torchaudio",
        "datasets", "tiktoken", "transformers", "tqdm", "matplotlib"
    ]
    command = f"uv pip install --reinstall {' '.join(packages)}"
    
    success, stdout, stderr = run_command(command)
    
    if success:
        print("Packages installed successfully.")
        print("\n" + "="*60)
        print("IMPORTANT: A KERNEL RESTART IS REQUIRED.")
        print("The kernel will now be automatically restarted to load the correct packages.")
        print("PLEASE RE-RUN THIS CELL once the kernel has restarted.")
        print("="*60)
        import os
        os.kill(os.getpid(), 9)
    else:
        print("Package installation failed.")
        print(f"--- stdout ---\n{stdout}")
        print(f"--- stderr ---\n{stderr}")
        raise RuntimeError("Failed to install required packages.")

# This function will either do nothing or install and then crash the kernel.
# The rest of the script will only run correctly on the second execution of the cell.
install_and_restart_if_needed()
print("-" * 20)


# --- Cell 3: Python imports and environment setup ---
print("\n--- Cell 3: Python imports and environment setup ---")
import os
import time
import json
import gc
from typing import List, Tuple, Optional

# Work around NumPy 2.x ABI issues
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
import tiktoken
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Print versions for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Torch version: {torch.__version__}")

try:
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
except Exception as e:
    print(f"CUDA query failed: {e}")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cuDNN benchmarking for fixed input shapes
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"Environment setup complete! Device: {device}")
print("-" * 20)


# --- Cell 4: Load TinyStories dataset ---
print("\n--- Cell 4: Load TinyStories dataset ---")
def load_tinystories_dataset(max_size: int = 20000) -> Dataset:
    """Load and prepare TinyStories dataset"""
    print("Loading TinyStories dataset...")
    try:
        dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=False)
        
        # Select subset
        dataset_size = min(max_size, len(dataset))
        dataset = dataset.select(range(dataset_size))
        
        print(f"Dataset loaded: {len(dataset)} stories")
        print("Sample story:")
        print(dataset[0]['text'][:200] + "...")
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

dataset = load_tinystories_dataset(max_size=20000)
print("-" * 20)


# --- Cell 5: Configuration and tokenizer setup ---
print("\n--- Cell 5: Configuration and tokenizer setup ---")
# Initialize tokenizer
try:
    tokenizer = tiktoken.encoding_for_model("gpt2")
    VOCAB_SIZE = tokenizer.n_vocab
    print(f"Tokenizer loaded successfully. Vocab size: {VOCAB_SIZE}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Training configuration (optimized for T4 16GB)
CONFIG = {
    'BLOCK_SIZE': 256,
    'BATCH_SIZE': 8,  # Reduced to prevent OOM
    'VOCAB_SIZE': VOCAB_SIZE,
    'MAX_DATASET_SIZE': 15000,  # Reduced for memory efficiency
    'LEARNING_RATE': 3e-4,
    'WEIGHT_DECAY': 0.1,
    'DROPOUT': 0.1,
    'N_EMBD': 384,
    'N_HEAD': 6,
    'N_LAYER': 6,
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("-" * 20)


# --- Cell 6: Efficient tokenization ---
print("\n--- Cell 6: Efficient tokenization ---")
def tokenize_batch(texts: List[str], tokenizer, block_size: int) -> List[List[int]]:
    """Tokenize a batch of texts efficiently"""
    tokenized = []
    for text in texts:
        try:
            tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            # Truncate if too long
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            if len(tokens) > 10:  # Only keep stories with reasonable length
                tokenized.append(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            continue
    return tokenized

def process_dataset_efficiently(dataset: Dataset, config: dict) -> List[List[int]]:
    """Process dataset with memory efficiency"""
    print("Processing dataset efficiently...")
    
    all_tokenized = []
    batch_size = 500  # Process in batches to manage memory
    
    total_stories = min(config['MAX_DATASET_SIZE'], len(dataset))
    
    for i in tqdm(range(0, total_stories, batch_size), desc="Tokenizing"):
        end_idx = min(i + batch_size, total_stories)
        batch_texts = [dataset[j]['text'] for j in range(i, end_idx)]
        
        # Tokenize batch
        tokenized_batch = tokenize_batch(batch_texts, tokenizer, config['BLOCK_SIZE'])
        all_tokenized.extend(tokenized_batch)
        
        # Clear memory periodically
        if i % (batch_size * 4) == 0:
            gc.collect()
    
    print(f"Tokenization complete! Processed {len(all_tokenized)} valid stories")
    return all_tokenized

# Process the dataset
tokenized_data = process_dataset_efficiently(dataset, CONFIG)
print("-" * 20)


# --- Cell 7: Create training batches ---
print("\n--- Cell 7: Create training batches ---")
def create_training_data(tokenized_stories: List[List[int]], 
                        batch_size: int, 
                        block_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create training batches from tokenized stories"""
    print("Creating training batches...")
    
    # Flatten all tokens
    all_tokens = []
    for story in tokenized_stories:
        all_tokens.extend(story)
        all_tokens.append(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0])  # Add separator
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Create batches
    batches = []
    max_batches = 300  # Limit for training time
    
    # Generate batches with sliding window
    for i in range(0, len(all_tokens) - block_size - 1, block_size):
        if len(batches) >= max_batches:
            break
            
        batch_x = []
        batch_y = []
        
        for b in range(batch_size):
            start_idx = i + b * (len(all_tokens) // batch_size)
            if start_idx + block_size >= len(all_tokens):
                break
                
            x = all_tokens[start_idx:start_idx + block_size]
            y = all_tokens[start_idx + 1:start_idx + block_size + 1]
            
            if len(x) == block_size and len(y) == block_size:
                batch_x.append(x)
                batch_y.append(y)
        
        if len(batch_x) == batch_size:
            x_tensor = torch.tensor(batch_x, dtype=torch.long)
            y_tensor = torch.tensor(batch_y, dtype=torch.long)
            batches.append((x_tensor, y_tensor))
    
    print(f"Created {len(batches)} training batches")
    return batches

# Create training batches
train_batches = create_training_data(tokenized_data, CONFIG['BATCH_SIZE'], CONFIG['BLOCK_SIZE'])

# Free memory
del tokenized_data, dataset
gc.collect()

print(f"Training data ready: {len(train_batches)} batches")
print("-" * 20)


# --- Cell 8: PyTorch Model Definition ---
print("\n--- Cell 8: PyTorch Model Definition ---")
class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer('mask', torch.tril(torch.ones(1024, 1024)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(y)

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    """Improved TinyGPT model"""
    def __init__(self, vocab_size: int, n_embd: int = 384, n_head: int = 6, 
                 n_layer: int = 6, block_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, T = x.shape
        device = x.device
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# Initialize model
model = TinyGPT(
    vocab_size=CONFIG['VOCAB_SIZE'],
    n_embd=CONFIG['N_EMBD'],
    n_head=CONFIG['N_HEAD'],
    n_layer=CONFIG['N_LAYER'],
    block_size=CONFIG['BLOCK_SIZE'],
    dropout=CONFIG['DROPOUT']
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params/1e6:.1f}M parameters on {device}")
print("-" * 20)


# --- Cell 9: Training setup ---
print("\n--- Cell 9: Training setup ---")
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=CONFIG['LEARNING_RATE'], 
    weight_decay=CONFIG['WEIGHT_DECAY']
)

def train_step(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
    """Perform one training step"""
    model.train()
    optimizer.zero_grad()
    
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, CONFIG['VOCAB_SIZE']), y.view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()
print("Training components ready.")
print("-" * 20)


# --- Cell 10: Training loop ---
print("\n--- Cell 10: Training loop ---")
def train_model(model: nn.Module, train_batches: List, target_minutes: int = 30):
    """Train the model for a specified time"""
    print(f"Starting training for ~{target_minutes} minutes...")
    
    train_losses = []
    start_time = time.time()
    target_time = target_minutes * 60
    
    step = 0
    epoch = 0
    
    while (time.time() - start_time) < target_time and epoch < 10:
        epoch_start = time.time()
        epoch_losses = []
        
        print(f"\n--- Epoch {epoch + 1} ---")
        
        for batch in tqdm(train_batches, desc=f"Epoch {epoch + 1}"):
            if (time.time() - start_time) >= target_time:
                print("\nReached time limit!")
                break
            
            loss = train_step(model, batch)
            train_losses.append(loss)
            epoch_losses.append(loss)
            step += 1
            
            if step % 20 == 0:
                elapsed = time.time() - start_time
                remaining = max(0, target_time - elapsed)
                print(f"\nStep {step}, Loss: {loss:.4f}, "
                      f"Elapsed: {elapsed/60:.1f}m, Remaining: {remaining/60:.1f}m")
        
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} completed in {epoch_time/60:.1f}m, "
                  f"Avg Loss: {avg_loss:.4f}")
        
        epoch += 1
        
        if (time.time() - start_time) >= target_time:
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Total steps: {step}")
    if train_losses:
        print(f"Final loss: {train_losses[-1]:.4f}")
    
    return train_losses

# Start training
train_losses = train_model(model, train_batches, target_minutes=30)
print("-" * 20)


# --- Cell 11: Plot training progress ---
print("\n--- Cell 11: Plot training progress ---")
def plot_training_progress(losses: List[float]):
    """Plot training loss"""
    if not losses:
        print("No training losses to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Raw losses
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Moving average
    if len(losses) > 50:
        window = 50
        moving_avg = []
        for i in range(window, len(losses)):
            moving_avg.append(sum(losses[i-window:i]) / window)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(window, len(losses)), moving_avg)
        plt.title(f'Training Loss (Moving Average, window={window})')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    if losses:
        print(f"Loss improvement: {losses[0]:.4f} ", "\u2192", f" {losses[-1]:.4f}")

plot_training_progress(train_losses)
print("-" * 20)


# --- Cell 12: Text generation ---
print("\n--- Cell 12: Text generation ---")
def generate_text(model: nn.Module, 
                 prompt: str = "Once upon a time", 
                 max_length: int = 200, 
                 temperature: float = 0.8,
                 top_k: int = 50) -> str:
    """Generate text from the trained model"""
    model.eval()
    
    try:
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            return "Error: Empty prompt after tokenization"
        
        generated = prompt_tokens.copy()
        
        # Pre-encode the end-of-text token to use for comparison
        end_of_text_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        with torch.no_grad():
            for _ in range(max_length):
                # Get context (last block_size tokens)
                context = generated[-CONFIG['BLOCK_SIZE']:]
                x = torch.tensor([context], dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(x)
                next_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1).item()
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                # Stop at end of text token
                if next_token == end_of_text_token_id:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated)
        return generated_text
        
    except Exception as e:
        return f"Error generating text: {e}"

print("Text generation function is ready.")
print("-" * 20)


# --- Cell 13: Generate sample stories ---
print("\n--- Cell 13: Generate sample stories ---")
def test_model_generation():
    """Test the trained model with various prompts"""
    prompts = [
        "Once upon a time",
        "There was a little girl",
        "In a magical forest",
        "A brave knight",
        "The friendly dragon"
    ]
    
    print("\nGenerated Stories from Trained Model:")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Story {i} ---")
        print(f"Prompt: '{prompt}'")
        print("-" * 40)
        
        story = generate_text(
            model, 
            prompt=prompt, 
            max_length=150, 
            temperature=0.8,
            top_k=50
        )
        
        if story.startswith("Error"):
            print(story)
        else:
            print(story)
        print()
    
    print("=" * 60)

# Test generation
test_model_generation()

print("\nTraining and story generation completed successfully!")
