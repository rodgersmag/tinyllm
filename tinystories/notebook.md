# Cell 1: Environment and GPU check
!nvidia-smi || echo "nvidia-smi not available"
!nvcc --version || echo "nvcc not available"

# Cell 2: Install required packages (optimized for Colab T4)
!pip -q install --upgrade pip

# Pin NumPy to <2 to avoid ABI issues on Colab with some wheels (e.g., PyTorch add-ons)
# Install PyTorch via the official index (Colab typically has a good CUDA build preinstalled)
!pip -q install "numpy<2" datasets tiktoken transformers tqdm matplotlib

# Reinstall torch after numpy pin to ensure compatible ABI if needed
# If Colab already has torch with CUDA, this will no-op to the same version.
!pip -q install --upgrade --no-cache-dir torch

# Cell 3: Python imports and environment setup
import os
import time
import json

# Work around NumPy 2.x ABI issues by ensuring we import numpy after pinning
import numpy as np

# Matplotlib sometimes depends on numpy ABI as well
import matplotlib
import matplotlib.pyplot as plt

from datasets import load_dataset
import tiktoken
from tqdm import tqdm

# Import torch after numpy is ready to avoid numpy-related init warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Explicitly print versions to aid debugging environment issues
print(f"Python version: {os.sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Torch version: {torch.__version__}")
try:
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
except Exception as _e:
    print(f"CUDA query failed: {_e}")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cuDNN benchmarking for fixed input shapes
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"Environment setup complete! Device: {device}")

# Cell 4: Load TinyStories and sample
print("Loading TinyStories dataset...")
dataset = load_dataset('roneneldan/TinyStories', split='train')

DATASET_SIZE = 50000  # upper bound; we will cap below
dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))

print(f"Dataset size: {len(dataset)}")
print("Sample story:")
print(dataset[0]['text'][:200] + "...")

# Cell 5: Tokenizer and config
tokenizer = tiktoken.encoding_for_model("gpt2")
VOCAB_SIZE = tokenizer.n_vocab

# Training config optimized for T4 (16 GB)
BLOCK_SIZE = 256
BATCH_SIZE = 12  # tuned for memory; adjust if OOM, e.g., 8-12
TOKENIZATION_BATCH_SIZE = 200
MAX_DATASET_SIZE = 20000  # smaller subset for ~30 minutes on T4

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
DROPOUT = 0.1

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Using reduced dataset size: {MAX_DATASET_SIZE} stories")
print(f"Tokenization batch size: {TOKENIZATION_BATCH_SIZE}")

# Cell 6: Tokenization utilities (memory efficient)
def tokenize_stories_efficient(examples):
    tokenized = []
    for story in examples['text']:
        try:
            tokens = tokenizer.encode(story, allowed_special={"<|endoftext|>"})
            if len(tokens) > BLOCK_SIZE:
                tokens = tokens[:BLOCK_SIZE]
            tokenized.append(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            tokenized.append([])
    return {'tokens': tokenized}

print("Processing dataset in memory-efficient chunks...")
all_tokenized_data = []
processed_count = 0

for batch_start in range(0, min(MAX_DATASET_SIZE, len(dataset)), TOKENIZATION_BATCH_SIZE):
    batch_end = min(batch_start + TOKENIZATION_BATCH_SIZE, min(MAX_DATASET_SIZE, len(dataset)))
    batch_data = dataset.select(range(batch_start, batch_end))
    tokenized_batch = batch_data.map(
        tokenize_stories_efficient,
        batched=True,
        batch_size=TOKENIZATION_BATCH_SIZE,
        remove_columns=batch_data.column_names,
        keep_in_memory=False,
        writer_batch_size=50
    )
    for item in tokenized_batch:
        if item['tokens']:
            all_tokenized_data.append(item['tokens'])
    processed_count += len(tokenized_batch)
    del tokenized_batch, batch_data
    print(f"Processed {processed_count}/{min(MAX_DATASET_SIZE, len(dataset))} stories", end='\r')
    if processed_count >= MAX_DATASET_SIZE:
        break

print(f"\nTokenization complete! Processed {len(all_tokenized_data)} stories")

# Cell 7: Create training batches
def create_batches_efficient(tokenized_data, batch_size, block_size):
    print("Creating training batches...")
    all_tokens = []
    for tokens in tokenized_data:
        all_tokens.extend(tokens)
    print(f"Total tokens: {len(all_tokens):,}")

    batches = []
    max_batches = 200  # cap to control training time
    step_size = batch_size * block_size

    for i in range(0, len(all_tokens) - block_size - 1, step_size):
        if len(batches) >= max_batches:
            break
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            start_idx = i + j * block_size
            end_idx = start_idx + block_size
            if end_idx + 1 < len(all_tokens):
                x = all_tokens[start_idx:end_idx]
                y = all_tokens[start_idx + 1:end_idx + 1]
                if len(x) == block_size and len(y) == block_size:
                    batch_x.append(x)
                    batch_y.append(y)
        if len(batch_x) == batch_size:
            batches.append((np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32)))
        if len(batches) % 10 == 0 and len(batches) > 0:
            print(f"Created {len(batches)} batches", end='\r')

    print(f"\nCreated {len(batches)} training batches")
    return batches

train_batches = create_batches_efficient(all_tokenized_data, BATCH_SIZE, BLOCK_SIZE)
print(f"Training batches created: {len(train_batches)}")
print(f"Batch size: {BATCH_SIZE}, Sequence length: {BLOCK_SIZE}")

# Free memory
del all_tokenized_data
import gc; gc.collect()
print("Data ready for training.")

# Cell 8: PyTorch TinyGPT model
class PyTorchTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out
        return x

class PyTorchTinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            PyTorchTransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T = x.shape
        device = x.device
        
        pos = torch.arange(T, device=device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

MODEL_CONFIG = {
    'vocab_size': VOCAB_SIZE,
    'n_embd': 384,
    'n_head': 6,
    'n_layer': 6,
    'block_size': BLOCK_SIZE,
    'dropout': DROPOUT
}
model = PyTorchTinyGPT(**MODEL_CONFIG).to(device)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params/1e6:.1f}M parameters on {device}")

# Cell 9: Optimizer and train_step
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def train_step(model, x, y):
    model.train()
    optimizer.zero_grad()
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    loss.backward()
    # Optional gradient clipping for stability on T4
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# Cell 10: Training loop (~30 minutes target on T4)
from tqdm import tqdm

try:
    TRAIN_STEPS = len(train_batches)
except NameError:
    raise NameError("train_batches is not defined. Please run data cells before training.")

PRINT_EVERY = 10

train_losses = []
training_start_time = time.time()
target_training_time = 30 * 60  # 30 minutes

print(f"Starting training for ~30 minutes...")
print(f"Total training steps (epochs * batches-per-epoch capped by time): {TRAIN_STEPS} per epoch")

step = 0
epoch = 0

while (time.time() - training_start_time) < target_training_time and epoch < 10:
    epoch_start_time = time.time()
    epoch_losses = []
    print(f"\n--- Epoch {epoch + 1} ---")
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_batches, desc=f"Epoch {epoch + 1}")):
        if (time.time() - training_start_time) >= target_training_time:
            print("\nReached 30-minute time limit!")
            break
        loss_value = train_step(model, x_batch, y_batch)
        train_losses.append(loss_value)
        epoch_losses.append(loss_value)
        step += 1

        if step % PRINT_EVERY == 0:
            elapsed_time = time.time() - training_start_time
            remaining_time = max(0.0, target_training_time - elapsed_time)
            print(f"Step {step}, Loss: {loss_value:.4f}, Elapsed: {elapsed_time/60:.1f}m, Remaining: {remaining_time/60:.1f}m")
    avg_epoch_loss = sum(epoch_losses)/len(epoch_losses) if epoch_losses else float('nan')
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} completed in {epoch_time/60:.1f} minutes, Avg Loss: {avg_epoch_loss:.4f}")
    epoch += 1
    if (time.time() - training_start_time) >= target_training_time:
        break

total_training_time = time.time() - training_start_time
print("\nTraining completed!")
print(f"Total training time: {total_training_time/60:.1f} minutes")
print(f"Total steps: {step}")
print(f"Final loss: {train_losses[-1]:.4f}" if train_losses else "No training losses recorded.")

# Cell 11: Plot training loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)

window_size = 50
if len(train_losses) > window_size:
    moving_avg = []
    for i in range(window_size, len(train_losses)):
        moving_avg.append(sum(train_losses[i-window_size:i]) / window_size)
    plt.subplot(1, 2, 2)
    plt.plot(range(window_size, len(train_losses)), moving_avg)
    plt.title(f'Training Loss (Moving Average, window={window_size})')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Training completed in {total_training_time/60:.1f} minutes")
if train_losses:
    print(f"Loss decreased from {train_losses[0]:.4f} to {train_losses[-1]:.4f}")

# Cell 12: Text generation (PyTorch-only)
import traceback

EOT_TOKEN = 50256  # GPT-2 end-of-text in tiktoken

def generate_story(model, prompt="Once upon a time", max_length=200, temperature=0.8):
    if not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
        raise ValueError("Tokenizer is not properly initialized")

    vocab_size = VOCAB_SIZE
    try:
        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            raise ValueError("Prompt produced no tokens")
        if any((not isinstance(t, int)) or (t < 0) or (t >= vocab_size) for t in prompt_tokens):
            raise ValueError(f"Invalid prompt tokens: {prompt_tokens}")

        generated = prompt_tokens.copy()
        model.eval()
        with torch.no_grad():
            for step in range(max_length):
                context = generated[-BLOCK_SIZE:] if len(generated) > BLOCK_SIZE else generated
                if not context:
                    break
                if any((not isinstance(t, int)) or (t < 0) or (t >= vocab_size) for t in context):
                    raise ValueError(f"Invalid context tokens: {context}")

                x = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(x)
                if logits.shape[-1] != vocab_size:
                    raise ValueError(f"Logits last dim {logits.shape[-1]} != vocab_size {vocab_size}")
                next_token_logits = logits[0, -1, :] / temperature
                if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                    raise ValueError("Logits contain NaN or Inf")
                probs = torch.softmax(next_token_logits, dim=-1)
                probs_np = probs.detach().float().cpu().numpy()
                probs_np = np.clip(probs_np, 1e-8, 1.0)
                probs_np = probs_np / probs_np.sum()
                next_token = int(np.random.choice(len(probs_np), p=probs_np))
                generated.append(next_token)
                if next_token == EOT_TOKEN:
                    break
                if len(generated) >= 2048:
                    break

        story = tokenizer.decode(generated)
        return story
    except Exception as e:
        return f"Error generating story: {e}\n{traceback.format_exc()}"

def test_generation():
    try:
        print("Testing story generation...")
        test_prompt = "Once upon a time"
        test_story = generate_story(model, prompt=test_prompt, max_length=50, temperature=0.8)
        if not isinstance(test_story, str) or len(test_story.strip()) <= len(test_prompt):
            raise ValueError("Generated story is empty or too short")
        print(f"Test successful! Generated: {test_story[:100]}...")
        return True
    except Exception as e:
        print(f"Generation test failed: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        return False

# Cell 13: Generate from prompts
if test_generation():
    print("Story generation function is working!")
else:
    print("Story generation function has issues. Check the model and tokenizer.")

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
    story = generate_story(model, prompt=prompt, max_length=150, temperature=0.8)
    if isinstance(story, str) and len(story.strip()) > len(prompt):
        story_content = story[len(prompt):].strip() if story.startswith(prompt) else story.strip()
        if story_content:
            print(f"{prompt} {story_content}")
        else:
            print("Generated story was too short or empty.")
    else:
        print("Generated story was empty or an error occurred.")
    print()
print("=" * 60)
print("Training and story generation completed!")

# Sample a subset for 30-minute training
DATASET_SIZE = 50000  # Adjust based on training time constraint
dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))

print(f"Dataset size: {len(dataset)}")
print("Sample story:")
# FIXED: Use 'text' instead of 'story' as the column name
print(dataset[0]['text'][:200] + "...")

# Initialize tokenizer (GPT-2 tokenizer)
tokenizer = tiktoken.encoding_for_model("gpt2")
vocab_size = tokenizer.n_vocab

print(f"Vocabulary size: {vocab_size}")


# Configuration - Reduced for memory efficiency
BLOCK_SIZE = 256          # Context length
BATCH_SIZE = 8            # Reduced batch size for T4 GPU memory
TOKENIZATION_BATCH_SIZE = 100  # Small batches for tokenization
MAX_DATASET_SIZE = 10000  # Limit dataset size for 30-min training

print(f"Using reduced dataset size: {MAX_DATASET_SIZE} stories")
print(f"Tokenization batch size: {TOKENIZATION_BATCH_SIZE}")

def tokenize_stories_efficient(examples):
    """Memory-efficient tokenization with smaller batches"""
    tokenized = []
    for story in examples['text']:
        try:
            tokens = tokenizer.encode(story, allowed_special={"<|endoftext|>"})
            if len(tokens) > BLOCK_SIZE:
                tokens = tokens[:BLOCK_SIZE]
            tokenized.append(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            tokenized.append([])
    return {'tokens': tokenized}

# Process dataset in small batches, avoid loading all in RAM
print("Processing dataset in memory-efficient chunks...")
all_tokenized_data = []
processed_count = 0

for batch_start in range(0, min(MAX_DATASET_SIZE, len(dataset)), TOKENIZATION_BATCH_SIZE):
    batch_end = min(batch_start + TOKENIZATION_BATCH_SIZE, min(MAX_DATASET_SIZE, len(dataset)))
    batch_data = dataset.select(range(batch_start, batch_end))
    tokenized_batch = batch_data.map(
        tokenize_stories_efficient,
        batched=True,
        batch_size=TOKENIZATION_BATCH_SIZE,
        remove_columns=batch_data.column_names,
        keep_in_memory=False,
        writer_batch_size=50
    )
    for item in tokenized_batch:
        if item['tokens']:
            all_tokenized_data.append(item['tokens'])
    processed_count += len(tokenized_batch)
    del tokenized_batch, batch_data
    print(f"Processed {processed_count}/{min(MAX_DATASET_SIZE, len(dataset))} stories", end='\r')
    if processed_count >= MAX_DATASET_SIZE:
        break

print(f"\nTokenization complete! Processed {len(all_tokenized_data)} stories")

# Create training batches function (complete)
def create_batches_efficient(tokenized_data, batch_size, block_size):
    print("Creating training batches...")
    all_tokens = []
    for tokens in tokenized_data:
        all_tokens.extend(tokens)
    print(f"Total tokens: {len(all_tokens):,}")

    batches = []
    max_batches = 150  # limit batches to keep training time reasonable

    # Step size here: batch_size * block_size (jump) to avoid overlap
    step_size = batch_size * block_size

    for i in range(0, len(all_tokens) - block_size - 1, step_size):
        if len(batches) >= max_batches:
            break
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            start_idx = i + j * block_size
            end_idx = start_idx + block_size
            if end_idx + 1 < len(all_tokens):
                x = all_tokens[start_idx:end_idx]
                y = all_tokens[start_idx + 1:end_idx + 1]
                if len(x) == block_size and len(y) == block_size:
                    batch_x.append(x)
                    batch_y.append(y)
        if len(batch_x) == batch_size:
            batches.append((np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32)))
        if len(batches) % 10 == 0 and len(batches) > 0:
            print(f"Created {len(batches)} batches", end='\r')

    print(f"\nCreated {len(batches)} training batches")
    return batches

# Create batches for training
train_batches = create_batches_efficient(all_tokenized_data, BATCH_SIZE, BLOCK_SIZE)

print(f"Training batches created: {len(train_batches)}")
print(f"Batch size: {BATCH_SIZE}, Sequence length: {BLOCK_SIZE}")

# Free memory after batch creation
del all_tokenized_data
import gc; gc.collect()

print("Cell 4 processing complete, ready for training.")


if USE_MLX:
    # MLX-based transformer model
    import mlx.nn as nn
    import mlx.core as mx
    
    class MLXTransformerBlock(nn.Module):
        def __init__(self, n_embd, n_head, dropout=0.1):
            super().__init__()
            self.n_embd = n_embd
            self.n_head = n_head
            
            self.attention = nn.MultiHeadAttention(n_embd, n_head, bias=True)
            self.feed_forward = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout)
            )
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
            self.dropout = nn.Dropout(dropout)
        
        def __call__(self, x):
            # Self-attention with residual connection
            attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
            x = x + self.dropout(attn_out)
            
            # Feed-forward with residual connection  
            ff_out = self.feed_forward(self.ln2(x))
            x = x + ff_out
            
            return x
    
    class MLXTinyGPT(nn.Module):
        def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_embd = n_embd
            self.block_size = block_size
            
            self.token_embedding = nn.Embedding(vocab_size, n_embd)
            self.position_embedding = nn.Embedding(block_size, n_embd)
            
            self.blocks = [MLXTransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            
            self.dropout = nn.Dropout(dropout)
        
        def __call__(self, x):
            B, T = x.shape
            
            # Token and position embeddings
            pos = mx.arange(T)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            
            x = self.dropout(tok_emb + pos_emb)
            
            # Apply transformer blocks
            for block in self.blocks:
                x = block(x)
            
            x = self.ln_f(x)
            logits = self.head(x)
            
            return logits

else:
    # PyTorch fallback implementation
    import torch.nn as nn
    import torch.nn.functional as F
    
    class PyTorchTransformerBlock(nn.Module):
        def __init__(self, n_embd, n_head, dropout=0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
            self.feed_forward = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout)
            )
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            x_norm = self.ln1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
            x = x + self.dropout(attn_out)
            
            ff_out = self.feed_forward(self.ln2(x))
            x = x + ff_out
            
            return x
    
    class PyTorchTinyGPT(nn.Module):
        def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_embd = n_embd
            self.block_size = block_size
            
            self.token_embedding = nn.Embedding(vocab_size, n_embd)
            self.position_embedding = nn.Embedding(block_size, n_embd)
            
            self.blocks = nn.ModuleList([
                PyTorchTransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
            ])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            B, T = x.shape
            device = x.device
            
            pos = torch.arange(T, device=device)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos)
            
            x = self.dropout(tok_emb + pos_emb)
            
            for block in self.blocks:
                x = block(x)
            
            x = self.ln_f(x)
            logits = self.head(x)
            
            return logits

# Model configuration for 30-minute training
MODEL_CONFIG = {
    'vocab_size': vocab_size,
    'n_embd': 384,      # Embedding dimension
    'n_head': 6,        # Number of attention heads  
    'n_layer': 6,       # Number of transformer layers
    'block_size': BLOCK_SIZE,
    'dropout': 0.1
}

print(f"Model configuration: {MODEL_CONFIG}")

# Calculate approximate parameter count
n_params = (
    vocab_size * MODEL_CONFIG['n_embd'] +  # Token embedding
    MODEL_CONFIG['block_size'] * MODEL_CONFIG['n_embd'] +  # Position embedding
    MODEL_CONFIG['n_layer'] * (
        4 * MODEL_CONFIG['n_embd'] ** 2 +  # Attention weights
        8 * MODEL_CONFIG['n_embd'] ** 2    # Feed-forward weights
    ) +
    vocab_size * MODEL_CONFIG['n_embd']    # Output head
)

print(f"Approximate model parameters: {n_params / 1e6:.1f}M")


# Initialize model
if USE_MLX:
    model = MLXTinyGPT(**MODEL_CONFIG)
    import mlx.optimizers as optim
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)

    def compute_loss(model, x, y):
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)))
        return loss

    def train_step(model, x, y):
        loss_fn = lambda m: compute_loss(m, x, y)
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    # Show parameter count for MLX model (using precomputed n_params)
    print(f"Model initialized with {n_params / 1e6:.1f}M parameters")
    print("Using device: MLX CUDA")

else:
    model = PyTorchTinyGPT(**MODEL_CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    def train_step(model, x, y):
        model.train()
        optimizer.zero_grad()        
        x = torch.tensor(x, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    # Show parameter count using PyTorch API
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"Using device: {device}")


import time
from tqdm import tqdm

# --- Check if train_batches exists ---
try:
    TRAIN_STEPS = len(train_batches)
except NameError:
    raise NameError("train_batches is not defined. Please run the data tokenization and batching cell before training.")

PRINT_EVERY = 10
EVAL_EVERY = 50

# Training metrics
train_losses = []
training_start_time = time.time()
target_training_time = 30 * 60  # 30 minutes in seconds

print(f"Starting training for ~30 minutes...")
print(f"Total training steps: {TRAIN_STEPS}")

step = 0
epoch = 0

while (time.time() - training_start_time) < target_training_time and epoch < 10:
    epoch_start_time = time.time()
    epoch_losses = []
    
    print(f"\n--- Epoch {epoch + 1} ---")
    
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_batches, desc=f"Epoch {epoch + 1}")):
        # Check time limit
        if (time.time() - training_start_time) >= target_training_time:
            print(f"\nReached 30-minute time limit!")
            break
            
        if USE_MLX:
            x_mx = mx.array(x_batch)
            y_mx = mx.array(y_batch)
            loss = train_step(model, x_mx, y_mx)
            loss_value = float(loss)
        else:
            loss_value = train_step(model, x_batch, y_batch)
        
        train_losses.append(loss_value)
        epoch_losses.append(loss_value)
        step += 1
        
        # Print progress
        if step % PRINT_EVERY == 0:
            elapsed_time = time.time() - training_start_time
            remaining_time = target_training_time - elapsed_time
            print(f"Step {step}/{TRAIN_STEPS}, Loss: {loss_value:.4f}, "
                  f"Elapsed: {elapsed_time/60:.1f}min, Remaining: {remaining_time/60:.1f}min")
    
    # Epoch summary
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} completed in {epoch_time/60:.1f} minutes, Average Loss: {avg_epoch_loss:.4f}")
    
    epoch += 1
    
    # Break if time limit reached
    if (time.time() - training_start_time) >= target_training_time:
        break

total_training_time = time.time() - training_start_time
print(f"\nTraining completed!")
print(f"Total training time: {total_training_time/60:.1f} minutes")
print(f"Total steps: {step}")
print(f"Final loss: {train_losses[-1]:.4f}" if train_losses else "No training losses recorded.")


# Plot training loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)

# Moving average
window_size = 50
if len(train_losses) > window_size:
    moving_avg = []
    for i in range(window_size, len(train_losses)):
        moving_avg.append(sum(train_losses[i-window_size:i]) / window_size)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(window_size, len(train_losses)), moving_avg)
    plt.title(f'Training Loss (Moving Average, window={window_size})')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Training completed in {total_training_time/60:.1f} minutes")
print(f"Loss decreased from {train_losses[0]:.4f} to {train_losses[-1]:.4f}")

## THIS IS THE LAST CELL 

import numpy as np
import traceback

# Use the previously determined backend and objects
if 'model' not in globals():
    raise RuntimeError("Model is not initialized. Run training cells first.")
if 'tokenizer' not in globals():
    raise RuntimeError("Tokenizer is not initialized. Run setup cells first.")
if 'USE_MLX' not in globals():
    print("Warning: USE_MLX not defined; defaulting to PyTorch path.")
    USE_MLX = False

# Resolve BLOCK_SIZE
if 'BLOCK_SIZE' not in globals():
    BLOCK_SIZE = 256

# Resolve vocab and EOT token from tokenizer if possible
try:
    vocab_size = tokenizer.n_vocab
except Exception:
    vocab_size = 50257  # GPT-2 default
eot_token = 50256  # GPT-2 end-of-text in tiktoken

# Backend availability checks for MLX, otherwise fall back to PyTorch
MLX_AVAILABLE = False
if USE_MLX:
    try:
        import mlx.core as mx
        MLX_AVAILABLE = True
    except Exception:
        MLX_AVAILABLE = False
        print("MLX requested but not available; falling back to PyTorch.")
        USE_MLX = False

# Prepare PyTorch device if needed
if not USE_MLX:
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device if hasattr(model, 'parameters') else (
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

def generate_story(model, prompt="Once upon a time", max_length=200, temperature=0.8):
    """Generate a story from the trained model using the active backend"""
    if not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
        raise ValueError("Tokenizer is not properly initialized")

    try:
        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            raise ValueError("Prompt produced no tokens")
        if any((not isinstance(t, int)) or (t < 0) or (t >= vocab_size) for t in prompt_tokens):
            raise ValueError(f"Invalid prompt tokens: {prompt_tokens}")

        generated = prompt_tokens.copy()

        if USE_MLX:
            # MLX inference loop (no model.eval in MLX)
            for step in range(max_length):
                context = generated[-BLOCK_SIZE:] if len(generated) > BLOCK_SIZE else generated
                if not context:
                    break
                if any((not isinstance(t, int)) or (t < 0) or (t >= vocab_size) for t in context):
                    raise ValueError(f"Invalid context tokens: {context}")

                x = mx.array([context], dtype=mx.int32)
                try:
                    logits = model(x)
                    if logits.shape[-1] != vocab_size:
                        raise ValueError(f"Logits last dim {logits.shape[-1]} != vocab_size {vocab_size}")
                    next_token_logits = logits[0, -1, :] / temperature
                    if mx.any(mx.isnan(next_token_logits)) or mx.any(mx.isinf(next_token_logits)):
                        raise ValueError("Logits contain NaN or Inf")
                    probs = mx.softmax(next_token_logits, axis=-1)
                    probs_np = np.asarray(probs, dtype=np.float64)
                    probs_np = np.clip(probs_np, 1e-8, 1.0)
                    probs_np = probs_np / probs_np.sum()
                    next_token = int(np.random.choice(len(probs_np), p=probs_np))
                    generated.append(next_token)

                    # Light synchronization and cleanup
                    del x, logits, next_token_logits, probs
                    mx.eval(mx.array([]))

                    if next_token == eot_token:
                        break
                    if len(generated) >= 2048:
                        break
                except Exception as e:
                    return f"Error during MLX inference at step {step}: {e}"
        else:
            # PyTorch inference loop
            model.eval()
            with torch.no_grad():
                for step in range(max_length):
                    context = generated[-BLOCK_SIZE:] if len(generated) > BLOCK_SIZE else generated
                    if not context:
                        break
                    if any((not isinstance(t, int)) or (t < 0) or (t >= vocab_size) for t in context):
                        raise ValueError(f"Invalid context tokens: {context}")

                    x = torch.tensor([context], dtype=torch.long, device=device)
                    logits = model(x)
                    if logits.shape[-1] != vocab_size:
                        raise ValueError(f"Logits last dim {logits.shape[-1]} != vocab_size {vocab_size}")
                    next_token_logits = logits[0, -1, :] / temperature
                    if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                        raise ValueError("Logits contain NaN or Inf")
                    probs = torch.softmax(next_token_logits, dim=-1)
                    probs_np = probs.detach().cpu().numpy().astype(np.float64)
                    probs_np = np.clip(probs_np, 1e-8, 1.0)
                    probs_np = probs_np / probs_np.sum()
                    next_token = int(np.random.choice(len(probs_np), p=probs_np))
                    generated.append(next_token)
                    if next_token == eot_token:
                        break
                    if len(generated) >= 2048:
                        break

        story = tokenizer.decode(generated)
        return story

    except Exception as e:
        return f"Error generating story: {e}\n{traceback.format_exc()}"

def test_generation():
    """Test story generation with a simple prompt"""
    try:
        print("Testing story generation...")
        test_prompt = "Once upon a time"
        test_story = generate_story(model, prompt=test_prompt, max_length=50, temperature=0.8)
        if not isinstance(test_story, str) or len(test_story.strip()) <= len(test_prompt):
            raise ValueError("Generated story is empty or too short")
        print(f"Test successful! Generated: {test_story[:100]}...")
        return True
    except Exception as e:
        print(f"Generation test failed: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        return False

# Run the test and generate stories
if test_generation():
    print("Story generation function is working!")
else:
    print("Story generation function has issues. Check the model and tokenizer.")

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
    try:
        story = generate_story(model, prompt=prompt, max_length=150, temperature=0.8)
        if isinstance(story, str) and len(story.strip()) > len(prompt):
            story_content = story[len(prompt):].strip() if story.startswith(prompt) else story.strip()
            if story_content:
                print(f"{prompt} {story_content}")
            else:
                print("Generated story was too short or empty.")
        else:
            print("Generated story was empty or an error occurred.")
    except Exception as e:
        print(f"Error generating story: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
    print()

print("=" * 60)
print("Training and story generation completed!")
