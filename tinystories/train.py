import os
import time
import gc
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import tiktoken
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
CONFIG = {
    'BLOCK_SIZE': 256,
    'BATCH_SIZE': 8,
    'VOCAB_SIZE': 50257,  # Default for gpt2, will be updated by tokenizer
    'MAX_DATASET_SIZE': 15000,
    'LEARNING_RATE': 3e-4,
    'WEIGHT_DECAY': 0.1,
    'DROPOUT': 0.1,
    'N_EMBD': 384,
    'N_HEAD': 6,
    'N_LAYER': 6,
    'TRAIN_MINUTES': 30,
    'MODEL_SAVE_PATH': 'model.pth',
    'LOSS_PLOT_PATH': 'training_loss.png',
    'LOSS_MA_PLOT_PATH': 'training_loss_ma.png',
}

# --- Dataset and Tokenization ---

def load_tinystories_dataset(max_size: int) -> Dataset:
    """Load and prepare TinyStories dataset"""
    print("Loading TinyStories dataset...")
    try:
        dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=False)
        dataset_size = min(max_size, len(dataset))
        dataset = dataset.select(range(dataset_size))
        print(f"Dataset loaded: {len(dataset)} stories")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def tokenize_batch(texts: List[str], tokenizer, block_size: int) -> List[List[int]]:
    """Tokenize a batch of texts"""
    tokenized = []
    for text in texts:
        try:
            tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            if len(tokens) > 10:
                tokenized.append(tokens)
        except Exception as e:
            print(f"Tokenization error: {e}")
            continue
    return tokenized

def process_dataset_efficiently(dataset: Dataset, tokenizer, config: dict) -> List[List[int]]:
    """Process dataset with memory efficiency"""
    print("Processing dataset efficiently...")
    all_tokenized = []
    batch_size = 500
    total_stories = min(config['MAX_DATASET_SIZE'], len(dataset))
    
    for i in tqdm(range(0, total_stories, batch_size), desc="Tokenizing"):
        end_idx = min(i + batch_size, total_stories)
        batch_texts = [dataset[j]['text'] for j in range(i, end_idx)]
        tokenized_batch = tokenize_batch(batch_texts, tokenizer, config['BLOCK_SIZE'])
        all_tokenized.extend(tokenized_batch)
        if i % (batch_size * 4) == 0:
            gc.collect()
    
    print(f"Tokenization complete! Processed {len(all_tokenized)} valid stories")
    return all_tokenized

def create_training_data(tokenized_stories: List[List[int]], tokenizer, batch_size: int, block_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create training batches from tokenized stories"""
    print("Creating training batches...")
    all_tokens = []
    for story in tokenized_stories:
        all_tokens.extend(story)
        all_tokens.append(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0])
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    batches = []
    max_batches = 300
    
    for i in range(0, len(all_tokens) - block_size - 1, block_size):
        if len(batches) >= max_batches:
            break
        batch_x, batch_y = [], []
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

# --- Model Definition ---

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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

# --- Training ---

def train_step(model: nn.Module, optimizer, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    model.train()
    optimizer.zero_grad()
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, model.config['VOCAB_SIZE']), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def train_model(model: nn.Module, optimizer, train_batches: List, config: dict, device: torch.device):
    print(f"Starting training for ~{config['TRAIN_MINUTES']} minutes...")
    train_losses = []
    start_time = time.time()
    target_time = config['TRAIN_MINUTES'] * 60
    step, epoch = 0, 0

    while (time.time() - start_time) < target_time and epoch < 10:
        epoch_start, epoch_losses = time.time(), []
        print(f"\n--- Epoch {epoch + 1} ---")
        for batch in tqdm(train_batches, desc=f"Epoch {epoch + 1}"):
            if (time.time() - start_time) >= target_time:
                print("\nReached time limit!")
                break
            loss = train_step(model, optimizer, batch, device)
            train_losses.append(loss)
            epoch_losses.append(loss)
            step += 1
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1} completed in {(time.time() - epoch_start)/60:.1f}m, Avg Loss: {avg_loss:.4f}")
        epoch += 1

    print(f"\nTraining completed in {(time.time() - start_time)/60:.1f} minutes")
    return train_losses

def plot_and_save_loss(losses: List[float], path: str, title: str, moving_avg: bool = False):
    if not losses:
        print(f"No losses to plot for {title}")
        return
    plt.figure(figsize=(10, 5))
    if moving_avg:
        window = 50
        if len(losses) > window:
            avg = [sum(losses[i-window:i])/window for i in range(window, len(losses))]
            plt.plot(range(window, len(losses)), avg)
    else:
        plt.plot(losses)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f"Saved {title} plot to {path}")

# --- Main Execution ---

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = tiktoken.encoding_for_model("gpt2")
    CONFIG['VOCAB_SIZE'] = tokenizer.n_vocab
    print(f"Tokenizer loaded. Vocab size: {CONFIG['VOCAB_SIZE']}")

    # Data pipeline
    dataset = load_tinystories_dataset(CONFIG['MAX_DATASET_SIZE'])
    tokenized_data = process_dataset_efficiently(dataset, tokenizer, CONFIG)
    train_batches = create_training_data(tokenized_data, tokenizer, CONFIG['BATCH_SIZE'], CONFIG['BLOCK_SIZE'])
    del dataset, tokenized_data
    gc.collect()

    # Model and optimizer
    model = TinyGPT(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters.")

    # Training
    train_losses = train_model(model, optimizer, train_batches, CONFIG, device)

    # Save artifacts
    torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
    print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")
    
    plot_and_save_loss(train_losses, CONFIG['LOSS_PLOT_PATH'], 'Training Loss')
    plot_and_save_loss(train_losses, CONFIG['LOSS_MA_PLOT_PATH'], 'Training Loss (Moving Average)', moving_avg=True)

if __name__ == "__main__":
    main()
