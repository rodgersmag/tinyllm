import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from utils import GPT, CharTokenizer
from config import GPTConfig
from datasets import load_dataset
import os

def get_batch(data, batch_size, ctx_len):
    # Ensure we don't go out of bounds
    max_start = len(data) - ctx_len - 1
    if max_start <= 0:
        raise ValueError(f"Data too short. Need at least {ctx_len + 1} tokens, got {len(data)}")
    
    # Generate random starting indices and convert to Python integers
    idx = mx.random.randint(0, max_start, (batch_size,))
    idx_list = idx.tolist()  # Convert MLX array to Python list
    
    # Create batches using Python integer indices
    x = mx.stack([data[i:i+ctx_len] for i in idx_list])
    y = mx.stack([data[i+1:i+ctx_len+1] for i in idx_list])
    return x, y

if __name__ == "__main__":
    config = GPTConfig()
    
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(script_dir, "train.txt")
    
    # Check for train.txt and download TinyStories if missing
    if not os.path.exists(train_file_path):
        print("Downloading TinyStories dataset (10,000 stories)...")
        try:
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            dataset = dataset.select(range(10000))  # Subsample to fit 8GB RAM
            with open(train_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(dataset["text"]))
            print(f"Dataset saved as {train_file_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure 'datasets' is installed and you have an internet connection.")
            exit(1)
    
    # Load and preprocess data
    try:
        with open(train_file_path, "r", encoding="utf-8") as f:
            data = f.read()
        print(f"Loaded {len(data)} characters from dataset")
    except FileNotFoundError:
        print(f"Error: {train_file_path} not found. Dataset download may have failed.")
        exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)
    
    # Initialize tokenizer and check data size
    tokenizer = CharTokenizer(data)
    config.vocab_size = len(tokenizer.chars)
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Unique characters: {''.join(tokenizer.chars[:50])}..." if len(tokenizer.chars) > 50 else f"Unique characters: {''.join(tokenizer.chars)}")
    
    # Encode data
    try:
        encoded_data = tokenizer.encode(data)
        data = mx.array(encoded_data)
        print(f"Encoded data length: {len(data)} tokens")
    except Exception as e:
        print(f"Error encoding data: {e}")
        exit(1)
    
    # Check if we have enough data for training
    min_data_len = config.ctx_len + 1
    if len(data) < min_data_len:
        print(f"Error: Data too short for training. Need at least {min_data_len} tokens, got {len(data)}")
        exit(1)
    
    # Initialize model and optimizer
    print("Initializing model...")
    try:
        model = GPT(config)
        optimizer = optim.Adam(learning_rate=config.lr)
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)
    
    # Calculate number of batches
    num_batches = max(1, (len(data) - config.ctx_len) // config.batch_size)
    print(f"Starting training with {num_batches} batches per epoch")
    
    # Training loop
    try:
        for epoch in range(config.num_epochs):
            total_loss = 0
            print(f"Starting epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx in range(num_batches):
                try:
                    # Get batch
                    x, y = get_batch(data, config.batch_size, config.ctx_len)
                    
                    # Forward pass and compute loss
                    def loss_fn():
                        logits = model(x)
                        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                                                     y.reshape(-1), 
                                                     reduction='mean')
                    
                    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
                    loss, grads = loss_and_grad_fn()
                    
                    # Update model
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    
                    total_loss += loss.item()
                    
                    # Print progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        progress_pct = ((batch_idx + 1) / num_batches) * 100
                        print(f"  Batch {batch_idx+1}/{num_batches} ({progress_pct:.1f}%), Avg Loss: {avg_loss:.4f}")
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{config.num_epochs} completed, Average Loss: {avg_loss:.4f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save model
    try:
        print("Saving model...")
        model_path = os.path.join(script_dir, "model.npz")
        mx.savez(model_path, **model.parameters())
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")