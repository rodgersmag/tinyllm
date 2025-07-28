import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from utils import GPT, CharTokenizer
from config import GPTConfig
from datasets import load_dataset
import os

def get_batch(data, batch_size, ctx_len):
    idx = mx.random.randint(0, len(data) - ctx_len - 1, (batch_size,)).tolist()  # Convert to Python list
    x = mx.stack([data[i:i+ctx_len] for i in idx])
    y = mx.stack([data[i+1:i+ctx_len+1] for i in idx])
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
    except FileNotFoundError:
        print(f"Error: {train_file_path} not found. Dataset download may have failed.")
        exit(1)
    
    tokenizer = CharTokenizer(data)
    config.vocab_size = len(tokenizer.chars)
    data = mx.array(tokenizer.encode(data))
    
    # Initialize model and optimizer
    model = GPT(config)
    optimizer = optim.Adam(learning_rate=config.lr)
    
    # Training loop
    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = (len(data) - config.ctx_len) // config.batch_size
        for i in range(0, len(data) - config.ctx_len, config.batch_size):
            x, y = get_batch(data, config.batch_size, config.ctx_len)
            loss, grads = nn.value_and_grad(model, lambda m: nn.losses.cross_entropy(m.forward(x), y, reduction='mean'))(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss / num_batches:.4f}")
    
    # Save model
    mx.savez("model.npz", **model.parameters())