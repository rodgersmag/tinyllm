import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from utils import GPT, CharTokenizer
from config import GPTConfig

def get_batch(data, batch_size, ctx_len):
    idx = mx.random.randint(0, len(data) - ctx_len - 1, (batch_size,))
    x = mx.stack([data[i:i+ctx_len] for i in idx])
    y = mx.stack([data[i+1:i+ctx_len+1] for i in idx])
    return x, y

if __name__ == "__main__":
    config = GPTConfig()
    
    # Load and preprocess data
    with open("train.txt", "r") as f:
        data = f.read()
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
            loss, grads = nn.value_and_grad(model, lambda m: nn.losses.cross_entropy(m(x), y))(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss / num_batches:.4f}")
    
    # Save model
    mx.savez("model.npz", **model.parameters())