import mlx.core as mx
import mlx.nn as nn
from utils import GPT, CharTokenizer
from config import GPTConfig
import numpy as np

def generate(model, tokenizer, prompt, max_len=100, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(prompt)
    for _ in range(max_len):
        input_tokens = mx.array(tokens[-config.ctx_len:]).reshape(1, -1)
        logits = model(input_tokens)
        logits = logits[:, -1, :] / temperature
        probs = nn.softmax(logits, axis=-1)
        next_token = np.random.choice(len(tokenizer.chars), p=probs[0].numpy())
        tokens.append(next_token)
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    config = GPTConfig()
    
    # Load data to initialize tokenizer
    with open("train.txt", "r") as f:
        data = f.read()
    tokenizer = CharTokenizer(data)
    config.vocab_size = len(tokenizer.chars)
    
    # Load model
    model = GPT(config)
    model.load_weights("model.npz")
    
    # Generate text
    prompt = "Once upon a time"
    output = generate(model, tokenizer, prompt, max_len=100)
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}")