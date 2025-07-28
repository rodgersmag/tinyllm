import mlx.core as mx
import mlx.nn as nn
from utils import GPT, CharTokenizer
from config import GPTConfig
import numpy as np
import os

def generate(model, tokenizer, prompt, max_len=100, temperature=1.0, config=None):
    """Generate text using the trained model"""
    if config is None:
        raise ValueError("Config must be provided")
    
    model.eval()
    tokens = tokenizer.encode(prompt)
    
    print(f"Starting generation with prompt: '{prompt}'")
    print(f"Initial tokens: {tokens}")
    
    for i in range(max_len):
        # Use the last ctx_len tokens as context
        context_tokens = tokens[-config.ctx_len:] if len(tokens) > config.ctx_len else tokens
        
        # Pad if necessary
        if len(context_tokens) < config.ctx_len:
            context_tokens = [0] * (config.ctx_len - len(context_tokens)) + context_tokens
        
        input_tokens = mx.array(context_tokens).reshape(1, -1)
        
        try:
            # Forward pass
            with mx.no_grad():
                logits = model(input_tokens)
                logits = logits[0, -1, :] / temperature  # Get last token logits and apply temperature
                
                # Convert to probabilities
                probs = nn.softmax(logits, axis=-1)
                probs_np = probs.numpy()
                
                # Sample next token
                next_token = np.random.choice(len(tokenizer.chars), p=probs_np)
                tokens.append(next_token)
                
                # Print progress occasionally
                if i % 10 == 0:
                    current_text = tokenizer.decode(tokens)
                    print(f"Step {i}: Generated so far: '{current_text[-50:]}'")
                
        except Exception as e:
            print(f"Error during generation at step {i}: {e}")
            break
    
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    config = GPTConfig()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(script_dir, "train.txt")
    model_path = os.path.join(script_dir, "model.npz")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        exit(1)
    
    # Load data to initialize tokenizer
    if not os.path.exists(train_file_path):
        print(f"Error: Training data '{train_file_path}' not found. Please run train.py first.")
        exit(1)
    
    try:
        with open(train_file_path, "r", encoding="utf-8") as f:
            data = f.read()
        print(f"Loaded training data: {len(data)} characters")
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit(1)
    
    # Initialize tokenizer
    tokenizer = CharTokenizer(data)
    config.vocab_size = len(tokenizer.chars)
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Initialize and load model
    try:
        model = GPT(config)
        model.load_weights(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Generate text with different prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest",
        "The magic"
    ]
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Generating with prompt: '{prompt}'")
        print('='*50)
        
        try:
            output = generate(model, tokenizer, prompt, max_len=100, temperature=0.8, config=config)
            print(f"\nFull generated text:\n{output}")
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")
            continue