# inference.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import config
from utils import get_device

def generate_text(prompt, model_path="./results/TinyLLM"):
    """
    Generates text from a given prompt using a trained model.
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)

    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    outputs = model.generate(
        inputs,
        max_length=config.MAX_LENGTH,
        num_return_sequences=config.NUM_RETURN_SEQUENCES,
        do_sample=True,
        temperature=config.TEMPERATURE,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Text:\n")
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to start generation from.")
    parser.add_argument("--model_path", type=str, default="./results/TinyLLM", help="Path to the trained model.")
    args = parser.parse_args()

    generate_text(args.prompt, args.model_path)
