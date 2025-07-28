# train.py
import torch
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from utils import get_dataset, get_tokenizer, TinyStoriesDataset, get_device
import config

def main():
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset and tokenizer
    dataset = get_dataset(config.DATASET_NAME, config.DATASET_CONFIG_NAME)
    tokenizer = get_tokenizer(config.MODEL_NAME)

    # Use preprocessed datasets directly
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    # Load model
    model_config = GPT2Config(**config.MODEL_CONFIG)
    model = GPT2LMHeadModel(config=model_config)
    model.to(device)
    model.config.loss_type = "ForCausalLMLoss"  # Explicitly set loss type

    # Define training arguments
    training_args = TrainingArguments(
        **config.TRAINING_ARGS,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./results/TinyLLM")
    tokenizer.save_pretrained("./results/TinyLLM")

if __name__ == "__main__":
    main()
