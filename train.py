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

    train_stories = [item[config.TEXT_COLUMN] for item in dataset['train']]
    validation_stories = [item[config.TEXT_COLUMN] for item in dataset['validation']]

    train_dataset = TinyStoriesDataset(train_stories, tokenizer)
    validation_dataset = TinyStoriesDataset(validation_stories, tokenizer)

    # Load model
    model_config = GPT2Config(**config.MODEL_CONFIG)
    model = GPT2LMHeadModel(config=model_config)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        **config.TRAINING_ARGS,
        # Use 'mps' for Apple Silicon GPUs
        use_mps_device=torch.backends.mps.is_available()
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
