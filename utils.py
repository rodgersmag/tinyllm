# utils.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import config

def get_dataset(dataset_name, config_name):
    """Downloads and returns the specified dataset from the Hugging Face Hub."""
    # Check for preprocessed dataset first
    import os
    if os.path.exists("./data/preprocessed_dataset.pt"):
        print("Loading preprocessed dataset...")
        # Use safe_globals context to allow dataset class during unpickling
        with torch.serialization.safe_globals([TinyStoriesDataset]):
            return torch.load("./data/preprocessed_dataset.pt")
    
    print("Processing dataset for the first time...")
    dataset = load_dataset(dataset_name, config_name, cache_dir="./data")
    
    # Tokenize and preprocess dataset
    tokenizer = get_tokenizer(config.MODEL_NAME)
    tokenized_dataset = {}
    for split in dataset.keys():
        tokenized_dataset[split] = TinyStoriesDataset(
            [item[config.TEXT_COLUMN] for item in dataset[split]],
            tokenizer
        )
    
    # Save preprocessed dataset
    os.makedirs("./data", exist_ok=True)
    torch.save(tokenized_dataset, "./data/preprocessed_dataset.pt")
    print("Dataset preprocessed and saved.")
    return tokenized_dataset

def get_tokenizer(model_name):
    """Initializes and returns the tokenizer for the specified model."""
    # Use the gpt2 tokenizer for our custom "TinyLLM"
    tokenizer_name = "gpt2" if model_name == "TinyLLM" else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class TinyStoriesDataset(Dataset):
    """PyTorch Dataset for the TinyStoriesInstruct dataset."""
    def __init__(self, stories, tokenizer, max_length=512):
        self.stories = stories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        inputs = self.tokenizer(
            story,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Squeeze to remove the batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # The labels are the same as the input_ids
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

def create_data_loader(dataset, batch_size, shuffle=True):
    """Creates a DataLoader for the given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_device():
    """Returns the appropriate device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

if __name__ == '__main__':
    # Example usage
    dataset = get_dataset(config.DATASET_NAME, config.DATASET_CONFIG_NAME)
    tokenizer = get_tokenizer(config.MODEL_NAME)
    
    train_stories = [item[config.TEXT_COLUMN] for item in dataset['train']]
    validation_stories = [item[config.TEXT_COLUMN] for item in dataset['validation']]

    train_dataset = TinyStoriesDataset(train_stories, tokenizer)
    validation_dataset = TinyStoriesDataset(validation_stories, tokenizer)

    train_loader = create_data_loader(train_dataset, batch_size=config.TRAINING_ARGS['per_device_train_batch_size'])
    validation_loader = create_data_loader(validation_dataset, batch_size=config.TRAINING_ARGS['per_device_eval_batch_size'], shuffle=False)

    print(f"Device: {get_device()}")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(validation_dataset)}")

    # Print a sample batch
    for batch in train_loader:
        print("Sample batch:")
        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        break
