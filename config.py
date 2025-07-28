# config.py

# Data settings
DATASET_NAME = "roneneldan/TinyStoriesInstruct"
DATASET_CONFIG_NAME = "default"
TEXT_COLUMN = "text"

# Model settings
MODEL_NAME = "TinyLLM"
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": None,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
    "initializer_range": 0.02,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "summary_activation": None,
    "summary_proj_to_labels": True,
    "summary_first_dropout": 0.1,
    "scale_attn_weights": True,
    "use_cache": True,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}

# Training settings
TRAINING_ARGS = {
    "output_dir": "./results/TinyLLM",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 500,
    "load_best_model_at_end": True,
    "report_to": "tensorboard",
    "push_to_hub": False,
    "dataloader_pin_memory": False,
}

# Inference settings
MAX_LENGTH = 50
NUM_RETURN_SEQUENCES = 1
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2
