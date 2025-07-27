# TinyLLM Training on MacBook Air M2

This project provides a comprehensive guide for training a small Large Language Model (LLM) on a MacBook Air M2 using the TinyStoriesInstruct dataset. The focus is on optimizing the training process for the M2's unified memory architecture and leveraging its GPU acceleration via Metal Performance Shaders (MPS).

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware Considerations](#hardware-considerations)
- [Environment Setup with `uv`](#environment-setup-with-uv)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Performance Monitoring](#performance-monitoring)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Project Overview

The goal of this project is to demonstrate how to effectively train a small LLM on a resource-constrained device like the MacBook Air M2. We will use the TinyStoriesInstruct dataset, which is well-suited for this purpose due to its small size and high-quality instructional data.

## Hardware Considerations

The MacBook Air M2 presents unique challenges and opportunities for LLM training:

- **Unified Memory:** The 8GB or 16GB of unified memory is shared between the CPU and GPU. This requires careful memory management to avoid swapping and performance degradation.
- **M2 GPU:** The M2's GPU can be leveraged for significant acceleration using PyTorch's Metal Performance Shaders (MPS) backend.
- **Thermal Management:** The fanless design of the MacBook Air requires monitoring core temperatures during long training runs to prevent thermal throttling.
- **Storage:** Ensure you have sufficient storage for the dataset, model checkpoints, and Python environment.

## Environment Setup with `uv`

We will use `uv`, a fast Python package installer and resolver, to manage our environment.

1.  **Install `uv`:** If you don't have it, install `uv`.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Virtual Environment:**
    ```bash
    uv venv
    ```
    This will create a `.venv` directory in your project folder.

3.  **Activate the Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Data Preprocessing

The data loading and preprocessing pipeline will be handled by a `utils.py` script. This will involve:

-   Downloading the `TinyStoriesInstruct` dataset from the Hugging Face Hub.
-   Tokenizing the text data using the `gpt2` tokenizer.
-   Creating a PyTorch `Dataset` and `DataLoader` for efficient batching.

## Model Architecture

We are training our own model, which we are calling `TinyLLM`. For the M2's memory constraints, we are using a GPT-2 small configuration as a starting point. The model is defined in the `train.py` script and configured in `config.py`.

## Training

The main training logic will be in `train.py`. To start training, you can run the script directly after activating your environment, or use `uv` to run it without activation.

**To start training:**
```bash
uv run python train.py
```
This command will automatically use the python interpreter from your `.venv`.

## Inference

The `inference.py` script will load the trained `TinyLLM` model and allow you to generate text.

**To run inference:**
```bash
uv run python inference.py --prompt "Once upon a time"
```

## Performance Monitoring

During training, it's crucial to monitor:

-   **Memory Usage:** Use `htop` or Activity Monitor to track memory pressure.
-   **GPU Utilization:** Monitor GPU usage in the Activity Monitor.
-   **Temperature:** Use `istats` or a similar tool to monitor CPU/GPU temperature.

## Troubleshooting

-   **`RuntimeError: MPS backend out of memory`:** Reduce the batch size in `config.py`, enable gradient accumulation, or use a smaller model configuration.
-   **Slow Training:** Ensure the `mps` device is being used. Profile the code to identify bottlenecks.

## Next Steps

-   **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and other hyperparameters in `config.py`.
-   **Model Scaling:** Try larger models if you have a 16GB M2 or access to more powerful hardware.
-   **Quantization:** Use techniques like `bitsandbytes` for 8-bit or 4-bit quantization to further reduce the memory footprint.