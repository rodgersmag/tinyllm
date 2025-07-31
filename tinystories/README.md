# TinyStories Research Project

This project is a research initiative focused on training small-scale Language Learning Models (LLMs). The primary workflow involves leveraging the power of a Google Colab T4 GPU for initial training and then transferring this knowledge to a local environment for testing and inference on an Apple M-series device.

## Workflow

The project follows a two-stage process:

1.  **Cloud-Based Training (GPU):** The initial model training is performed in a Google Colab environment, which provides free access to T4 GPUs. The script `main.py` is a self-contained notebook that handles environment setup, data processing, and model training.

2.  **Local Inference & Development (MPS):** After training the model in the cloud, the saved weights are brought into a local development environment. This allows for faster iteration, testing, and application development on an MPS-compatible device (like an Apple Silicon Mac). The local workflow is managed by a set of dedicated scripts:
    *   `train.py`: Refactored training script for local execution.
    *   `inference.py`: A command-line tool for generating text from the trained model.
    *   `api.py`: A FastAPI server to expose the model's generation capabilities via a REST API.

## How to Replicate

This project uses `uv` for fast and efficient project and virtual environment management.

### 1. Setup the Environment

First, create a virtual environment using `uv`.

```bash
uv venv
```

This will create a `.venv` directory in the current folder.

### 2. Install Dependencies

Activate the environment and install the required packages from `requirements.txt`.

```bash
# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate

# Install packages
uv pip install -r requirements.txt
```

### 3. Running the Scripts

You can run the Python scripts using `uv run`.

#### Training the Model

To start the training process locally (e.g., on an MPS device), run the `train.py` script. This will download the dataset, train the model, and save the final weights to `model.pth`.

```bash
uv run train.py
```

#### Generating Text via CLI

Use the `inference.py` script to generate text from the command line. You can provide one or more prompts. The output will be printed to the console and saved to `output.txt`.

```bash
# Use the default prompts
uv run inference.py

# Provide custom prompts
uv run inference.py --prompt "A lonely robot" "The secret of the ocean"
```

#### Running the API Server

To serve the model via a web API, run the `api.py` script.

```bash
uv run api.py
```

The API will be available at `http://localhost:8000`. You can interact with it using tools like `curl` or by visiting the interactive documentation at `http://localhost:8000/docs`.

**Example `curl` command:**

```bash
curl -X POST "http://localhost:8000/generate" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "In a land of talking animals,"
}'
```
