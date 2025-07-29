# TinyLLM

## MLX-GPT2 Implementation

This project extends the original [mlx-gpt2](https://github.com/pranavjad/mlx-gpt2) implementation with additional functionality for model persistence and inference.

### Enhancements Made

1. **Model Saving in `train.py`**
   - Added model serialization using `mx.savez()` to save trained weights
   - Fixed the `tree_flatten` import issue (changed from `mx.tree_flatten` to `utils.tree_flatten`)
   - Model is saved as `model.npz` after training completes

2. **Inference Script (`inference.py`)**
   - Created a standalone inference script for testing trained models
   - Supports command-line arguments for:
     - Custom prompts
     - Token generation length
     - Temperature control
     - Model path specification
   - Fixed context length mismatch issue (training uses ctx_len=8)

3. **FastAPI REST API (`api.py`)**
   - Built a REST API endpoint for model inference
   - Single `/generate` endpoint accepting:
     - `prompt`: Starting text (optional)
     - `max_tokens`: Number of tokens to generate (1-1000)
     - `temperature`: Sampling temperature (0.1-2.0)
   - Handles unknown characters gracefully
   - Returns generated text with metadata

### Usage

#### Setup Environment

We recommend using `uv` for Python virtual environment management, though you can use your favorite Python environment manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -r mlx-gpt2/requirements.txt
```

#### Training
```bash
# Using uv (recommended)
uv run mlx-gpt2/train.py

# Or with your Python environment
python mlx-gpt2/train.py
```

#### Inference (CLI)
```bash
# Basic usage with uv
uv run mlx-gpt2/inference.py

# With custom prompt
uv run mlx-gpt2/inference.py --prompt "Once upon a time" --max_tokens 200 --temperature 0.8

# Or with your Python environment
python mlx-gpt2/inference.py --prompt "Once upon a time"
```

#### API Server
```bash
# Start the server with uvicorn
cd mlx-gpt2
uvicorn api:app --reload

# Make a request
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50, "temperature": 0.8}'
```

### Technical Notes

- The model uses character-level tokenization
- Context length is set to 8 tokens (matching the training configuration)
- Model architecture: 3 layers, 4 attention heads, 128-dimensional embeddings
- Trained on the provided `input.txt` file