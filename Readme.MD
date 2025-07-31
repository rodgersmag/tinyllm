# TinyLLM: A Research Project for Small Language Models

This repository is a research project dedicated to the training and implementation of small, specialized language models. Our goal is to explore various architectures and train different types of models using publicly available datasets from sources like Hugging Face.

The ultimate ambition of this project is to incrementally build our expertise and capabilities to train a highly proficient coding model, with the aspirational goal of reaching the level of performance seen in models like Anthropic's Claude Opus.

## Completed Models

We have successfully implemented and trained the following models:

*   **`mlx-gpt2`**: A character-level GPT-2 model implemented in MLX for Apple Silicon.
*   **`tinystories`**: A PyTorch-based GPT model trained on the TinyStories dataset, designed for GPU training and local MPS inference.

## Future Models

Our roadmap includes the development of the following models:

- [ ] Image Classifier
- [ ] Image gen 
- [ ] Music gen
- [ ] Text-to-Speech (TTS)
- [ ] Translation Model
- [ ] Pose Detection
- [ ] Virtual try-on Model
- [ ] Breast Cancer and Testicular cancel detector
- [ ] Lotto Number Generator
- [ ] facial recognition
- [ ] Fine tune Lamma 2 , qwen and deepseek 
- [ ] reinforcement learning & reinforcement learning with Human feedback 
- [ ] Lora 
- [ ] Qlora
- [ ] GRPO 
- [ ] Advanced Coding Model

## How to Contribute

We welcome contributions to this research project! To maintain consistency and ensure that others can easily replicate the work, please follow these guidelines:

1.  **Environment Management**: Use `uv` for creating virtual environments and managing dependencies. This ensures a consistent and fast setup process.

2.  **Development Workflow**:
    *   **GPU First**: Begin by developing and validating your model training script in a GPU-accelerated environment. We primarily use Google Colab for this, as it provides free access to T4 GPUs.
    *   **Structure for Reproducibility**: Once the model is successfully trained, structure the code into a dedicated folder for your model. This folder must include:
        *   A `README.md` with instructions on how to set up and run your code. Remember to include any training result images (like loss curves) in the README.
        *   A `requirements.txt` file listing all necessary dependencies.
        *   The core scripts: `train.py`, `inference.py`, and `api.py`.

3.  **Example Directory Structure**: Each completed model directory should follow a structure similar to this:
    ```
    Folder-name/
    ├── api.py
    ├── inference.py
    ├── main.py - #GPU Code
    ├── model.pth
    ├── output.txt
    ├── README.md
    ├── requirements.txt
    ├── train.py
    ├── training_loss_ma.png
    └── training_loss.png
    ```

This structured approach ensures that every model in this repository can be easily trained, tested, and integrated by other users.

---

A special thank you to **Google Colab** for providing the free GPU resources that make this research possible.
