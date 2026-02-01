# dd-creator

A Gradio-based tool to help users create high-quality datasets for training diffusion models (FLUX.1, SDXL, etc.).

## Features

- **Local Processing**: Runs entirely on your machine using your GPU.
- **Auto-Captioning**: Uses State-of-the-Art VLM (Vision Language Models) like Florence-2 for detailed captions.
- **Masking**: Built-in tool to draw training masks for in-painting or specific concept training.
- **Dual Modes**:
    - **Wizard Mode**: A guided step-by-step process for beginners.
    - **Dashboard Mode**: A flexible interface for advanced users.
- **Non-Destructive**: Preserves original image aspect ratios (relies on Bucketing during training).

## Installation & Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

1.  **Prerequisites**:
    - [uv](https://github.com/astral-sh/uv) installed.
    - NVIDIA GPU with CUDA drivers installed (Recommended).

2.  **Run the Application**:
    `uv` will automatically create the virtual environment and install dependencies the first time you run the app.

    ```bash
    uv run app.py
    ```

    Open your browser to the local URL provided (usually `http://127.0.0.1:7860`).

## Project Structure

```text
dd-creator/
├── app.py              # Main entry point (Gradio App)
├── src/
│   ├── core/           # Backend logic
│   │   ├── state.py    # State management
│   │   ├── captioning.py # Florence-2 VLM integration
│   │   └── segmentation.py # Logic for masking/segmentation
│   └── ui/             # Frontend components
│       ├── wizard.py   # Guided workflow UI
│       └── dashboard.py # Advanced dashboard UI
├── datasets/           # Directory for storing dataset inputs/outputs
├── assets/             # Static assets
└── pyproject.toml      # Project configuration and dependencies
```

## Development

To add new dependencies:

```bash
uv add <package_name>
```