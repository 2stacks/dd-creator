# Project Context: Diffusion Dataset Creator

## Operational Guidelines
*   **Shell Commands**: When running shell commands, assume you are in a restricted Docker environment. Do not attempt to use `sudo` unless explicitly asked or if the error indicates permissions are required.
*  **Gemini CLI Sandbox**: This project uses Gemini CLI in sandbox mode.

## Project Overview
This project is a local GUI application designed to streamline the creation of training datasets for image diffusion models (e.g., Stable Diffusion, FLUX.1). It provides a non-destructive workflow for captioning and masking images without permanently altering the source files (relying on aspect ratio bucketing during training).

**Key Features:**
*   **Auto-Captioning**: Integrates Qwen2.5-VL VLM (Vision Language Model) locally for generating detailed image descriptions.
*   **Masking**: Integrated drawing tool for creating in-painting masks.
*   **UI Modes**: Offers a linear "Wizard" mode for beginners and an extensible "Dashboard" mode.
*   **Output**: Generates standard `.txt` caption files and mask images compatible with major training scripts (OneTrainer, Kohya_ss, AI-Toolkit).

## Architecture
The application is built using **Python** and **Gradio**.

### Directory Structure
*   `app.py`: The main entry point. Sets up the Gradio interface and themes.
*   `src/`: Core source code.
    *   `core/`: Backend logic.
        *   `state.py`: Singleton `ProjectState` class to manage the active session (image paths, captions, masks).
        *   `captioning.py`: Wrapper for the `Qwen2.5-VL` model using `transformers` and `torch`.
    *   `ui/`: Frontend components.
        *   `wizard.py`: Implements the 4-step guided workflow (Import -> Image Tools -> Captioning -> Export).
        *   `dashboard.py`: Placeholder for advanced features.
*   `pyproject.toml`: Dependency configuration managed by `uv`.

### Key Dependencies
*   **Gradio**: Web UI framework.
*   **Torch/Transformers**: DL backend for running Qwen2.5-VL.
*   **Pillow**: Image processing.
*   **UV**: Package and environment manager.

## Building and Running

This project uses `uv` for dependency management.

### Prerequisites
*   **UV**: Installed via `curl -LsSf https://astral.sh/uv/install.sh | sh` (or equivalent).
*   **GPU**: NVIDIA GPU with CUDA drivers is highly recommended for Qwen2.5-VL inference.

### Commands

**Start the Application:**
```bash
uv run app.py
```
This command automatically sets up the virtual environment (`.venv`), installs dependencies defined in `pyproject.toml`, and launches the Gradio server (default: `http://127.0.0.1:7860`).

**Add Dependencies:**
```bash
uv add <package_name>
```

## Development Conventions
*   **State Management**: The application uses a global singleton (`global_state`) in `src/core/state.py` to share data between Gradio steps.
*   **UI Components**: Complex UI logic is split into separate modules within `src/ui/`.
*   **Model Handling**: Heavy models (like Qwen2.5-VL) are loaded lazily to conserve VRAM until needed and can be unloaded manually (or via logic) to free resources.
*   **Gradio 6.0 Compatibility**: Ensure `theme` parameters are passed to `launch()` rather than `Blocks()`, and use `gr.ImageEditor` instead of deprecated `gr.Image(tool="sketch")`.
