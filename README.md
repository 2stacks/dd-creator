# Diffusion Dataset Creator

A local, GPU-accelerated tool for creating high-quality training datasets for diffusion models (FLUX.1, SDXL, SD 1.5, etc.).

![Step 2: Image Tools](assets/screenshot-image-tools.png)

## Features

- **Image Upscaling** - Real-ESRGAN upscaling via Spandrel for enhancing low-resolution source images
- **Background Removal** - BiRefNet-powered automatic mask generation and transparency
- **Auto-Captioning** - Multiple model options:
  - Florence-2 (Base/Large) - Fast, detailed captions
  - BLIP (Base/Large) - Lightweight natural language captions
  - JoyCaption - High-quality descriptive captions (BF16 or 8-bit quantized)
  - WD14 Taggers (ONNX) - Booru-style tags via ViT, ConvNext, or SwinV2
- **Non-Destructive Workflow** - Separate input/output directories preserve originals
- **Local Processing** - Runs entirely on your machine, no cloud dependencies

## Screenshots

| Import | Image Tools | Captioning |
|--------|-------------|------------|
| ![Import](assets/screenshot-import.png) | ![Tools](assets/screenshot-image-tools.png) | ![Captioning](assets/screenshot-captioning.png) |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### GPU Memory Requirements

| Model | VRAM |
|-------|------|
| Florence-2 | ~4GB |
| BLIP | ~2-4GB |
| JoyCaption (BF16) | ~17GB (requires 20GB+ GPU) |
| JoyCaption (8-bit) | ~12-16GB (requires 16GB+ GPU) |
| WD14 ONNX | ~2GB |
| BiRefNet | ~4GB |
| Real-ESRGAN | ~2-4GB |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dd-creator.git
cd dd-creator

# Run (uv auto-creates venv and installs dependencies)
uv run app.py
```

Open your browser to `http://127.0.0.1:7860`

### Upscaler Models

Place Real-ESRGAN `.pth` or `.safetensors` model files in the `.models/` directory. Popular options:
- [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN)
- [4x-UltraSharp](https://openmodeldb.info/models/4x-UltraSharp)

## Usage

The wizard guides you through 4 steps:

1. **Import** - Select source directory and output location
2. **Image Tools** - Per-image editing (upscale, masks, transparency) or bulk processing with smart resize/upscale routing
3. **Captioning** - Generate and edit captions with powerful tools:
   - Batch generation with prefix/suffix tags (trigger words, quality tags)
   - Automatic Danbooru rating tag filtering (optional, on by default)
   - Search/filter images by caption content
   - Hygiene tools: fix formatting, deduplicate tags, undo changes
   - Bulk operations: add/remove tags, search & replace across all captions
4. **Export** - Review and finalize your dataset

## Project Structure

```
dd-creator/
├── app.py                 # Gradio application entry point
├── src/
│   ├── core/
│   │   ├── state.py       # Project state management
│   │   ├── captioning.py  # VLM/tagger model wrappers
│   │   ├── segmentation.py # BiRefNet background removal
│   │   └── upscaling.py   # Real-ESRGAN upscaling
│   └── ui/
│       ├── wizard.py      # 4-step guided workflow
│       └── dashboard.py   # Advanced tools (WIP)
├── .models/               # User-provided upscaler models
├── .hf_cache/             # HuggingFace model cache
└── assets/                # README screenshots
```

## Development

```bash
# Add dependencies
uv add <package_name>

# Run with auto-reload (if using gradio dev mode)
uv run app.py
```

## License

MIT
