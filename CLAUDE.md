# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diffusion Dataset Creator is a local, GPU-accelerated Gradio application for creating training datasets for diffusion models. It provides a non-destructive workflow that preserves original images while managing captions and masks.

## Commands

```bash
# Run the application (launches on http://127.0.0.1:7860)
uv run app.py

# Add dependencies
uv add <package_name>
```

## Architecture

```
app.py                    # Entry point - Gradio interface setup, env config
src/
  core/
    state.py              # ProjectState singleton - shared across all modules
    captioning.py         # ML model wrappers (Florence2, BLIP, JoyCaption, WD14)
  ui/
    wizard.py             # 5-step guided workflow (Import → Caption → Review → Masks → Export)
    dashboard.py          # Advanced tools placeholder
```

### Key Patterns

**State Management:** `global_state` singleton in `state.py` holds project data (source directory, image paths, captions dict, masks dict). All modules mutate this shared state directly.

**Model Lifecycle:** Captioning models follow lazy-load pattern via `BaseCaptioner` base class. Models are loaded on-demand and must be explicitly unloaded with `model.unload()` followed by `torch.cuda.empty_cache()` and `gc.collect()` to free VRAM.

**File I/O:**
- Captions: `.txt` files (UTF-8) saved alongside images with matching names
- Masks: Binary PNG files in `masks/` subdirectory
- Model cache: `.hf_cache/` directory (local, avoids permission issues)

## Gradio 6.0+ Specifics

- Pass `theme` to `launch()`, not `Blocks()`
- Use `gr.ImageEditor` instead of deprecated `Image(tool="sketch")`
- `ImageEditor` with `type="filepath"` returns dict with `"layers"` key
- Images downscaled to 1024px max before editor display for browser stability

## GPU Memory Reference

- Florence-2: <4GB VRAM
- JoyCaption BF16: ~17GB VRAM
- JoyCaption 4-bit: ~8GB VRAM
