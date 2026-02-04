# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diffusion Dataset Creator is a local, GPU-accelerated Gradio application for creating training datasets for diffusion models. It provides a non-destructive workflow with separate input/output directories, supporting upscaling, background removal, and captioning.

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
    captioning.py         # ML model wrappers (Florence2, BLIP, JoyCaption, WD14 ONNX)
    segmentation.py       # BiRefNet background removal and mask generation
    upscaling.py          # Real-ESRGAN/Spandrel upscaling with tiled processing
    birefnet_impl/        # Local BiRefNet model implementation (avoids trust_remote_code)
  ui/
    wizard.py             # 4-step guided workflow (Import → Image Tools → Captioning → Export)
    dashboard.py          # Advanced tools placeholder
```

### Key Patterns

**State Management:** `global_state` singleton in `state.py` holds project data:
- `source_directory`, `output_directory` - separate input/output paths
- `image_paths` - list of source images
- `captions`, `masks`, `upscaled`, `transparent` - dicts mapping source paths to outputs

**Model Lifecycle:** All GPU models follow lazy-load pattern. Models are loaded on-demand and must be explicitly unloaded with `model.unload_model()` followed by `torch.cuda.empty_cache()` and `gc.collect()` to free VRAM. Global instance getters: `get_captioner()`, `get_segmenter()`, `get_upscaler()`.

**File I/O:**
- Captions: `.txt` files (UTF-8) in output directory
- Masks: Binary PNG files in output directory
- Model cache: `.hf_cache/` directory (HuggingFace models)
- Upscaler models: `.models/` directory for user-provided `.pth`/`.safetensors` files

## Step 2: Image Tools UI Structure

The Image Tools step has two main areas:

**Workbench (per-image editing):**
- Original tab: View/resize/save individual images
- Upscale tab: Real-ESRGAN upscaling with model selector and "Unload All Models" button
- Mask tab: Generate segmentation masks with invert option
- Transparent tab: Remove backgrounds with alpha threshold

**Bulk Actions accordion (batch processing):**
- Left column: Smart Processing (passthrough/upscale/downscale routing by image size)
- Right column: Mask Generation, Background Removal, Bypass Editing (copy source to output)

Gallery selection resets workbench to Original tab. All save buttons use `variant="primary"` (purple).

## Step 3: Captioning UI Structure

The Captioning step has three main areas:

**Batch Generation accordion (open by default):**
- Model dropdown + Threshold slider (WD14 only)
- Prefix/Suffix textboxes for static tags applied to all captions
- "Generate Tags for All Images" button

**Main area (two-column layout, 50/50 split):**
- Left column: Search/filter textbox + Gallery (4x4 grid, height=700)
- Right column: Image preview + Caption editor (6 lines) + Hygiene tools + Save & Next button

**Hygiene tools:**
- Fix Format: Normalize comma spacing, remove empty tags
- Dedup Tags: Remove duplicate tags (case-insensitive)
- Undo Changes: Revert to caption before last hygiene action

**Bulk Tagging Tools accordion (closed by default):**
- Add Tags: Prepend/Append radio + tags input + Add to All button
- Remove Tags: Comma-separated tags input + Remove from All button
- Search & Replace: Exact/Partial match radio + semicolon-separated pairs + Replace All button

Gallery filtering tracks displayed images separately (`_displayed_images`) so selection works correctly with filtered results.

## Gradio 6.0+ Specifics

- Pass `theme` to `launch()`, not `Blocks()`
- Use `gr.ImageEditor` instead of deprecated `Image(tool="sketch")`
- `ImageEditor` with `type="filepath"` returns dict with `"layers"` key
- Images downscaled to 1024px max before editor display for browser stability

## GPU Memory Reference

**Captioning:**
- Florence-2 Base/Large: ~4GB VRAM
- BLIP Base/Large: ~2-4GB VRAM
- JoyCaption BF16: ~17GB VRAM (requires 20GB+ GPU)
- JoyCaption 8-bit: ~12-16GB VRAM (requires minimum 16GB GPU)
- WD14 ONNX taggers: ~2GB VRAM (uses onnxruntime-gpu)

**Image Processing:**
- BiRefNet segmentation: ~4GB VRAM
- Real-ESRGAN upscaling: ~2-4GB VRAM (tiled processing for large images)
