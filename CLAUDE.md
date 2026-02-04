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

## Step 1: Project Setup UI Structure

Two-column layout with tabbed sections for source data and workspace configuration.

**Layout:** Two equal columns (50/50 split)

**Column 1: Source Data (Input)**
- Tabs: "Local Folder" | "Upload"
- Local Folder tab:
  - Path to Source Images textbox (default: `datasets/input`)
  - "Browse Directories" accordion with FileExplorer
- Upload tab:
  - gr.File component (file_count="multiple") for uploading individual files or entire folders
  - Upload status textbox

**Column 2: Workspace (Output)**
- Tabs: "New Project" | "Continue Existing"
- New Project tab:
  - Project Name textbox
  - Info text showing path that will be created
- Continue Existing tab:
  - Output Directory textbox
  - "Browse Existing Projects" accordion with FileExplorer

**Common Elements:**
- Full-width "Scan & Initialize Project" button (primary, large)
- Collapsible "Console Log" accordion with status textbox
- "Next >" navigation button

**FileExplorer Usage:** Each explorer starts in its default directory (`datasets/input` or `datasets/output`) and shows files. Clicking any file selects its containing directory. Directories are created automatically if they don't exist.

**Upload Staging:** If the Local Folder path is set to a custom directory, uploads go directly into that folder. If it's the default (`datasets/input`), a new timestamped folder is created at `datasets/input/uploads/upload_<timestamp>/`. This lets users accumulate images in a chosen folder across multiple upload sessions.

**Scan Logic (value-based, no tab state tracking):**
- Source: Uses upload staging only if local folder is still the default. A custom local folder path always takes priority.
- Output: Project Name (creates `datasets/output/<name>/`) > existing Output Directory > fallback to `datasets/output/`.

## Step 2: Image Tools UI Structure

Two-column layout (40/60 split) with gallery height=700 and image viewer height=500.

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

Two-column layout (40/60 split) with gallery height=700 and image preview height=500.

**Batch Generation accordion (open by default):**
- Left column: Model dropdown (sorted alphabetically) + collapsible VRAM Requirements accordion
- Right column: Threshold slider (WD14 only) + Filter rating tags checkbox (on by default)
- Prefix/Suffix textboxes for static tags applied to all captions
- "Generate Tags for All Images" button

**Main area (two-column layout, 40/60 split):**
- Left column: Gallery (4x4 grid, height=700) + Search/filter textbox below
- Right column: Image preview (height=500) + Caption editor (6 lines) + Hygiene tools + Save & Next button

**Hygiene tools:**
- Collapsible help accordion explaining button functions (above buttons)
- Fix Format: Normalize comma spacing, remove empty tags
- Dedup Tags: Remove duplicate tags (case-insensitive)
- Undo Changes: Revert to caption before last hygiene action

**Bulk Tagging Tools accordion (closed by default):**
- Add Tags: Prepend/Append radio + tags input + Add to All button
- Remove Tags: Comma-separated tags input + Remove from All button
- Search & Replace: Exact/Partial match radio + semicolon-separated pairs + Replace All button

**Rating Tag Filtering:** Filters Danbooru rating tags during batch generation. Removes prefixed tags (rating:general, etc.) anywhere, and standalone tags (general, sensitive, questionable, explicit) from first/last position only.

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
