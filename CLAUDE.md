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
app.py                    # Entry point - Gradio interface setup
src/
  core/
    state.py              # ProjectState singleton - shared across all modules
    captioning.py         # ML model wrappers (Qwen2.5-VL, BLIP, JoyCaption, WD14 ONNX)
    segmentation.py       # BiRefNet background removal and mask generation
    upscaling.py          # Spandrel upscaling with tiled processing
    inpainting.py         # LaMa + Stable Diffusion inpainting backends
    sam_segmenter.py      # MobileSAM click-to-segment for inpainting masks
    smart_crop.py         # Face-centric training crop generation
    birefnet_impl/        # Local BiRefNet model implementation (avoids trust_remote_code)
  ui/
    wizard.py             # 4-step guided workflow (Import → Image Tools → Captioning → Export)
    dashboard.py          # Advanced tools placeholder
```

### Key Patterns

**State Management:** `global_state` singleton in `state.py` holds project data:
- `source_directory`, `output_directory` - separate input/output paths
- `image_paths` - list of source images
- `captions`, `masks`, `upscaled`, `transparent`, `inpainted` - dicts mapping source paths to outputs

**Model Lifecycle:** All GPU models follow lazy-load pattern. Models are loaded on-demand and must be explicitly unloaded with `model.unload_model()` followed by `torch.cuda.empty_cache()` and `gc.collect()` to free VRAM. Global instance getters: `get_captioner()`, `get_segmenter()`, `get_upscaler()`, `get_lama_inpainter()`, `get_sd_inpainter()`, `get_sam_segmenter()`.

**RGBA Handling:** All captioning models use `_ensure_rgb()` in `captioning.py` to convert images to RGB. RGBA images (transparent PNGs from background removal) are composited onto a white background before captioning. This prevents models from seeing the original background pixels that `PIL.Image.convert("RGB")` would preserve when simply dropping the alpha channel.

**File I/O:**
- Captions: `.txt` files (UTF-8) in output directory
- Masks: Binary PNG files in output directory
- Inpainted images: `<basename>_inpainted.jpg` (98% quality) in output directory
- Model cache: `~/.cache/huggingface/` (standard HuggingFace location, no local override)
- Upscaler models: `models/` directory for user-provided `.pth`/`.safetensors` files
- SAM checkpoint: `models/mobile_sam.pt` (auto-downloaded from HuggingFace)

## Step 1: Project Setup UI Structure

Two-column layout with tabbed sections for source data and workspace configuration.

**Layout:** Two equal columns (50/50 split)

**Column 1: Source Data (Input)**
- Tabs: "Local Folder" | "Upload"
- Local Folder tab:
  - "Browse Directories" accordion with FileExplorer
  - Image Source Directory textbox (default: `datasets/input`)
- Upload tab:
  - gr.File component (file_count="multiple") for uploading individual files or entire folders
  - Info text explaining upload destination (custom folder or `datasets/input/uploads/`)
  - Upload status textbox

**Column 2: Workspace (Output)**
- Tabs: "New Project" | "Continue Existing"
- New Project tab:
  - "About Projects" accordion explaining new vs. continue existing workflows
  - Project Name textbox
- Continue Existing tab:
  - "Browse Existing Projects" accordion with FileExplorer
  - Output Directory textbox

**Common Elements:**
- Full-width "Scan & Initialize Project" button (primary, large)
- Collapsible "Console Log" accordion with status textbox
- "Next >" navigation button

**FileExplorer Usage:** Each explorer starts in its default directory (`datasets/input` or `datasets/output`) and shows files. Clicking any file selects its containing directory. Directories are created automatically if they don't exist.

**Upload Staging:** If the Local Folder path is set to a custom directory, uploads go directly into that folder. If it's the default (`datasets/input`), a new timestamped folder is created at `datasets/input/uploads/upload_<timestamp>/`. This lets users accumulate images in a chosen folder across multiple upload sessions.

**Scan Logic (value-based, no tab state tracking):**
- Source: Uses upload staging only if local folder is still the default. A custom local folder path always takes priority.
- Output: Project Name (creates `datasets/output/<name>/`) > existing Output Directory. Using the bare `datasets/output/` as output is not allowed — users must define a project name or select an existing project folder.
- Scan reports existing caption counts: output captions vs source-only captions (`.txt` in source dir with no corresponding output `.txt`). Source-only captions are loaded into memory for review in Step 3 but are not saved until the user explicitly saves them.

## Step 2: Image Tools UI Structure

Two-column layout (40/60 split) with gallery height=700 and image viewer height=700. Image previews use `show_label=False` — tabs provide context.

**Left column (40%):** Image gallery + unified Status textbox (2 lines, scrollable) below gallery. All workbench actions across all tabs output status messages to this single status bar. First image auto-selects when Step 2 loads.

**Workbench (per-image editing, right column 60%):**
- Resize tab: View/resize/save individual images
- Upscale tab: Spandrel upscaling with model selector, Save Original / Upscale / Save Upscale buttons, "Unload All Models" button
- Inpaint tab: Mask creation + inpainting (see below)
- Smart Crop tab: Face-centric training crops (see below)
- Mask tab: `gr.Gallery` — Generate BiRefNet segmentation masks with invert option (processes all crops when available)
- Transparent tab: `gr.Gallery` — Remove backgrounds with alpha threshold (processes all crops when available)

**Inpaint tab sub-tabs:**
- Manual Mask: Click two points to draw rectangle mask regions (undo/clear)
- SAM Click: MobileSAM click-to-segment for object-based masks (undo/clear)
- Watermark Preset: Quick rectangle masks for common watermark positions with width/height % sliders (add/undo/clear)
- Generate Inpaint: Backend selector (LaMa/SD 1.5/SDXL), prompt (SD only), advanced settings accordion (negative prompt, steps, guidance, strength), Run Inpaint / Save Inpaint buttons, Unload All Models button, Model Information accordion

**Image source priority chain:** All downstream tabs use the best available source image via `_get_inpaint_source()`. Priority: inpainted > upscaled > resized > original. When resize, upscale, inpaint, or smart crop completes, downstream tab previews update automatically via `.then()` chains.

**Inpaint source consistency:** All inpaint mask functions (rectangles, SAM, watermarks, overlays) receive `upscaled_image_state`, `resized_image_state`, and `inpaint_result_state` as inputs and use `_get_inpaint_source()` to get the correct image dimensions. This ensures click coordinates match the displayed image, not the original file on disk. The `workbench_inpaint` component uses `interactive=False` — `.select()` events still capture click coordinates without the upload/edit UI interfering.

**Inpaint tool state tracking:** Each sub-tab uses `TabItem.select()` to update an `inpaint_tool_state` (`gr.State`). The image click handler routes to Manual Mask or SAM Click based on this state.

**Mask compositing:** All inpaint mask sources (rectangles, SAM mask, watermark presets) are combined via binary OR in `composite_masks()`. A red overlay preview is generated server-side with `create_mask_overlay()`. Preset masks are stored as a list for undo support.

**Smart Crop → Mask/Transparent pipeline:** When smart crop results exist, the Mask and Transparent galleries show crop images as previews. Generating masks or transparents processes all crops through BiRefNet, producing gallery results with per-crop labels. Save functions use `{basename}_{crop_label}_mask.png` / `{basename}_{crop_label}_transparent.png` filenames for crops, or backward-compatible `{basename}_mask.png` / `{basename}_transparent.png` for single images. State format: `[(label, img), ...]` for save functions, `[(img, label), ...]` for gallery display.

**Bulk Actions accordion (batch processing):**
- Left column: Smart Processing (passthrough/upscale/downscale routing by image size)
- Right column: Mask Generation, Background Removal, Bypass Editing (copy source to output)

Gallery selection resets workbench to Resize tab. All save buttons use `variant="primary"` (purple).

## Step 3: Captioning UI Structure

Two-column layout (40/60 split) matching Step 2's "Left Sidebar + Right Tabbed Workspace" pattern.

**Left column (40%):** Filter textbox + Gallery (4x4 grid, height=700) + Status textbox (2 lines).

**Right column (60%) — Tabbed Workspace:**

- **Editor tab (default):** Image preview (height=500) + Caption editor (6 lines) + Hygiene buttons row (Fix Format, Dedup Tags, Undo Changes) + Help accordion (closed) + Save & Next (primary, lg, scale=3) / Delete Image (stop, lg, scale=1)
- **Auto-Tagging tab:** Image preview (height=500, synced with Editor preview via `.then()` chains) + Model dropdown (default: BLIP-Base) + Model Information accordion (closed) + WD14 options column (Threshold slider + Filter rating tags checkbox, auto-hidden when non-WD14 model selected) + Prefix/Suffix textboxes (same row) + Generate buttons row (Generate for All Images, Generate for Selected Image)

**Hygiene tools:**
- Fix Format: Normalize comma spacing, collapse extra whitespace, strip invisible Unicode chars, remove empty tags
- Dedup Tags: Remove duplicate tags (case-insensitive), strips invisible chars before comparing
- Undo Changes: Reload caption from disk (replaces in-memory undo tracking)
- All hygiene tools use a `gr.State` + `.then()` chain to update the textbox (workaround for Gradio bug where a textbox used as both input and output of the same callback doesn't visually refresh)

**Bulk Edit Tools accordion (closed by default, full-width below bottom bar):**
- Add Tags: Prepend/Append radio + tags input + Add to All button
- Remove Tags: Comma-separated tags input + Remove from All button
- Search & Replace: Partial/Exact match radio (default: Partial) + semicolon-separated pairs + Replace All button. Partial match does substring replace (works with natural language captions). Exact match replaces whole comma-separated tags only.

**Rating Tag Filtering:** Filters Danbooru rating tags during batch generation. Removes prefixed tags (rating:general, etc.) anywhere, and standalone tags (general, sensitive, questionable, explicit) from first/last position only.

Gallery filtering tracks displayed images separately (`_displayed_images`) so selection works correctly with filtered results.

**Source Caption Fallback:** `get_output_images()` falls back to `global_state.captions` (populated during Step 1 scan) when an output image has no `.txt` file on disk. Matching is by filename basename. This allows source-only captions to appear in the editor for review.

**Next Button Validation:** The Step 3 "Next >" button checks that all output images have a saved `.txt` caption file on disk before proceeding to Step 4. Images with only in-memory captions (from source fallback) are flagged.

## Step 4: Export UI Structure

- Session stats JSON display showing Source Images, Output Images, Saved Captions, Masks, Transparent Images, Upscaled Images, Inpainted Images
- Stats are computed from actual output directory contents (not in-memory state) for accuracy
- Refresh Stats button + Back navigation

**HuggingFace tab — Push to Hub:**
- Collapsible "Push to HuggingFace Hub" accordion inside the HuggingFace export tab
- HF Token textbox (`type="password"`, optional if already logged in via `huggingface-cli login` or `HF_TOKEN` env var)
- Repository Name textbox (auto-populated from project name on tab load)
- Private Repository checkbox (default True)
- Push to Hub button → calls `push_to_huggingface()` from `src/core/export.py`
- Token resolution: explicit UI field → `huggingface_hub.get_token()` (env var → `~/.cache/huggingface/token`)
- Token is never persisted by our code — only passed in-memory to `HfApi(token=...)`

## Gradio 6.0+ Specifics

- Pass `theme` to `launch()`, not `Blocks()`
- Use `gr.ImageEditor` instead of deprecated `Image(tool="sketch")`
- `ImageEditor` with `type="filepath"` returns dict with `"layers"` key
- Images downscaled to 1024px max before editor display for browser stability
- **Textbox input/output bug:** When a Textbox is used as both input and output of the same `.click()` callback, the textbox may not visually refresh. Workaround: write result to a hidden `gr.State`, then chain `.then(lambda x: x, inputs=state, outputs=textbox)` to push the value in a separate event cycle.

## GPU Memory Reference

**Captioning:**
- BLIP Base/Large: ~1GB VRAM
- Qwen2.5-VL 3B (bf16/fp16): ~10GB VRAM
- Qwen2.5-VL 7B (4-bit): ~9GB VRAM
- Qwen2.5-VL 7B (8-bit): ~12GB VRAM
- JoyCaption BF16: ~14GB VRAM (requires 16GB+ GPU)
- JoyCaption 8-bit: ~10GB VRAM (requires 12GB+ GPU)
- WD14 ONNX taggers: ~1GB VRAM (uses onnxruntime-gpu)

**Image Processing:**
- BiRefNet segmentation: ~4GB VRAM
- Spandrel upscaling: ~2-4GB VRAM (tiled processing for large images)
- MobileSAM (click-to-segment): ~1GB VRAM (cached image embeddings)
- LaMa inpainting: ~2GB VRAM (fast, no prompt needed)
- SD 1.5 inpainting: ~6GB VRAM (prompt-guided)
- SDXL inpainting: ~10GB VRAM (highest quality, requires 12GB+ GPU)
