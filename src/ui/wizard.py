import gradio as gr
import os
import torch
import gc
import numpy as np
from PIL import Image, ImageOps
from src.core.state import global_state
from src.core.captioning import get_captioner
from src.core.segmentation import get_segmenter
from src.core.upscaling import (
    get_upscaler, get_available_models, should_upscale, resize_to_target,
    process_image, ProcessorAction
)

def render_wizard():
    # --- HELPER FUNCTIONS (Logic isolated from UI Layout) ---
    
    def browse_directory(initial_path=None):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            # Resolve starting directory
            start_dir = os.getcwd()
            if initial_path:
                temp_path = os.path.abspath(initial_path)
                # Traverse up until we find a directory that exists
                while temp_path and not os.path.exists(temp_path):
                    parent = os.path.dirname(temp_path)
                    if parent == temp_path: # Prevent infinite loop at root
                        break
                    temp_path = parent
                
                if os.path.exists(temp_path) and os.path.isdir(temp_path):
                    start_dir = temp_path
            
            directory = filedialog.askdirectory(initialdir=start_dir)
            root.destroy()
            return directory if directory else initial_path
        except Exception as e:
            print(f"Browse error: {e}")
            return initial_path

    def scan_action(path, output_path, project_name):
        if not path or not os.path.isdir(path):
            return "Please select a valid directory.", gr.update(interactive=False)
        
        # Combine output path with project name if provided
        final_output_path = output_path
        if project_name and project_name.strip():
            final_output_path = os.path.join(output_path, project_name.strip())

        # Create output dir if it doesn't exist
        if final_output_path and not os.path.exists(final_output_path):
            try:
                os.makedirs(final_output_path, exist_ok=True)
            except Exception as e:
                return f"Error creating output directory: {e}", gr.update(interactive=False)

        msg = global_state.scan_directory(path, final_output_path)
        count = len(global_state.image_paths)
        
        # Add a note about where files are saving if it's different from source
        if final_output_path and os.path.abspath(final_output_path) != os.path.abspath(path):
            msg += f"\nOutput Directory: {os.path.abspath(final_output_path)}"
            
        return msg, gr.update(interactive=(count > 0))

    def run_captioning_action(model_name, progress=gr.Progress()):
        if not global_state.image_paths:
            return "No images to caption."
        captioner = None
        total = len(global_state.image_paths)
        success_count = 0
        try:
            captioner = get_captioner(model_name)
            for i, img_path in enumerate(progress.tqdm(global_state.image_paths)):
                try:
                    cap = captioner.generate_caption(img_path)
                    global_state.captions[img_path] = cap
                    # Auto-save to .txt (using output path logic)
                    txt_path = global_state.get_output_path(img_path, ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(cap)
                    success_count += 1
                except Exception as e:
                    print(f"Captioning error for {img_path}: {e}")
        except Exception as e:
            return f"Model Error: {e}"
        finally:
            if captioner:
                captioner.unload_model()
        return f"Completed. {success_count}/{total} images captioned."

    def on_select_image(evt: gr.SelectData):
        # evt.index is the index in the gallery list
        index = evt.index
        if index < len(global_state.image_paths):
            path = global_state.image_paths[index]
            caption = global_state.captions.get(path, "")
            return path, caption
        return None, ""

    def save_caption_action(path, new_caption):
        if path and path in global_state.captions:
            global_state.captions[path] = new_caption
            txt_path = global_state.get_output_path(path, ".txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_caption)
                return f"Saved caption for {os.path.basename(path)}"
            except Exception as e:
                return f"Error: {e}"
        return "Error: No image selected"

    def bulk_replace_action(find_text, replace_text):
        if not find_text: return "Error: Find text empty."
        count = 0
        for path, cap in global_state.captions.items():
            if find_text in cap:
                new_cap = cap.replace(find_text, replace_text)
                global_state.captions[path] = new_cap
                try:
                    txt_path = global_state.get_output_path(path, ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f: f.write(new_cap)
                    count += 1
                except: pass
        return f"Updated {count} captions."

    def bulk_add_action(text, mode):
        if not text: return "Error: Text empty."
        count = 0
        for path, cap in global_state.captions.items():
            new_cap = f"{text}{cap}" if mode == "prepend" else f"{cap}{text}"
            global_state.captions[path] = new_cap
            try:
                txt_path = global_state.get_output_path(path, ".txt")
                with open(txt_path, "w", encoding="utf-8") as f: f.write(new_cap)
                count += 1
            except: pass
        return f"{mode.capitalize()}ed text to {count} captions."

    def on_gallery_select(evt: gr.SelectData):
        """Handle gallery selection in Step 2."""
        index = evt.index
        if index < len(global_state.image_paths):
            path = global_state.image_paths[index]
            # Load original image
            try:
                img = Image.open(path).convert("RGB")
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                status_text = f"Selected: {os.path.basename(path)} ({width}x{height}px)"
            except Exception as e:
                img = None
                status_text = f"Error loading image: {e}"

            # Check for existing processed versions
            proc_status = global_state.get_processing_status(path)
            existing_upscaled = None
            existing_mask = None
            existing_transparent = None

            if proc_status["has_upscaled"] and path in global_state.upscaled:
                try:
                    existing_upscaled = Image.open(global_state.upscaled[path])
                except:
                    pass

            if proc_status["has_mask"] and path in global_state.masks:
                try:
                    existing_mask = Image.open(global_state.masks[path])
                except:
                    pass

            if proc_status["has_transparent"] and path in global_state.transparent:
                try:
                    existing_transparent = Image.open(global_state.transparent[path])
                except:
                    pass

            # Return: path, original_img, status for each tab, existing processed images, reset states, tab selection
            return (
                path,                    # selected_path_state
                img,                     # workbench_original
                status_text,             # original_status
                existing_upscaled,       # workbench_upscaled
                "Run upscale to generate." if not existing_upscaled else "Upscaled version loaded.",  # upscale_status
                existing_mask,           # workbench_mask
                "Create mask to generate." if not existing_mask else "Mask loaded.",  # mask_status
                existing_transparent,    # workbench_transparent
                "Create transparent to generate." if not existing_transparent else "Transparent loaded.",  # transparent_status
                None,                    # upscaled_image_state (reset)
                None,                    # resized_image_state (reset)
                None,                    # mask_image_state (reset)
                None,                    # transparent_image_state (reset)
                gr.Tabs(selected="tab_original")  # Reset to Original tab
            )
        return (None, None, "Select an image from the library.", None, "Select an image first.",
                None, "Select an image first.", None, "Select an image first.", None, None, None, None,
                gr.Tabs(selected="tab_original"))

    def upscale_action(image_path, model_name, target_resolution):
        """Upscale the image with smart resize for training.

        Workflow: Upscale with high-quality model, then downscale to target
        resolution on shortest side. This produces sharp details at practical sizes.
        """
        if not image_path:
            return None, None, "No image selected."
        if not model_name or model_name == "No models found":
            return None, None, "No upscaler model selected. Place .pth files in .models/"
        try:
            from src.core.upscaling import resize_to_shortest_side

            upscaler = get_upscaler(model_name)
            # Upscale first
            upscaled = upscaler.upscale(image_path)
            # Then downscale to target shortest side
            result = resize_to_shortest_side(upscaled, int(target_resolution))
            width, height = result.size
            return result, result, f"Upscaled {upscaler.scale}x → resized to {width}x{height}px. Click 'Save' to save as JPG."
        except Exception as e:
            return None, None, f"Upscale Error: {e}"

    def generate_mask_action(image_path, upscaled_img, resized_img, invert_mask):
        """Generate mask from processed image (upscaled/resized) if available, otherwise from original."""
        if not image_path:
            return None, None, "No image selected."

        try:
            seg = get_segmenter()

            # Priority: upscaled > resized > original
            processed_img = upscaled_img if upscaled_img is not None else resized_img

            if processed_img is not None:
                # Save processed image to temp file for segmenter
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    if isinstance(processed_img, Image.Image):
                        processed_img.save(tmp.name)
                    elif isinstance(processed_img, np.ndarray):
                        Image.fromarray(processed_img).save(tmp.name)
                    temp_path = tmp.name

                mask = seg.segment(temp_path)
                os.unlink(temp_path)
                source_info = "upscaled image" if upscaled_img is not None else "resized image"
            else:
                mask = seg.segment(image_path)
                source_info = "original image"

            if invert_mask and mask:
                # Invert the mask (white becomes black, black becomes white)
                mask = ImageOps.invert(mask.convert("L")).convert("RGB")

            return mask, mask, f"Mask generated from {source_info}. Click 'Save Mask' to save."
        except Exception as e:
            return None, None, f"Mask Error: {e}"

    def generate_transparent_action(image_path, upscaled_img, resized_img, alpha_threshold):
        """Generate transparent image from processed image (upscaled/resized) if available, otherwise from original."""
        if not image_path:
            return None, None, "No image selected."

        try:
            seg = get_segmenter()

            # Priority: upscaled > resized > original
            processed_img = upscaled_img if upscaled_img is not None else resized_img

            if processed_img is not None:
                # Save processed image to temp file for segmenter
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    if isinstance(processed_img, Image.Image):
                        processed_img.save(tmp.name)
                    elif isinstance(processed_img, np.ndarray):
                        Image.fromarray(processed_img).save(tmp.name)
                    temp_path = tmp.name

                transparent = seg.segment(temp_path, return_transparent=True)
                os.unlink(temp_path)
                source_info = "upscaled image" if upscaled_img is not None else "resized image"
            else:
                transparent = seg.segment(image_path, return_transparent=True)
                source_info = "original image"

            return transparent, transparent, f"Transparent generated from {source_info}. Click 'Save Transparent' to save."
        except Exception as e:
            return None, None, f"Transparent Error: {e}"

    def save_original_action(image_path, original_img):
        """Save the original image to the output directory."""
        if not image_path:
            return "No image selected."
        if original_img is None:
            return "No image to save."

        try:
            ext = os.path.splitext(image_path)[1]
            output_path = global_state.get_output_path(image_path, ext)

            if isinstance(original_img, Image.Image):
                original_img.save(output_path)
            elif isinstance(original_img, np.ndarray):
                Image.fromarray(original_img).save(output_path)
            else:
                import shutil
                shutil.copy2(image_path, output_path)

            return f"Saved original to {os.path.basename(output_path)}"
        except Exception as e:
            return f"Save Error: {e}"

    def resize_action(image_path, target_short_side):
        """Resize image to target shortest side (preview only, doesn't save)."""
        if not image_path:
            return None, None, "No image selected."

        try:
            from src.core.upscaling import resize_to_shortest_side

            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size

            result = resize_to_shortest_side(img, int(target_short_side))
            new_w, new_h = result.size

            if result.size == img.size:
                return result, result, f"Image already ≤{int(target_short_side)}px ({orig_w}x{orig_h}). No resize needed."
            return result, result, f"Resized: {orig_w}x{orig_h} → {new_w}x{new_h}. Click 'Save Resized' to save."
        except Exception as e:
            return None, None, f"Resize Error: {e}"

    def save_resized_action(image_path, resized_img):
        """Save the resized image as JPG."""
        if not image_path:
            return "No image selected."
        if resized_img is None:
            return "No resized image. Click 'Resize' first."

        try:
            if isinstance(resized_img, np.ndarray):
                resized_img = Image.fromarray(resized_img)

            if resized_img.mode != "RGB":
                resized_img = resized_img.convert("RGB")

            output_path = global_state.get_output_path(image_path, ".jpg")
            resized_img.save(output_path, "JPEG", quality=98, optimize=True)
            global_state.upscaled[image_path] = output_path

            file_size = os.path.getsize(output_path) / 1024
            w, h = resized_img.size
            return f"Saved {os.path.basename(output_path)} ({w}x{h}, {file_size:.0f} KB)"
        except Exception as e:
            return f"Save Error: {e}"

    def save_upscale_action(image_path, upscaled_img):
        """Save the upscaled image as high-quality JPG for training efficiency."""
        if not image_path:
            return "No image selected."
        if upscaled_img is None:
            return "No upscaled image. Run upscale first."

        try:
            # Always save as JPG for training - much smaller files, no quality loss at 98%
            output_path = global_state.get_output_path(image_path, ".jpg")

            if isinstance(upscaled_img, np.ndarray):
                upscaled_img = Image.fromarray(upscaled_img)

            # Ensure RGB mode for JPG (no alpha channel)
            if upscaled_img.mode != "RGB":
                upscaled_img = upscaled_img.convert("RGB")

            upscaled_img.save(output_path, "JPEG", quality=98, optimize=True)

            global_state.upscaled[image_path] = output_path
            file_size = os.path.getsize(output_path) / 1024  # KB
            return f"Saved {os.path.basename(output_path)} ({file_size:.0f} KB)"
        except Exception as e:
            return f"Save Error: {e}"

    def save_mask_action(image_path, mask_img):
        """Save the mask as grayscale PNG to the masks/ subdirectory."""
        if not image_path:
            return "No image selected."
        if mask_img is None:
            return "No mask. Create mask first."

        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            masks_dir = os.path.join(base_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            save_path = os.path.join(masks_dir, base_name + "_mask.png")

            if isinstance(mask_img, np.ndarray):
                mask_img = Image.fromarray(mask_img)

            # Convert to grayscale "L" mode for smaller file size
            if mask_img.mode != "L":
                mask_img = mask_img.convert("L")

            mask_img.save(save_path, "PNG", optimize=True)

            global_state.masks[image_path] = save_path
            file_size = os.path.getsize(save_path) / 1024
            return f"Saved {base_name}_mask.png ({file_size:.0f} KB)"
        except Exception as e:
            return f"Save Error: {e}"

    def save_transparent_action(image_path, transparent_img):
        """Save the transparent image as WebP lossless to the output directory."""
        if not image_path:
            return "No image selected."
        if transparent_img is None:
            return "No transparent image. Create transparent first."

        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            save_path = os.path.join(base_dir, base_name + "_transparent.webp")

            if isinstance(transparent_img, np.ndarray):
                transparent_img = Image.fromarray(transparent_img)

            # Ensure RGBA mode for transparency
            if transparent_img.mode != "RGBA":
                transparent_img = transparent_img.convert("RGBA")

            # Save as WebP lossless for best quality + compression
            transparent_img.save(save_path, "WEBP", lossless=True)

            global_state.transparent[image_path] = save_path
            file_size = os.path.getsize(save_path) / 1024
            return f"Saved {base_name}_transparent.webp ({file_size:.0f} KB)"
        except Exception as e:
            return f"Save Error: {e}"


    def on_unload_all_models():
        """Unload all models (segmenter and upscaler) to free VRAM."""
        try:
            seg = get_segmenter()
            seg.unload_model()
        except:
            pass
        try:
            upscaler = get_upscaler()
            upscaler.unload_model()
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "All models unloaded from VRAM."

    def run_bulk_masks_action(invert_mask, progress=gr.Progress()):
        """Generate grayscale PNG masks for all images."""
        if not global_state.image_paths:
            return "No images loaded."

        seg = get_segmenter()
        total = len(global_state.image_paths)
        success = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Generating masks"):
            try:
                mask = seg.segment(img_path)
                if mask:
                    # Convert to grayscale for smaller file size
                    if mask.mode != "L":
                        mask = mask.convert("L")

                    # Invert if requested
                    if invert_mask:
                        mask = ImageOps.invert(mask)

                    # Save mask to masks/ subdirectory
                    dummy_path = global_state.get_output_path(img_path, ".txt")
                    base_dir = os.path.dirname(dummy_path)
                    masks_dir = os.path.join(base_dir, "masks")
                    os.makedirs(masks_dir, exist_ok=True)
                    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
                    save_path = os.path.join(masks_dir, base_name + "_mask.png")
                    mask.save(save_path, "PNG", optimize=True)
                    global_state.masks[img_path] = save_path
                    success += 1
            except Exception as e:
                print(f"Mask error for {img_path}: {e}")

        return f"Generated {success}/{total} masks (grayscale PNG)."

    def run_bulk_transparent_action(alpha_threshold, progress=gr.Progress()):
        """Generate WebP lossless transparent images for all images."""
        if not global_state.image_paths:
            return "No images loaded."

        seg = get_segmenter()
        total = len(global_state.image_paths)
        success = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Generating transparent"):
            try:
                img = seg.segment(img_path, return_transparent=True)
                if img:
                    # Ensure RGBA mode for transparency
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")

                    # Apply alpha threshold if not default
                    if alpha_threshold != 128:
                        # Get alpha channel and apply threshold
                        r, g, b, a = img.split()
                        a = a.point(lambda x: 255 if x > alpha_threshold else 0)
                        img = Image.merge("RGBA", (r, g, b, a))

                    # Save as WebP lossless to flat output folder
                    dummy_path = global_state.get_output_path(img_path, ".txt")
                    base_dir = os.path.dirname(dummy_path)
                    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
                    save_path = os.path.join(base_dir, base_name + "_transparent.webp")
                    img.save(save_path, "WEBP", lossless=True)
                    global_state.transparent[img_path] = save_path
                    success += 1
            except Exception as e:
                print(f"Transparent error for {img_path}: {e}")

        return f"Generated {success}/{total} transparent images."

    def run_bulk_process(model_name, target_short_side, upscale_threshold, passthrough_max, progress=gr.Progress()):
        """Process all images with conditional routing.

        Routes images based on shortest side dimension:
        - < upscale_threshold: Upscale with Real-ESRGAN → Lanczos resize to target
        - upscale_threshold to passthrough_max: Copy as JPG 98% (no resize)
        - > passthrough_max: Lanczos downscale to target

        Target is based on SHORTEST side, not longest.
        """
        if not global_state.image_paths:
            return "No images loaded."

        total = len(global_state.image_paths)
        stats = {"upscaled": 0, "passthrough": 0, "resized": 0}

        upscaler = None
        if model_name and model_name != "No models found":
            try:
                upscaler = get_upscaler(model_name)
            except Exception as e:
                return f"Error loading upscaler: {e}"

        for img_path in progress.tqdm(global_state.image_paths, desc="Processing images"):
            try:
                # Always save as JPG for training efficiency
                output_path = global_state.get_output_path(img_path, ".jpg")

                # Check if already processed
                if img_path in global_state.upscaled:
                    continue

                result, action = process_image(
                    img_path,
                    upscaler,
                    target_short_side=int(target_short_side),
                    upscale_threshold=int(upscale_threshold),
                    passthrough_max=int(passthrough_max)
                )

                if result.mode != "RGB":
                    result = result.convert("RGB")
                result.save(output_path, "JPEG", quality=98, optimize=True)
                global_state.upscaled[img_path] = output_path

                stats[action.value] += 1

            except Exception as e:
                print(f"Image processing error for {img_path}: {e}")

        return f"Done: {stats['upscaled']} upscaled, {stats['passthrough']} passthrough, {stats['resized']} resized. All saved as JPG 98%."

    def run_bulk_copy(progress=gr.Progress()):
        """Copy all source images to output folder unchanged.

        Simply copies each source image to the output directory without any
        processing, preserving the original format and quality.
        """
        if not global_state.image_paths:
            return "No images loaded."

        import shutil
        total = len(global_state.image_paths)
        success = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Copying images"):
            try:
                # Skip if already copied
                if img_path in global_state.upscaled:
                    continue

                # Get original extension and copy
                ext = os.path.splitext(img_path)[1]
                output_path = global_state.get_output_path(img_path, ext)
                shutil.copy2(img_path, output_path)
                global_state.upscaled[img_path] = output_path
                success += 1
            except Exception as e:
                print(f"Copy error for {img_path}: {e}")

        return f"Copied {success}/{total} images to output folder."

    def refresh_upscaler_models():
        """Refresh the list of available upscaler models."""
        models = get_available_models()
        if not models:
            return gr.update(choices=["No models found"], value=None)
        return gr.update(choices=models, value=models[0])

    def get_output_images():
        """Scan output directory for images (excluding masks/ subdirectory, and transparent only if base exists)."""
        if not global_state.output_directory or not os.path.isdir(global_state.output_directory):
            return [], {}

        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        all_files = set()
        images = []
        captions = {}

        # First pass: collect all image filenames (skip masks/ subdirectory)
        for root, dirs, files in os.walk(global_state.output_directory):
            # Skip masks subdirectory
            if 'masks' in dirs:
                dirs.remove('masks')
            for file in files:
                if file.lower().endswith(valid_extensions):
                    all_files.add(os.path.join(root, file))

        # Second pass: filter images
        for img_path in all_files:
            file = os.path.basename(img_path)
            file_lower = file.lower()

            # Always skip mask files
            if '_mask.' in file_lower:
                continue

            # For transparent files, only include if no base image exists
            if '_transparent.' in file_lower:
                # Check if base image exists (try common extensions)
                base_stem = file.rsplit('_transparent.', 1)[0]
                dir_path = os.path.dirname(img_path)
                base_exists = any(
                    os.path.join(dir_path, base_stem + ext) in all_files
                    for ext in ['.jpg', '.jpeg', '.png', '.webp']
                )
                if base_exists:
                    continue  # Skip - base image exists

            images.append(img_path)

            # Try to load existing caption
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        captions[img_path] = f.read().strip()
                except:
                    captions[img_path] = ""
            else:
                captions[img_path] = ""

        images.sort()
        return images, captions

    # State for output images (used in Step 3)
    _output_images = []
    _output_captions = {}
    _displayed_images = []  # Currently displayed images (may be filtered)

    def refresh_output_gallery():
        """Refresh the output images gallery for Step 3."""
        nonlocal _output_images, _output_captions, _displayed_images
        _output_images, _output_captions = get_output_images()
        _displayed_images = _output_images.copy()
        if not _output_images:
            return [], "No images found in output directory. Process images in Step 2 first."
        return _output_images, f"Found {len(_output_images)} images in output directory."

    def on_select_output_image(evt: gr.SelectData):
        """Handle gallery selection in Step 3 (output images)."""
        index = evt.index
        if index < len(_displayed_images):
            path = _displayed_images[index]
            caption = _output_captions.get(path, "")
            # Load the image for preview
            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)
            except Exception:
                img = None
            # Find the index in the full list for Save & Next navigation
            full_index = _output_images.index(path) if path in _output_images else -1
            return path, img, caption, full_index
        return None, None, "", -1

    def save_output_caption_action(path, new_caption):
        """Save caption for an output image."""
        if not path:
            return "Error: No image selected"
        _output_captions[path] = new_caption
        txt_path = os.path.splitext(path)[0] + ".txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(new_caption)
            return f"Saved caption for {os.path.basename(path)}"
        except Exception as e:
            return f"Error: {e}"

    def save_and_next_action(path, new_caption, current_index):
        """Save caption and move to next image."""
        if not path:
            return "Error: No image selected", None, None, "", -1

        # Save current caption
        _output_captions[path] = new_caption
        txt_path = os.path.splitext(path)[0] + ".txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(new_caption)
        except Exception as e:
            return f"Error saving: {e}", None, None, "", current_index

        # Move to next image
        next_index = current_index + 1
        if next_index >= len(_output_images):
            return f"Saved. Reached end of gallery ({len(_output_images)} images).", None, None, "", -1

        next_path = _output_images[next_index]
        next_caption = _output_captions.get(next_path, "")
        try:
            next_img = Image.open(next_path)
            next_img = ImageOps.exif_transpose(next_img)
        except Exception:
            next_img = None

        return f"Saved. Now editing: {os.path.basename(next_path)}", next_path, next_img, next_caption, next_index

    # State for undo functionality
    _undo_caption = {"path": None, "caption": None}

    def fix_format_action(caption):
        """Fix caption formatting: normalize commas, spacing, remove empty tags."""
        if not caption:
            return caption, "Nothing to fix."

        # Store for undo
        original = caption

        # Split by comma, clean each tag, rejoin
        tags = [t.strip() for t in caption.split(",")]
        # Remove empty tags and duplicates while preserving order
        seen = set()
        cleaned = []
        for tag in tags:
            if tag and tag not in seen:
                cleaned.append(tag)
                seen.add(tag)

        result = ", ".join(cleaned)

        if result == original:
            return result, "Already formatted correctly."
        return result, f"Fixed formatting ({len(tags)} → {len(cleaned)} tags)."

    def dedup_tags_action(caption):
        """Remove duplicate tags while preserving order."""
        if not caption:
            return caption, "Nothing to deduplicate."

        original = caption
        tags = [t.strip() for t in caption.split(",")]
        seen = set()
        unique = []
        duplicates = 0
        for tag in tags:
            tag_lower = tag.lower()
            if tag and tag_lower not in seen:
                unique.append(tag)
                seen.add(tag_lower)
            elif tag:
                duplicates += 1

        result = ", ".join(unique)

        if duplicates == 0:
            return result, "No duplicates found."
        return result, f"Removed {duplicates} duplicate(s)."

    def store_undo_state(path, caption):
        """Store current state for undo."""
        _undo_caption["path"] = path
        _undo_caption["caption"] = caption

    def undo_changes_action(current_path, current_caption):
        """Restore the previous caption state."""
        if _undo_caption["path"] == current_path and _undo_caption["caption"] is not None:
            restored = _undo_caption["caption"]
            _undo_caption["caption"] = None  # Clear after use
            return restored, "Restored previous caption."
        return current_caption, "Nothing to undo."

    def filter_gallery_action(search_text):
        """Filter gallery images by caption content."""
        nonlocal _displayed_images
        if not search_text or not search_text.strip():
            # Return all images
            _displayed_images = _output_images.copy()
            return _output_images

        search_lower = search_text.lower().strip()
        filtered = []
        for path in _output_images:
            caption = _output_captions.get(path, "").lower()
            if search_lower in caption:
                filtered.append(path)
        _displayed_images = filtered
        return filtered

    def bulk_add_tag_to_all(tag_text, position):
        """Add tags to all captions (append or prepend)."""
        if not tag_text or not tag_text.strip():
            return "Error: Tag text empty."
        tags = tag_text.strip().strip(",").strip()
        prepend = "Prepend" in position
        count = 0
        for path in _output_images:
            cap = _output_captions.get(path, "")
            if cap:
                if prepend:
                    new_cap = f"{tags}, {cap}"
                else:
                    new_cap = f"{cap}, {tags}"
            else:
                new_cap = tags
            _output_captions[path] = new_cap
            try:
                txt_path = os.path.splitext(path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_cap)
                count += 1
            except:
                pass
        action = "Prepended" if prepend else "Appended"
        return f"{action} '{tags}' to {count} captions."

    def bulk_remove_tag_from_all(tag_text):
        """Remove multiple tags from all captions."""
        if not tag_text or not tag_text.strip():
            return "Error: Tag text empty."
        # Split by comma to get list of tags to remove
        tags_to_remove = [t.strip() for t in tag_text.split(",") if t.strip()]
        if not tags_to_remove:
            return "Error: No valid tags to remove."

        count = 0
        for path in _output_images:
            cap = _output_captions.get(path, "")
            original_cap = cap
            for tag in tags_to_remove:
                if tag in cap:
                    # Try to remove with comma variations
                    cap = cap.replace(f", {tag}", "")
                    cap = cap.replace(f"{tag}, ", "")
                    cap = cap.replace(tag, "")
            # Clean up any double commas or trailing/leading commas
            while ",," in cap:
                cap = cap.replace(",,", ",")
            cap = cap.strip().strip(",").strip()

            if cap != original_cap:
                _output_captions[path] = cap
                try:
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(cap)
                    count += 1
                except:
                    pass
        tags_str = ", ".join(tags_to_remove[:3])
        if len(tags_to_remove) > 3:
            tags_str += f"... (+{len(tags_to_remove) - 3} more)"
        return f"Removed [{tags_str}] from {count} captions."

    def bulk_search_replace_action(replacement_string, match_mode):
        """Apply multiple search & replace operations.

        Format: 'old_tag, new_tag; another_old, another_new'
        """
        if not replacement_string or not replacement_string.strip():
            return "Error: Replacement string empty."

        exact_match = match_mode == "Exact Match"

        # Parse semicolon-separated pairs
        pairs = []
        for pair in replacement_string.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split(",", 1)  # Split on first comma only
            if len(parts) != 2:
                continue
            old_tag = parts[0].strip()
            new_tag = parts[1].strip()
            if old_tag:
                pairs.append((old_tag, new_tag))

        if not pairs:
            return "Error: No valid replacement pairs found. Use format: old, new; old2, new2"

        count = 0
        for path in _output_images:
            cap = _output_captions.get(path, "")
            original_cap = cap

            for old_tag, new_tag in pairs:
                if exact_match:
                    # Exact match: only replace whole tags
                    # Split by comma, replace matching tags, rejoin
                    cap_tags = [t.strip() for t in cap.split(",")]
                    cap_tags = [new_tag if t == old_tag else t for t in cap_tags]
                    cap = ", ".join(t for t in cap_tags if t)
                else:
                    # Substring match
                    cap = cap.replace(old_tag, new_tag)

            # Clean up
            while ",," in cap:
                cap = cap.replace(",,", ",")
            cap = cap.strip().strip(",").strip()

            if cap != original_cap:
                _output_captions[path] = cap
                try:
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(cap)
                    count += 1
                except:
                    pass

        pairs_str = "; ".join([f"{o} → {n}" for o, n in pairs[:2]])
        if len(pairs) > 2:
            pairs_str += f"... (+{len(pairs) - 2} more)"
        return f"Applied [{pairs_str}] to {count} captions."

    # Danbooru rating tags to filter (with "rating:" prefix)
    RATING_TAGS_PREFIXED = {"rating:general", "rating:sensitive", "rating:questionable", "rating:explicit"}
    # Standalone rating tags (only filtered at start/end of caption)
    RATING_TAGS_STANDALONE = {"general", "sensitive", "questionable", "explicit"}

    def filter_rating_tags(tags):
        """Filter rating tags from a list of tags.

        - Always removes 'rating:*' prefixed tags anywhere
        - Removes standalone rating words only from first/last position
        """
        if not tags:
            return tags

        # First pass: remove all prefixed rating tags
        tags = [t for t in tags if t.lower() not in RATING_TAGS_PREFIXED]

        if not tags:
            return tags

        # Second pass: remove standalone rating tags from first/last position
        while tags and tags[0].lower() in RATING_TAGS_STANDALONE:
            tags = tags[1:]
        while tags and tags[-1].lower() in RATING_TAGS_STANDALONE:
            tags = tags[:-1]

        return tags

    def run_output_captioning_action(model_name, threshold, prefix_tags, suffix_tags, filter_ratings, progress=gr.Progress()):
        """Generate captions for all output images."""
        if not _output_images:
            return "No images in output directory."
        captioner = None
        total = len(_output_images)
        success_count = 0
        prefix = prefix_tags.strip().rstrip(",").strip() if prefix_tags else ""
        suffix = suffix_tags.strip().lstrip(",").strip() if suffix_tags else ""

        try:
            captioner = get_captioner(model_name)
            for img_path in progress.tqdm(_output_images):
                try:
                    # Pass threshold for WD14 models
                    if "WD" in model_name:
                        cap = captioner.generate_caption(img_path, threshold=threshold)
                    else:
                        cap = captioner.generate_caption(img_path)

                    # Filter out rating tags if enabled
                    if filter_ratings and cap:
                        tags = [t.strip() for t in cap.split(",") if t.strip()]
                        tags = filter_rating_tags(tags)
                        cap = ", ".join(tags)

                    # Build final caption with prefix and suffix
                    parts = []
                    if prefix:
                        parts.append(prefix)
                    if cap:
                        parts.append(cap)
                    if suffix:
                        parts.append(suffix)
                    cap = ", ".join(parts)

                    _output_captions[img_path] = cap
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(cap)
                    success_count += 1
                except Exception as e:
                    print(f"Captioning error for {img_path}: {e}")
        except Exception as e:
            return f"Model Error: {e}"
        finally:
            if captioner:
                captioner.unload_model()
        return f"Completed. {success_count}/{total} images captioned."

    # --- UI LAYOUT ---
    
    with gr.Column() as wizard_container:
        with gr.Tabs(elem_id="wizard_tabs", selected=0) as tabs:
            
            # STEP 1
            with gr.TabItem("Step 1: Import", id=0):
                gr.Markdown("## Step 1: Select Dataset Directories")
                gr.Markdown("Define where your source images are located and where you want generated files (captions, masks) to be saved.")
                
                with gr.Group():
                    with gr.Row():
                        input_dir = gr.Textbox(label="Source Image Directory", placeholder="Path to source images", value="datasets/input", scale=4)
                        browse_input_btn = gr.Button("📂 Browse", scale=1)
                    
                    with gr.Row():
                        output_dir = gr.Textbox(
                            label="Base Output Directory", 
                            placeholder="Path to save results", 
                            value="datasets/output", 
                            scale=4,
                            info="If this folder does not exist, it will be created automatically when you click Scan."
                        )
                        browse_output_btn = gr.Button("📂 Browse", scale=1)
                    
                    with gr.Row():
                        project_name = gr.Textbox(
                            label="Project Name / Subfolder (Optional)", 
                            placeholder="e.g. my_new_dataset", 
                            info="Appends to the Base Output Directory. Useful for organizing different experiments."
                        )

                scan_btn = gr.Button("Scan & Initialize", variant="primary")
                scan_output = gr.Textbox(label="Status", interactive=False)
                next_btn_1 = gr.Button("Next >", interactive=False)

            # STEP 2: Image Tools (moved before captioning)
            with gr.TabItem("Step 2: Image Tools", id=1) as tab_step_2:
                gr.Markdown("## Step 2: Image Tools")

                # Hidden states
                selected_path_state = gr.State()
                upscaled_image_state = gr.State()
                resized_image_state = gr.State()
                mask_image_state = gr.State()
                transparent_image_state = gr.State()

                with gr.Row():
                    # COLUMN 1: THE LIBRARY (35%)
                    with gr.Column(scale=35):
                        step2_gallery = gr.Gallery(
                            label="Image Library",
                            columns=4,
                            height=750,
                            allow_preview=False,
                            show_label=False
                        )

                    # COLUMN 2: THE WORKBENCH (65%)
                    with gr.Column(scale=65):
                        # Tabs with context-sensitive toolbars
                        with gr.Tabs() as workbench_tabs:
                            # TAB 1: ORIGINAL
                            with gr.TabItem("Original", id="tab_original"):
                                workbench_original = gr.Image(
                                    label="Original Image",
                                    type="pil",
                                    height=500,
                                    interactive=False
                                )
                                original_status = gr.Textbox(
                                    label="Status",
                                    value="Select an image from the library.",
                                    interactive=False
                                )
                                resize_target_slider = gr.Slider(
                                    label="Resize Target (shortest side, px)",
                                    minimum=1024,
                                    maximum=4096,
                                    value=2048,
                                    step=64,
                                    info="For downsizing large images. Use Upscale tab for small images."
                                )
                                with gr.Row():
                                    save_original_btn = gr.Button("Save Original", variant="primary")
                                    resize_btn = gr.Button("Resize", variant="secondary")
                                    save_resized_btn = gr.Button("Save Resized", variant="primary")

                            # TAB 2: UPSCALE
                            with gr.TabItem("Upscale", id="tab_upscale"):
                                workbench_upscaled = gr.Image(
                                    label="Upscaled Image",
                                    type="pil",
                                    height=500,
                                    interactive=False
                                )
                                upscale_status = gr.Textbox(
                                    label="Status",
                                    value="Select an image, then run upscale.",
                                    interactive=False
                                )
                                with gr.Row():
                                    upscaler_model = gr.Dropdown(
                                        label="Model",
                                        choices=get_available_models() or ["No models found"],
                                        value=get_available_models()[0] if get_available_models() else None,
                                        scale=1
                                    )
                                    with gr.Column(scale=1):
                                        refresh_models_btn = gr.Button("↻ Refresh Model List")
                                        unload_btn = gr.Button("⏏ Unload All Models", variant="secondary")
                                target_res_slider = gr.Slider(
                                    label="Target Resolution (shortest side)",
                                    minimum=1024,
                                    maximum=4096,
                                    value=2048,
                                    step=64,
                                    info="Upscales then resizes shortest side to this."
                                )
                                with gr.Row():
                                    run_upscale_btn = gr.Button("Run Upscale", variant="secondary")
                                    save_upscale_btn = gr.Button("Save Upscale", variant="primary")

                            # TAB 3: MASK
                            with gr.TabItem("Mask", id="tab_mask"):
                                workbench_mask = gr.Image(
                                    label="Mask Preview",
                                    type="pil",
                                    height=500,
                                    interactive=False
                                )
                                mask_status = gr.Textbox(
                                    label="Status",
                                    value="Select an image, then create mask.",
                                    interactive=False
                                )
                                with gr.Row():
                                    invert_mask_check = gr.Checkbox(
                                        label="Invert Mask",
                                        value=False
                                    )
                                with gr.Row():
                                    create_mask_btn = gr.Button("Create Mask", variant="secondary")
                                    save_mask_btn = gr.Button("Save Mask", variant="primary")

                            # TAB 4: TRANSPARENT
                            with gr.TabItem("Transparent", id="tab_transparent"):
                                workbench_transparent = gr.Image(
                                    label="Transparent Preview",
                                    type="pil",
                                    height=500,
                                    interactive=False
                                )
                                transparent_status = gr.Textbox(
                                    label="Status",
                                    value="Select an image, then create transparent.",
                                    interactive=False
                                )
                                with gr.Row():
                                    alpha_threshold_slider = gr.Slider(
                                        label="Alpha Threshold",
                                        minimum=0,
                                        maximum=255,
                                        value=128,
                                        step=1
                                    )
                                with gr.Row():
                                    create_transparent_btn = gr.Button("Create Transparent", variant="secondary")
                                    save_transparent_btn = gr.Button("Save Transparent", variant="primary")

                # Bulk Actions Accordion (at bottom, closed by default)
                with gr.Accordion("Bulk Actions", open=False):

                    with gr.Row():
                        # Left column: Smart Processing
                        with gr.Group():
                            gr.Markdown("### Smart Processing (Resize / Upscale)")
                            gr.Markdown("""Routes images by shortest side:
- **Passthrough** (between thresholds, saved as 98% JPG)
- **Upscale** (below lower, Real-ESRGAN)
- **Downscale** (above upper, Lanczos)""")
                            bulk_passthrough_max_slider = gr.Slider(
                                label="Passthrough Upper Limit (px)",
                                minimum=1500,
                                maximum=4096,
                                value=2500,
                                step=64,
                                info="Images with shortest side ABOVE this will be downscaled to target"
                            )
                            bulk_upscale_threshold_slider = gr.Slider(
                                label="Upscale Lower Limit (px)",
                                minimum=512,
                                maximum=2048,
                                value=1500,
                                step=64,
                                info="Images with shortest side BELOW this will be upscaled then resized to target"
                            )
                            bulk_target_short_side_slider = gr.Slider(
                                label="Resize Target (shortest side, px)",
                                minimum=1024,
                                maximum=4096,
                                value=2048,
                                step=64,
                                info="Target size for upscaled or downscaled images (passthrough images are unchanged)"
                            )
                            bulk_process_btn = gr.Button("Run Smart Process All", variant="primary")

                        # Right column: Mask, Background Removal, Bypass
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("### Mask Generation")
                                gr.Markdown("Create grayscale segmentation masks for all output images (saved to `masks/` subfolder).")
                                bulk_invert_mask_check = gr.Checkbox(label="Invert Masks", value=False)
                                bulk_masks_btn = gr.Button("Generate Masks for All", variant="primary")

                            with gr.Group():
                                gr.Markdown("### Background Removal")
                                gr.Markdown("Remove backgrounds and save as lossless WebP with transparency.")
                                bulk_alpha_threshold_slider = gr.Slider(
                                    label="Alpha Threshold",
                                    minimum=0,
                                    maximum=255,
                                    value=128,
                                    step=1,
                                    info="Threshold for alpha channel cutoff"
                                )
                                bulk_transparent_btn = gr.Button("Remove Backgrounds for All", variant="primary")

                            with gr.Group():
                                gr.Markdown("### Bypass Editing for All Images")
                                gr.Markdown("Skip all image processing and copy source images directly to the output folder unchanged for captioning.")
                                bulk_copy_btn = gr.Button("Copy All Source Images to Output", variant="primary")

                    bulk_status = gr.Textbox(label="Status", value="Ready...", interactive=False)

                with gr.Row():
                    back_btn_2 = gr.Button("< Back")
                    next_btn_2 = gr.Button("Next >")

            # STEP 3: Captioning (moved after image tools)
            with gr.TabItem("Step 3: Captioning", id=2) as tab_step_3:
                gr.Markdown("## Step 3: Captioning")

                # Hidden states
                current_path_state = gr.State()
                current_index_state = gr.State(-1)
                undo_caption_state = gr.State(None)

                # PHASE 1: Batch Generation (Accordion, Open by Default)
                with gr.Accordion("Batch Generation", open=True):
                    with gr.Row():
                        # Left column: Model selector (50%)
                        with gr.Column(scale=1):
                            model_dropdown = gr.Dropdown(
                                [
                                    "BLIP-Base",
                                    "BLIP-Large",
                                    "Florence-2-Base",
                                    "Florence-2-Large",
                                    "JoyCaption (Beta One)",
                                    "JoyCaption Quantized (8-bit)",
                                    "SmilingWolf WD ConvNext (v3)",
                                    "SmilingWolf WD ViT (v3)",
                                ],
                                label="Model",
                                value="SmilingWolf WD ConvNext (v3)"
                            )
                            with gr.Accordion("🖥️ VRAM Requirements", open=False):
                                gr.Markdown("""
- **SmilingWolf WD14 (ViT/ConvNext):** ~2GB VRAM. Fast Danbooru-style tagging using ONNX runtime. Best for anime/illustration.
- **BLIP-Base:** ~2GB VRAM. Natural language captions, good general purpose.
- **BLIP-Large:** ~4GB VRAM. More detailed natural language captions.
- **Florence-2-Base:** ~4GB VRAM. Microsoft's vision model, detailed scene descriptions.
- **Florence-2-Large:** ~4GB VRAM. More detailed than base, similar VRAM usage.
- **JoyCaption Quantized (8-bit):** ~12-16GB VRAM. High quality captions, requires 16GB+ GPU.
- **JoyCaption (Beta One):** ~17GB VRAM. Full BF16 precision, requires 20GB+ GPU (RTX 3090/4090).
""")
                        # Right column: Threshold and Filter (50%)
                        with gr.Column(scale=1):
                            threshold_slider = gr.Slider(
                                label="Threshold (WD14 only)",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.35,
                                step=0.05,
                                info="Lower = more tags, higher = fewer but more confident tags"
                            )
                            filter_ratings_check = gr.Checkbox(
                                label="Filter rating tags (general, sensitive, questionable, explicit)",
                                value=True
                            )
                    with gr.Row():
                        prefix_tags_box = gr.Textbox(
                            label="Prefix (Prepend)",
                            placeholder="e.g., sks, 1girl",
                            max_lines=1,
                            scale=1
                        )
                        suffix_tags_box = gr.Textbox(
                            label="Suffix (Append)",
                            placeholder="e.g., best quality, 4k",
                            max_lines=1,
                            scale=1
                        )
                    caption_btn = gr.Button("Generate Tags for All Images", variant="primary")
                    progress_bar = gr.Textbox(value="Idle", interactive=False, show_label=False, max_lines=1)

                # Main area: Library (left) and Editor (right)
                with gr.Row():
                    # LEFT COLUMN: Library (50%)
                    with gr.Column(scale=1):
                        search_filter_box = gr.Textbox(
                            placeholder="Filter images by caption content...",
                            show_label=False,
                            max_lines=1
                        )
                        gallery = gr.Gallery(
                            columns=4,
                            height=700,
                            allow_preview=False,
                            show_label=False
                        )

                    # RIGHT COLUMN: Editor (50%)
                    with gr.Column(scale=1):
                        editor_preview = gr.Image(
                            type="pil",
                            height=350,
                            interactive=False,
                            show_label=False
                        )
                        editor_caption = gr.Textbox(
                            label="Caption / Tags",
                            lines=6,
                            placeholder="Select an image to edit its caption...",
                            autofocus=True
                        )

                        # Hygiene Tools
                        with gr.Row():
                            fix_format_btn = gr.Button("Fix Format")
                            dedup_btn = gr.Button("Dedup Tags")
                            undo_btn = gr.Button("Undo Changes")

                        with gr.Accordion("What do these buttons do?", open=False):
                            gr.Markdown("""- **Fix Format**: Normalize comma spacing and remove empty tags
- **Dedup Tags**: Remove duplicate tags (case-insensitive)
- **Undo Changes**: Revert to caption before last edit""")

                        save_status = gr.Textbox(interactive=False, show_label=False, max_lines=1)

                        # Navigation
                        save_next_btn = gr.Button("Save & Next", variant="primary", size="lg")

                # Bulk Tagging Tools (Accordion, Closed)
                with gr.Accordion("Bulk Tagging Tools", open=False):
                    with gr.Row():
                        # Column 1: Add Tags
                        with gr.Column():
                            gr.Markdown("**Add Tags**")
                            bulk_add_position = gr.Radio(
                                choices=["Prepend", "Append"],
                                value="Prepend",
                                show_label=False
                            )
                            bulk_add_tags_box = gr.Textbox(
                                placeholder="e.g., masterpiece, best quality",
                                show_label=False,
                                max_lines=1
                            )
                            bulk_add_btn = gr.Button("Add to All", variant="primary")

                        # Column 2: Remove Tags
                        with gr.Column():
                            gr.Markdown("**Remove Tags**")
                            bulk_remove_placeholder = gr.Radio(
                                choices=["Comma-separated"],
                                value="Comma-separated",
                                show_label=False,
                                interactive=False
                            )
                            bulk_remove_tags_box = gr.Textbox(
                                placeholder="e.g., lowres, bad anatomy",
                                show_label=False,
                                max_lines=1
                            )
                            bulk_remove_btn = gr.Button("Remove from All", variant="primary")

                        # Column 3: Search & Replace
                        with gr.Column():
                            gr.Markdown("**Search & Replace**")
                            bulk_exact_match = gr.Radio(
                                choices=["Exact Match", "Partial Match"],
                                value="Exact Match",
                                show_label=False
                            )
                            bulk_replace_box = gr.Textbox(
                                placeholder="old, new; red, blue",
                                show_label=False,
                                max_lines=1
                            )
                            bulk_replace_btn = gr.Button("Replace All", variant="primary")

                    caption_bulk_status = gr.Textbox(interactive=False, show_label=False, max_lines=1)

                with gr.Row():
                    back_btn_3 = gr.Button("< Back")
                    next_btn_3 = gr.Button("Next >")

            # STEP 4: Export
            with gr.TabItem("Step 4: Export", id=3) as tab_step_4:
                gr.Markdown("## Step 4: Project Finalized")
                gr.Markdown("Your files have been saved alongside your images.")
                final_status = gr.JSON(label="Current Session Stats")
                refresh_btn = gr.Button("Refresh Stats")
                back_btn_4 = gr.Button("< Back")

    # --- EVENT BINDINGS (Wired at the end to ensure all components exist) ---

    # Step 1: Import
    browse_input_btn.click(browse_directory, inputs=input_dir, outputs=input_dir)
    browse_output_btn.click(browse_directory, inputs=output_dir, outputs=output_dir)
    scan_btn.click(scan_action, inputs=[input_dir, output_dir, project_name], outputs=[scan_output, next_btn_1])
    next_btn_1.click(lambda: gr.Tabs(selected=1), outputs=tabs)

    # Step 2: Image Tools
    tab_step_2.select(lambda: global_state.image_paths, outputs=step2_gallery)

    # Gallery selection - load image into all workbench tabs and reset to Original tab
    step2_gallery.select(
        on_gallery_select,
        outputs=[
            selected_path_state,       # path
            workbench_original,        # original image display
            original_status,           # original tab status
            workbench_upscaled,        # existing upscaled (if any)
            upscale_status,            # upscale tab status
            workbench_mask,            # existing mask (if any)
            mask_status,               # mask tab status
            workbench_transparent,     # existing transparent (if any)
            transparent_status,        # transparent tab status
            upscaled_image_state,      # reset upscaled state
            resized_image_state,       # reset resized state
            mask_image_state,          # reset mask state
            transparent_image_state,   # reset transparent state
            workbench_tabs             # reset to Original tab
        ]
    )

    # TAB 1: Original - Save, Resize, and Save Resized buttons
    save_original_btn.click(
        save_original_action,
        inputs=[selected_path_state, workbench_original],
        outputs=original_status
    )

    resize_btn.click(
        resize_action,
        inputs=[selected_path_state, resize_target_slider],
        outputs=[workbench_original, resized_image_state, original_status]
    )

    save_resized_btn.click(
        save_resized_action,
        inputs=[selected_path_state, resized_image_state],
        outputs=original_status
    )

    # TAB 2: Upscale - Run and Save buttons
    run_upscale_btn.click(
        upscale_action,
        inputs=[selected_path_state, upscaler_model, target_res_slider],
        outputs=[workbench_upscaled, upscaled_image_state, upscale_status]
    )

    save_upscale_btn.click(
        save_upscale_action,
        inputs=[selected_path_state, workbench_upscaled],
        outputs=upscale_status
    )

    refresh_models_btn.click(refresh_upscaler_models, outputs=upscaler_model)

    # TAB 3: Mask - Create and Save buttons
    create_mask_btn.click(
        generate_mask_action,
        inputs=[selected_path_state, workbench_upscaled, resized_image_state, invert_mask_check],
        outputs=[workbench_mask, mask_image_state, mask_status]
    )

    save_mask_btn.click(
        save_mask_action,
        inputs=[selected_path_state, workbench_mask],
        outputs=mask_status
    )

    # TAB 4: Transparent - Create and Save buttons
    create_transparent_btn.click(
        generate_transparent_action,
        inputs=[selected_path_state, workbench_upscaled, resized_image_state, alpha_threshold_slider],
        outputs=[workbench_transparent, transparent_image_state, transparent_status]
    )

    save_transparent_btn.click(
        save_transparent_action,
        inputs=[selected_path_state, workbench_transparent],
        outputs=transparent_status
    )

    # Bulk operations
    bulk_copy_btn.click(run_bulk_copy, outputs=bulk_status)
    bulk_masks_btn.click(run_bulk_masks_action, inputs=bulk_invert_mask_check, outputs=bulk_status)
    bulk_transparent_btn.click(run_bulk_transparent_action, inputs=bulk_alpha_threshold_slider, outputs=bulk_status)
    bulk_process_btn.click(
        run_bulk_process,
        inputs=[upscaler_model, bulk_target_short_side_slider, bulk_upscale_threshold_slider, bulk_passthrough_max_slider],
        outputs=bulk_status
    )

    # Unload models (on Upscale tab)
    unload_btn.click(on_unload_all_models, outputs=upscale_status)

    def go_to_step3():
        """Navigate to Step 3 and refresh output gallery."""
        images, status = refresh_output_gallery()
        return gr.Tabs(selected=2), images, status

    back_btn_2.click(lambda: gr.Tabs(selected=0), outputs=tabs)
    next_btn_2.click(go_to_step3, outputs=[tabs, gallery, save_status])

    # Step 3: Captioning (uses output directory images)
    caption_btn.click(
        run_output_captioning_action,
        inputs=[model_dropdown, threshold_slider, prefix_tags_box, suffix_tags_box, filter_ratings_check],
        outputs=progress_bar
    )

    # When tab is selected, refresh gallery with output images
    tab_step_3.select(refresh_output_gallery, outputs=[gallery, save_status])

    # Search/filter functionality
    search_filter_box.change(filter_gallery_action, inputs=search_filter_box, outputs=gallery)

    # When gallery image is selected, update editor with preview
    gallery.select(
        on_select_output_image,
        outputs=[current_path_state, editor_preview, editor_caption, current_index_state]
    )

    # Hygiene tools
    def fix_format_with_undo(caption, current_path):
        """Fix format and store undo state."""
        store_undo_state(current_path, caption)
        new_caption, status = fix_format_action(caption)
        return new_caption, status

    def dedup_with_undo(caption, current_path):
        """Deduplicate tags and store undo state."""
        store_undo_state(current_path, caption)
        new_caption, status = dedup_tags_action(caption)
        return new_caption, status

    fix_format_btn.click(
        fix_format_with_undo,
        inputs=[editor_caption, current_path_state],
        outputs=[editor_caption, save_status]
    )
    dedup_btn.click(
        dedup_with_undo,
        inputs=[editor_caption, current_path_state],
        outputs=[editor_caption, save_status]
    )
    undo_btn.click(
        undo_changes_action,
        inputs=[current_path_state, editor_caption],
        outputs=[editor_caption, save_status]
    )

    # Save & Next button
    save_next_btn.click(
        save_and_next_action,
        inputs=[current_path_state, editor_caption, current_index_state],
        outputs=[save_status, current_path_state, editor_preview, editor_caption, current_index_state]
    )

    # Bulk tools
    bulk_add_btn.click(
        bulk_add_tag_to_all,
        inputs=[bulk_add_tags_box, bulk_add_position],
        outputs=caption_bulk_status
    )
    bulk_remove_btn.click(
        bulk_remove_tag_from_all,
        inputs=bulk_remove_tags_box,
        outputs=caption_bulk_status
    )
    bulk_replace_btn.click(
        bulk_search_replace_action,
        inputs=[bulk_replace_box, bulk_exact_match],
        outputs=caption_bulk_status
    )

    # Nav buttons (Step 3)
    back_btn_3.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=3), outputs=tabs)

    # Step 4: Export
    def get_stats():
        return {
            "Images Found": len(global_state.image_paths),
            "Captions Loaded/Created": len(global_state.captions),
            "Masks Created": len(global_state.masks),
            "Transparent Images": len(global_state.transparent),
            "Upscaled Images": len(global_state.upscaled)
        }

    tab_step_4.select(get_stats, outputs=final_status)
    refresh_btn.click(get_stats, outputs=final_status)
    back_btn_4.click(lambda: gr.Tabs(selected=2), outputs=tabs)

    return wizard_container