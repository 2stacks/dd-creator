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
    # Ensure default directories exist for FileExplorer components
    os.makedirs("datasets/input", exist_ok=True)
    os.makedirs("datasets/output", exist_ok=True)

    # --- HELPER FUNCTIONS (Logic isolated from UI Layout) ---

    def on_explorer_select(selected):
        """Handle FileExplorer selection - convert to relative path for textbox."""
        if not selected:
            return gr.update()
        # FileExplorer returns a list of selected paths
        if isinstance(selected, list):
            selected = selected[0] if selected else ""
        if not selected:
            return gr.update()

        path = selected
        # If a file was selected, use parent directory
        if os.path.isfile(path):
            path = os.path.dirname(path)

        # Convert absolute path to relative path
        cwd = os.getcwd()
        if os.path.isabs(path) and path.startswith(cwd):
            path = os.path.relpath(path, cwd)

        return path

    def handle_upload(files, local_folder):
        """Handle file upload - copy images to local folder or auto-created staging directory.

        If local_folder is set to something other than the default (datasets/input),
        uploads go directly into that folder. Otherwise a new timestamped folder
        is created under datasets/input/uploads/.
        """
        import shutil
        from datetime import datetime

        if not files:
            return "", "No files uploaded."

        # Determine upload destination
        default_source = "datasets/input"
        if local_folder and local_folder.strip() and local_folder.strip() != default_source:
            upload_dir = local_folder.strip()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_dir = os.path.join("datasets", "input", "uploads", f"upload_{timestamp}")

        os.makedirs(upload_dir, exist_ok=True)

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        copied = 0

        for file_path in files:
            if not file_path:
                continue
            ext = os.path.splitext(file_path)[1].lower()
            if ext in valid_extensions:
                dest = os.path.join(upload_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dest)
                copied += 1

        if copied == 0:
            return "", "No valid images found. Supported: PNG, JPG, JPEG, WebP, BMP"

        return upload_dir, f"Uploaded {copied} images to {upload_dir}"

    def to_relative_path(path):
        """Convert absolute path to relative path if within cwd."""
        if not path:
            return path
        cwd = os.getcwd()
        if os.path.isabs(path) and path.startswith(cwd):
            return os.path.relpath(path, cwd)
        return path

    def scan_action_unified(local_input, upload_staging, project_name, existing_output):
        """Unified scan action - determines source and output from field values.

        Source priority: upload_staging (if populated) > local_input
        Output priority: project_name (creates new) > existing_output
        """
        DEFAULT_SOURCE = "datasets/input"
        DEFAULT_OUTPUT = "datasets/output"

        # --- Determine source path ---
        # Upload staging takes priority, but only if local_input is still the default
        # (if user explicitly set a local path, respect that over old uploads)
        if upload_staging and os.path.isdir(upload_staging) and os.listdir(upload_staging):
            if local_input.strip() == DEFAULT_SOURCE:
                source_path = upload_staging
            else:
                source_path = local_input
        else:
            source_path = local_input

        if not source_path or not os.path.isdir(source_path):
            return "Please select a valid source directory or upload images.", gr.update(interactive=False)

        # --- Determine output path ---
        # Project name takes priority (creates new subfolder)
        # Otherwise use existing_output if it's not the bare default
        output_created = False
        if project_name and project_name.strip():
            final_output_path = os.path.join("datasets/output", project_name.strip())
        elif existing_output and existing_output.strip() and existing_output.strip() != DEFAULT_OUTPUT:
            final_output_path = existing_output.strip()
        else:
            return "No output project defined. Either enter a Project Name (New Project tab) or select an existing project folder (Continue Existing tab).", gr.update(interactive=False)

        # Create output dir if it doesn't exist
        if not os.path.exists(final_output_path):
            try:
                os.makedirs(final_output_path, exist_ok=True)
                output_created = True
            except Exception as e:
                return f"Error creating output directory: {e}", gr.update(interactive=False)

        _, output_caps, source_only_caps = global_state.scan_directory(source_path, final_output_path)
        count = len(global_state.image_paths)

        # Build status message with relative paths
        source_rel = to_relative_path(source_path)
        output_rel = to_relative_path(final_output_path)

        lines = []
        lines.append(f"Source: {source_rel}")
        lines.append(f"Found {count} images")
        total_caps = output_caps + source_only_caps
        if total_caps > 0:
            cap_line = f"Found {total_caps} existing captions"
            if source_only_caps > 0:
                cap_line += f" ({source_only_caps} in source only)"
            lines.append(cap_line)
        if output_created:
            lines.append(f"Output: {output_rel} (created)")
        else:
            lines.append(f"Output: {output_rel}")

        return "\n".join(lines), gr.update(interactive=(count > 0))

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
                file_size = os.path.getsize(path) / 1024
                status_text = f"Selected: {os.path.basename(path)} ({width}x{height}px, {file_size:.0f} KB)"
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

            # Return: path, original_img, status, existing processed images, reset states, tab selection
            return (
                path,                    # selected_path_state
                img,                     # workbench_original
                status_text,             # workbench_status
                existing_upscaled,       # workbench_upscaled
                existing_mask,           # workbench_mask
                existing_transparent,    # workbench_transparent
                existing_upscaled,       # upscaled_image_state (preserve existing upscaled for mask/transparent)
                None,                    # resized_image_state (reset)
                None,                    # mask_image_state (reset)
                None,                    # transparent_image_state (reset)
                gr.Tabs(selected="tab_original")  # Reset to Original tab
            )
        return (None, None, "Select an image from the library.", None, None, None, None, None, None, None,
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
            orig_size = os.path.getsize(image_path) / 1024
            return result, result, f"Upscaled {upscaler.scale}x → {width}x{height}px (original: {orig_size:.0f} KB). Click 'Save' to save as JPG."
        except Exception as e:
            return None, None, f"Upscale Error: {e}"

    def generate_mask_action(image_path, upscaled_img, resized_img, invert_mask):
        """Generate mask from processed image (upscaled/resized) if available, otherwise from original."""
        if not image_path:
            return None, None, "No image selected."

        try:
            seg = get_segmenter()

            # Priority: in-memory upscaled > in-memory resized > saved upscaled on disk > original
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
            elif image_path in global_state.upscaled:
                mask = seg.segment(global_state.upscaled[image_path])
                source_info = "saved upscaled image"
            else:
                mask = seg.segment(image_path)
                source_info = "original image"

            if invert_mask and mask:
                # Invert the mask (white becomes black, black becomes white)
                mask = ImageOps.invert(mask.convert("L")).convert("RGB")

            mask_w, mask_h = mask.size if mask else (0, 0)
            return mask, mask, f"Mask generated from {source_info} ({mask_w}x{mask_h}px). Click 'Save Mask' to save."
        except Exception as e:
            return None, None, f"Mask Error: {e}"

    def generate_transparent_action(image_path, upscaled_img, resized_img, alpha_threshold):
        """Generate transparent image from processed image (upscaled/resized) if available, otherwise from original."""
        if not image_path:
            return None, None, "No image selected."

        try:
            seg = get_segmenter()

            # Priority: in-memory upscaled > in-memory resized > saved upscaled on disk > original
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
            elif image_path in global_state.upscaled:
                transparent = seg.segment(global_state.upscaled[image_path], return_transparent=True)
                source_info = "saved upscaled image"
            else:
                transparent = seg.segment(image_path, return_transparent=True)
                source_info = "original image"

            t_w, t_h = transparent.size if transparent else (0, 0)
            return transparent, transparent, f"Transparent generated from {source_info} ({t_w}x{t_h}px). Click 'Save Transparent' to save."
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
                w, h = original_img.size
            elif isinstance(original_img, np.ndarray):
                pil_img = Image.fromarray(original_img)
                pil_img.save(output_path)
                w, h = pil_img.size
            else:
                import shutil
                shutil.copy2(image_path, output_path)
                w, h = Image.open(output_path).size

            file_size = os.path.getsize(output_path) / 1024
            return f"Saved {os.path.basename(output_path)} ({w}x{h}, {file_size:.0f} KB)"
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
            w, h = upscaled_img.size
            file_size = os.path.getsize(output_path) / 1024
            return f"Saved {os.path.basename(output_path)} ({w}x{h}, {file_size:.0f} KB)"
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
            w, h = mask_img.size
            file_size = os.path.getsize(save_path) / 1024
            return f"Saved {base_name}_mask.png ({w}x{h}, {file_size:.0f} KB)"
        except Exception as e:
            return f"Save Error: {e}"

    def save_transparent_action(image_path, transparent_img):
        """Save the transparent image as PNG to the output directory."""
        if not image_path:
            return "No image selected."
        if transparent_img is None:
            return "No transparent image. Create transparent first."

        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            save_path = os.path.join(base_dir, base_name + "_transparent.png")

            if isinstance(transparent_img, np.ndarray):
                transparent_img = Image.fromarray(transparent_img)

            # Ensure RGBA mode for transparency
            if transparent_img.mode != "RGBA":
                transparent_img = transparent_img.convert("RGBA")

            # Save as PNG for universal transparency support
            transparent_img.save(save_path, "PNG", optimize=True)

            global_state.transparent[image_path] = save_path
            w, h = transparent_img.size
            file_size = os.path.getsize(save_path) / 1024
            return f"Saved {base_name}_transparent.png ({w}x{h}, {file_size:.0f} KB)"
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
        """Generate grayscale PNG masks for all images, using upscaled versions when available."""
        if not global_state.image_paths:
            return "No images loaded."

        seg = get_segmenter()
        total = len(global_state.image_paths)
        success = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Generating masks"):
            try:
                # Use upscaled version if available, otherwise original
                source_path = global_state.upscaled.get(img_path, img_path)

                mask = seg.segment(source_path)
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
        """Generate PNG transparent images for all images, using upscaled versions when available."""
        if not global_state.image_paths:
            return "No images loaded."

        seg = get_segmenter()
        total = len(global_state.image_paths)
        success = 0
        used_upscaled = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Generating transparent"):
            try:
                # Use upscaled version if available, otherwise original
                source_path = global_state.upscaled.get(img_path, img_path)
                if source_path != img_path:
                    used_upscaled += 1

                img = seg.segment(source_path, return_transparent=True)
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

                    # Save as PNG for universal transparency support
                    dummy_path = global_state.get_output_path(img_path, ".txt")
                    base_dir = os.path.dirname(dummy_path)
                    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
                    save_path = os.path.join(base_dir, base_name + "_transparent.png")
                    img.save(save_path, "PNG", optimize=True)
                    global_state.transparent[img_path] = save_path
                    success += 1
            except Exception as e:
                print(f"Transparent error for {img_path}: {e}")

        result = f"Generated {success}/{total} transparent images (PNG)."
        if used_upscaled:
            result += f" ({used_upscaled} from upscaled sources)"
        return result

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
        """Scan output directory for images (excluding masks/ subdirectory)."""
        if not global_state.output_directory or not os.path.isdir(global_state.output_directory):
            return [], {}

        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        images = []
        captions = {}

        for root, dirs, files in os.walk(global_state.output_directory):
            # Skip masks subdirectory
            if 'masks' in dirs:
                dirs.remove('masks')
            for file in files:
                if file.lower().endswith(valid_extensions):
                    file_lower = file.lower()
                    # Always skip mask files
                    if '_mask.' in file_lower:
                        continue
                    images.append(os.path.join(root, file))

        for img_path in images:

            # Try to load existing caption from output dir
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        captions[img_path] = f.read().strip()
                except:
                    captions[img_path] = ""
            else:
                captions[img_path] = ""

        # Fall back to captions loaded during scan (e.g. source-only captions)
        for img_path in images:
            if not captions.get(img_path):
                # Match by filename: find the source path that corresponds to this output image
                basename = os.path.basename(img_path)
                for src_path, cap in global_state.captions.items():
                    if cap and os.path.basename(src_path) == basename:
                        captions[img_path] = cap
                        break

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

    def delete_image_action(path):
        """Delete an image and its caption file from the output directory, then refresh gallery."""
        nonlocal _output_images, _output_captions, _displayed_images
        if not path:
            return "No image selected.", _output_images, None, "", -1

        basename = os.path.basename(path)
        try:
            # Delete image file
            if os.path.exists(path):
                os.remove(path)
            # Delete associated caption file
            txt_path = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(txt_path):
                os.remove(txt_path)

            # Remove from tracking
            _output_captions.pop(path, None)
            if path in _output_images:
                _output_images.remove(path)
            if path in _displayed_images:
                _displayed_images.remove(path)

            return f"Deleted {basename}", _output_images, None, "", -1
        except Exception as e:
            return f"Delete error: {e}", _output_images, None, "", -1

    # Regex to strip BOM + zero-width invisible characters that sneak in via clipboard/browser
    _INVISIBLE_RE = None

    def _clean_tag(tag):
        """Normalize a tag: strip invisible chars, collapse whitespace."""
        import re
        nonlocal _INVISIBLE_RE
        if _INVISIBLE_RE is None:
            _INVISIBLE_RE = re.compile(r'[\ufeff\u200b\u200c\u200d\u2060\ufffe]')
        tag = _INVISIBLE_RE.sub('', tag)
        tag = re.sub(r'\s+', ' ', tag).strip()
        return tag

    def fix_format_action(caption):
        """Fix caption formatting: normalize commas, spacing, collapse extra whitespace, remove empty tags."""
        if not caption:
            return caption, "Nothing to fix."

        original = caption

        # Split by comma, clean each tag, rejoin
        cleaned = [_clean_tag(t) for t in caption.split(",")]
        # Remove empty tags
        cleaned = [tag for tag in cleaned if tag]

        result = ", ".join(cleaned)

        if result == original:
            return result, "Already formatted correctly."
        return result, f"Fixed formatting ({len(original.split(','))} → {len(cleaned)} tags)."

    def dedup_tags_action(caption):
        """Remove duplicate tags while preserving order."""
        if not caption:
            return caption, "Nothing to deduplicate."

        original = caption
        # Normalize: split by comma, strip invisible chars + whitespace
        tags = [_clean_tag(t) for t in caption.split(",")]
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
        return result, f"Removed {duplicates} duplicate(s) ({len(unique)} unique tags)."

    def undo_changes_action(current_path, current_caption):
        """Reload caption from disk."""
        if not current_path:
            return current_caption, "No image selected."
        txt_path = os.path.splitext(current_path)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    restored = f.read().strip()
                _output_captions[current_path] = restored
                return restored, "Reloaded caption from disk."
            except Exception as e:
                return current_caption, f"Error reading file: {e}"
        return "", "No caption file on disk."

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
        result = f"Applied [{pairs_str}] to {count}/{len(_output_images)} captions."
        if count == 0 and _output_images:
            if exact_match:
                result += " Try Partial Match for natural language captions."
            else:
                result += " No matches found in any caption."
        return result

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

    def run_single_captioning_action(img_path, model_name, threshold, prefix_tags, suffix_tags, filter_ratings):
        """Generate caption for a single selected image."""
        if not img_path:
            return "No image selected.", ""

        prefix = prefix_tags.strip().rstrip(",").strip() if prefix_tags else ""
        suffix = suffix_tags.strip().lstrip(",").strip() if suffix_tags else ""
        captioner = None

        try:
            captioner = get_captioner(model_name)

            if "WD" in model_name:
                cap = captioner.generate_caption(img_path, threshold=threshold)
            else:
                cap = captioner.generate_caption(img_path)

            if filter_ratings and cap:
                tags = [t.strip() for t in cap.split(",") if t.strip()]
                tags = filter_rating_tags(tags)
                cap = ", ".join(tags)

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

            return f"Generated caption for {os.path.basename(img_path)}", cap
        except Exception as e:
            return f"Captioning error: {e}", ""
        finally:
            if captioner:
                captioner.unload_model()

    # --- UI LAYOUT ---
    
    with gr.Column() as wizard_container:
        with gr.Tabs(elem_id="wizard_tabs", selected=0) as tabs:
            
            # STEP 1: Project Setup
            with gr.TabItem("Step 1: Project Setup", id=0):
                gr.Markdown("## Step 1: Project Setup")

                # Hidden states
                upload_staging_state = gr.State("")

                with gr.Row():
                    # COLUMN 1: SOURCE DATA (Input) - 50%
                    with gr.Column(scale=1):
                        gr.Markdown("### Source Data")
                        with gr.Tabs() as source_tabs:
                            # Local Folder Tab
                            with gr.TabItem("Local Folder", id="local_folder"):
                                with gr.Accordion("Browse Directories", open=False):
                                    gr.Markdown("*Click any file to select its containing folder.*")
                                    input_explorer = gr.FileExplorer(
                                        root_dir="datasets/input",
                                        glob="**/*",
                                        file_count="single",
                                        height=250
                                    )
                                input_dir = gr.Textbox(
                                    label="Image Source Directory",
                                    placeholder="Path to source images",
                                    value="datasets/input"
                                )

                            # Upload Tab
                            with gr.TabItem("Upload", id="upload"):
                                upload_input = gr.File(
                                    file_count="multiple",
                                    file_types=["image"],
                                    label="Upload Images"
                                )
                                gr.Markdown("*Images are uploaded to the folder set on the Local Folder tab. If no custom folder is set, a new folder will be created at `datasets/input/uploads/`.*")
                                upload_status = gr.Textbox(
                                    label="Upload Status",
                                    interactive=False,
                                    value="Select image files to upload."
                                )

                    # COLUMN 2: WORKSPACE (Output) - 50%
                    with gr.Column(scale=1):
                        gr.Markdown("### Workspace")
                        with gr.Tabs() as workspace_tabs:
                            # New Project Tab
                            with gr.TabItem("New Project", id="new_project"):
                                with gr.Accordion("About Projects", open=False):
                                    gr.Markdown("""**New Project** creates a fresh folder at `datasets/output/[project_name]/` for your processed images and captions.

**Continue Existing** (next tab) lets you resume work on a previous project by selecting its output folder.""")
                                project_name = gr.Textbox(
                                    label="Project Name",
                                    placeholder="e.g. my_awesome_dataset"
                                )

                            # Continue Existing Tab
                            with gr.TabItem("Continue Existing", id="continue_existing"):
                                with gr.Accordion("Browse Existing Projects", open=False):
                                    gr.Markdown("*Click any file to select its containing folder.*")
                                    output_explorer = gr.FileExplorer(
                                        root_dir="datasets/output",
                                        glob="**/*",
                                        file_count="single",
                                        height=250
                                    )
                                output_dir = gr.Textbox(
                                    label="Output Directory",
                                    placeholder="Path to existing output folder",
                                    value="datasets/output"
                                )

                # Full-width Scan Button
                scan_btn = gr.Button("Scan & Initialize Project", variant="primary", size="lg")

                # Collapsible Console Log
                with gr.Accordion("Console Log", open=True):
                    scan_output = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        lines=3,
                        placeholder="Ready to scan..."
                    )

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
                    # COLUMN 1: THE LIBRARY (40%)
                    with gr.Column(scale=40):
                        step2_gallery = gr.Gallery(
                            label="Image Library",
                            columns=4,
                            height=700,
                            allow_preview=False,
                            show_label=False
                        )
                        workbench_status = gr.Textbox(
                            label="Status",
                            value="Select an image from the library.",
                            interactive=False
                        )

                    # COLUMN 2: THE WORKBENCH (60%)
                    with gr.Column(scale=60):
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
                                with gr.Row():
                                    alpha_threshold_slider = gr.Slider(
                                        label="Alpha Threshold",
                                        minimum=0,
                                        maximum=255,
                                        value=128,
                                        step=1,
                                        info="Pixels with opacity below this value are removed; lower values preserve faint details like hair or smoke."
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
                            gr.Markdown("""- **Passthrough** (between thresholds, saved as 98% JPG)
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
                hygiene_result_state = gr.State("")

                # PHASE 1: Caption / Tag Generation (Accordion, Open by Default)
                with gr.Accordion("Caption / Tag Generation", open=True):
                    with gr.Row():
                        # Left column: Model selector, Prefix, Suffix
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
                            prefix_tags_box = gr.Textbox(
                                label="Prefix (Prepend)",
                                placeholder="e.g., sks, 1girl",
                                max_lines=1
                            )
                            suffix_tags_box = gr.Textbox(
                                label="Suffix (Append)",
                                placeholder="e.g., best quality, 4k",
                                max_lines=1
                            )
                        # Right column: VRAM Requirements, Threshold, Filter
                        with gr.Column(scale=1):
                            with gr.Accordion("🖥️ Model Information", open=False):
                                gr.Markdown("""
- **SmilingWolf WD14 (ViT/ConvNext):** ~2GB VRAM. Fast Danbooru-style tagging using ONNX runtime. Best for anime/illustration.
- **BLIP-Base:** ~2GB VRAM. Natural language captions, good general purpose.
- **BLIP-Large:** ~4GB VRAM. More detailed natural language captions.
- **Florence-2-Base:** ~4GB VRAM. Microsoft's vision model, detailed scene descriptions.
- **Florence-2-Large:** ~4GB VRAM. More detailed than base, similar VRAM usage.
- **JoyCaption Quantized (8-bit):** ~12-16GB VRAM. High quality captions, requires 16GB+ GPU.
- **JoyCaption (Beta One):** ~17GB VRAM. Full BF16 precision, requires 20GB+ GPU (RTX 3090/4090).
""")
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
                        caption_btn = gr.Button("Generate for All Images", variant="primary")
                        caption_single_btn = gr.Button("Generate for Selected Image", variant="primary")

                # Main area: Library (left) and Editor (right)
                with gr.Row():
                    # LEFT COLUMN: Library (40%)
                    with gr.Column(scale=40):
                        gallery = gr.Gallery(
                            columns=4,
                            height=700,
                            allow_preview=False,
                            show_label=False
                        )
                        save_status = gr.Textbox(
                            label="Status",
                            value="Select an image from the gallery.",
                            interactive=False
                        )

                    # RIGHT COLUMN: Editor (60%)
                    with gr.Column(scale=60):
                        search_filter_box = gr.Textbox(
                            label="Filter",
                            placeholder="Filter images by caption content...",
                            max_lines=1
                        )
                        editor_preview = gr.Image(
                            type="pil",
                            height=500,
                            interactive=False,
                            show_label=False
                        )
                        editor_caption = gr.Textbox(
                            label="Caption / Tags",
                            lines=6,
                            placeholder="Select an image to edit its caption...",
                            autofocus=True
                        )

                        # Edit Tools
                        gr.Markdown("**Edit Tools**")
                        with gr.Row():
                            fix_format_btn = gr.Button("Fix Format")
                            dedup_btn = gr.Button("Dedup Tags")
                            undo_btn = gr.Button("Undo Changes")
                        with gr.Accordion("What do these buttons do?", open=False):
                            gr.Markdown("""- **Fix Format**: Normalize comma spacing, collapse extra whitespace, remove empty tags
- **Dedup Tags**: Remove duplicate tags (case-insensitive)
- **Undo Changes**: Reload caption from disk""")

                        # Navigation
                        with gr.Row():
                            save_next_btn = gr.Button("Save & Next", variant="primary", size="lg", scale=3)
                            delete_image_btn = gr.Button("Delete Image", variant="stop", size="lg", scale=1)

                # Bulk Edit Tools (Accordion, Closed)
                with gr.Accordion("Bulk Edit Tools", open=False):
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
                                choices=["Partial Match", "Exact Match"],
                                value="Partial Match",
                                show_label=False,
                                info="Partial: substring replace. Exact: whole comma-separated tags only."
                            )
                            bulk_replace_box = gr.Textbox(
                                placeholder="old, new; red, blue",
                                show_label=False,
                                max_lines=1
                            )
                            bulk_replace_btn = gr.Button("Replace All", variant="primary")


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

    # Step 1: Project Setup

    # FileExplorer → Textbox updates
    input_explorer.change(on_explorer_select, inputs=input_explorer, outputs=input_dir)
    output_explorer.change(on_explorer_select, inputs=output_explorer, outputs=output_dir)

    # File upload handling (uses local folder path as destination if set)
    upload_input.upload(
        handle_upload,
        inputs=[upload_input, input_dir],
        outputs=[upload_staging_state, upload_status]
    )

    # Unified scan
    scan_btn.click(
        scan_action_unified,
        inputs=[input_dir, upload_staging_state, project_name, output_dir],
        outputs=[scan_output, next_btn_1]
    )

    # Navigation
    next_btn_1.click(lambda: gr.Tabs(selected=1), outputs=tabs)

    # Step 2: Image Tools
    tab_step_2.select(lambda: global_state.image_paths, outputs=step2_gallery)

    # Gallery selection - load image into all workbench tabs and reset to Original tab
    step2_gallery.select(
        on_gallery_select,
        outputs=[
            selected_path_state,       # path
            workbench_original,        # original image display
            workbench_status,          # unified status
            workbench_upscaled,        # existing upscaled (if any)
            workbench_mask,            # existing mask (if any)
            workbench_transparent,     # existing transparent (if any)
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
        outputs=workbench_status
    )

    resize_btn.click(
        resize_action,
        inputs=[selected_path_state, resize_target_slider],
        outputs=[workbench_original, resized_image_state, workbench_status]
    )

    save_resized_btn.click(
        save_resized_action,
        inputs=[selected_path_state, resized_image_state],
        outputs=workbench_status
    )

    # TAB 2: Upscale - Run and Save buttons
    run_upscale_btn.click(
        upscale_action,
        inputs=[selected_path_state, upscaler_model, target_res_slider],
        outputs=[workbench_upscaled, upscaled_image_state, workbench_status]
    )

    save_upscale_btn.click(
        save_upscale_action,
        inputs=[selected_path_state, workbench_upscaled],
        outputs=workbench_status
    )

    refresh_models_btn.click(refresh_upscaler_models, outputs=upscaler_model)

    # TAB 3: Mask - Create and Save buttons (use upscaled_image_state for reliable full-res data)
    create_mask_btn.click(
        generate_mask_action,
        inputs=[selected_path_state, upscaled_image_state, resized_image_state, invert_mask_check],
        outputs=[workbench_mask, mask_image_state, workbench_status]
    )

    save_mask_btn.click(
        save_mask_action,
        inputs=[selected_path_state, workbench_mask],
        outputs=workbench_status
    )

    # TAB 4: Transparent - Create and Save buttons (use upscaled_image_state for reliable full-res data)
    create_transparent_btn.click(
        generate_transparent_action,
        inputs=[selected_path_state, upscaled_image_state, resized_image_state, alpha_threshold_slider],
        outputs=[workbench_transparent, transparent_image_state, workbench_status]
    )

    save_transparent_btn.click(
        save_transparent_action,
        inputs=[selected_path_state, workbench_transparent],
        outputs=workbench_status
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
    unload_btn.click(on_unload_all_models, outputs=workbench_status)

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
        outputs=save_status
    )

    caption_single_btn.click(
        run_single_captioning_action,
        inputs=[current_path_state, model_dropdown, threshold_slider, prefix_tags_box, suffix_tags_box, filter_ratings_check],
        outputs=[save_status, editor_caption]
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
    def fix_format_wrapper(caption, current_path):
        """Fix format on current caption."""
        new_caption, status = fix_format_action(caption)
        return new_caption, status

    def dedup_wrapper(caption, current_path):
        """Deduplicate tags on current caption."""
        new_caption, status = dedup_tags_action(caption)
        return new_caption, status

    # Hygiene tools write to a hidden State first, then .then() pushes to the textbox.
    # This avoids a Gradio bug where a textbox used as both input and output of the
    # same callback doesn't visually refresh.
    fix_format_btn.click(
        fix_format_wrapper,
        inputs=[editor_caption, current_path_state],
        outputs=[hygiene_result_state, save_status]
    ).then(
        lambda cap: cap,
        inputs=hygiene_result_state,
        outputs=editor_caption
    )
    dedup_btn.click(
        dedup_wrapper,
        inputs=[editor_caption, current_path_state],
        outputs=[hygiene_result_state, save_status]
    ).then(
        lambda cap: cap,
        inputs=hygiene_result_state,
        outputs=editor_caption
    )
    undo_btn.click(
        undo_changes_action,
        inputs=[current_path_state, editor_caption],
        outputs=[hygiene_result_state, save_status]
    ).then(
        lambda cap: cap,
        inputs=hygiene_result_state,
        outputs=editor_caption
    )

    # Save & Next button
    save_next_btn.click(
        save_and_next_action,
        inputs=[current_path_state, editor_caption, current_index_state],
        outputs=[save_status, current_path_state, editor_preview, editor_caption, current_index_state]
    )

    # Delete image button
    delete_image_btn.click(
        delete_image_action,
        inputs=current_path_state,
        outputs=[save_status, gallery, editor_preview, editor_caption, current_index_state]
    )

    # Bulk tools
    bulk_add_btn.click(
        bulk_add_tag_to_all,
        inputs=[bulk_add_tags_box, bulk_add_position],
        outputs=save_status
    )
    bulk_remove_btn.click(
        bulk_remove_tag_from_all,
        inputs=bulk_remove_tags_box,
        outputs=save_status
    )
    bulk_replace_btn.click(
        bulk_search_replace_action,
        inputs=[bulk_replace_box, bulk_exact_match],
        outputs=save_status
    )

    # Nav buttons (Step 3)
    def check_unsaved_captions():
        """Check that all output images have a saved .txt caption file on disk."""
        if not _output_images:
            return gr.Tabs(selected=3), gr.skip()
        missing = []
        for img_path in _output_images:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                missing.append(os.path.basename(img_path))
        if missing:
            msg = f"{len(missing)} image(s) have no saved caption: {', '.join(missing[:10])}"
            if len(missing) > 10:
                msg += f" ... and {len(missing) - 10} more"
            return gr.skip(), msg
        return gr.Tabs(selected=3), gr.skip()

    back_btn_3.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    next_btn_3.click(check_unsaved_captions, outputs=[tabs, save_status])

    # Step 4: Export
    def get_stats():
        output_images, output_captions = get_output_images()
        saved_captions = sum(1 for c in output_captions.values() if c)
        return {
            "Source Images": len(global_state.image_paths),
            "Output Images": len(output_images),
            "Saved Captions": saved_captions,
            "Masks Created": len(global_state.masks),
            "Transparent Images": len(global_state.transparent),
            "Upscaled Images": len(global_state.upscaled)
        }

    tab_step_4.select(get_stats, outputs=final_status)
    refresh_btn.click(get_stats, outputs=final_status)
    back_btn_4.click(lambda: gr.Tabs(selected=2), outputs=tabs)

    return wizard_container