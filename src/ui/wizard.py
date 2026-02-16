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
from src.core.sam_segmenter import get_sam_segmenter
from src.core.inpainting import get_lama_inpainter, get_sd_inpainter, SDInpainter
from src.core.export import export_kohya, export_ai_toolkit, export_onetrainer, export_huggingface, push_to_huggingface

SESSION_FILE = ".last_session.json"

def render_wizard(demo=None, resume=False):
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
        source_stats = [f"{count} images"]
        if output_caps > 0:
            source_stats.append(f"{output_caps} output captions")
        if source_only_caps > 0:
            source_stats.append(f"{source_only_caps} source-only captions")
        lines.append(f"Source contains: {', '.join(source_stats)}")
        if output_created:
            lines.append(f"Output: {output_rel} (created)")
        else:
            lines.append(f"Output: {output_rel}")
            # Report existing output directory contents
            output_stats = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            output_image_count = 0
            output_caption_count = 0
            mask_count = 0
            transparent_count = 0
            inpainted_count = 0
            for root, dirs, files in os.walk(final_output_path):
                if 'masks' in dirs:
                    dirs.remove('masks')
                if 'export' in dirs:
                    dirs.remove('export')
                for f in files:
                    fl = f.lower()
                    if fl.endswith('.txt'):
                        output_caption_count += 1
                        continue
                    if not fl.endswith(valid_extensions):
                        continue
                    if '_mask.' in fl:
                        continue
                    if '_transparent.' in fl:
                        transparent_count += 1
                    elif '_inpainted.' in fl:
                        inpainted_count += 1
                    else:
                        output_image_count += 1
            masks_dir = os.path.join(final_output_path, "masks")
            if os.path.isdir(masks_dir):
                mask_count = sum(1 for f in os.listdir(masks_dir)
                                 if f.lower().endswith('.png') and '_mask' in f.lower())
            if output_image_count:
                output_stats.append(f"{output_image_count} images")
            if output_caption_count:
                output_stats.append(f"{output_caption_count} captions")
            if mask_count:
                output_stats.append(f"{mask_count} masks")
            if transparent_count:
                output_stats.append(f"{transparent_count} transparent")
            if inpainted_count:
                output_stats.append(f"{inpainted_count} inpainted")
            if output_stats:
                lines.append(f"Output contains: {', '.join(output_stats)}")

        # Save session for --resume on next launch
        if count > 0:
            import json
            try:
                with open(SESSION_FILE, "w") as f:
                    json.dump({"source": source_path, "output": final_output_path}, f)
            except Exception:
                pass

        return "\n".join(lines), gr.update(interactive=(count > 0))

    # Cached captioner — stays loaded until the user switches model type
    _cached_captioner = None
    _cached_captioner_name = None

    def _get_or_load_captioner(model_name):
        nonlocal _cached_captioner, _cached_captioner_name
        if _cached_captioner is not None and _cached_captioner_name == model_name:
            return _cached_captioner
        # Different model requested — unload the old one
        if _cached_captioner is not None:
            _cached_captioner.unload_model()
            _cached_captioner = None
            _cached_captioner_name = None
        captioner = get_captioner(model_name)
        captioner.load_model()
        _cached_captioner = captioner
        _cached_captioner_name = model_name
        return captioner

    def _unload_captioner():
        nonlocal _cached_captioner, _cached_captioner_name
        if _cached_captioner is not None:
            _cached_captioner.unload_model()
            _cached_captioner = None
            _cached_captioner_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Captioning model unloaded from VRAM."
        return "No captioning model loaded."

    def run_captioning_action(model_name, progress=gr.Progress()):
        if not global_state.image_paths:
            return "No images to caption."
        total = len(global_state.image_paths)
        success_count = 0
        try:
            captioner = _get_or_load_captioner(model_name)
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

    def _select_image_by_index(index):
        """Core logic for selecting an image by index in Step 2."""
        if global_state.image_paths and 0 <= index < len(global_state.image_paths):
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
            existing_inpainted = None

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

            if proc_status["has_inpainted"] and path in global_state.inpainted:
                try:
                    existing_inpainted = Image.open(global_state.inpainted[path])
                except:
                    pass

            # Best available source: upscaled > original
            best_source = existing_upscaled if existing_upscaled is not None else img
            # For mask/transparent: inpainted > upscaled > original
            best_for_downstream = existing_inpainted if existing_inpainted is not None else best_source
            upscale_display = best_source
            # Gallery-format displays for mask/transparent
            mask_display = [(existing_mask, "mask")] if existing_mask is not None else (
                [(best_for_downstream, "original")] if best_for_downstream is not None else None
            )
            transparent_display = [(existing_transparent, "transparent")] if existing_transparent is not None else (
                [(best_for_downstream, "original")] if best_for_downstream is not None else None
            )
            inpaint_display = existing_inpainted if existing_inpainted is not None else best_source
            return (
                path,                    # selected_path_state
                img,                     # workbench_original
                status_text,             # workbench_status
                upscale_display,         # workbench_upscaled
                mask_display,            # workbench_mask
                transparent_display,     # workbench_transparent
                existing_upscaled,       # upscaled_image_state (preserve existing upscaled for mask/transparent)
                None,                    # resized_image_state (reset)
                None,                    # mask_image_state (reset)
                None,                    # transparent_image_state (reset)
                gr.Tabs(selected="tab_original"),  # Reset to Original tab
                inpaint_display,         # workbench_inpaint (show existing result or original)
                # Inpainting state resets
                [],                      # rect_list_state
                None,                    # rect_first_click_state
                None,                    # inpaint_mask_state
                [],                      # sam_points_state
                [],                      # sam_labels_state
                None,                    # sam_mask_state
                [],                      # preset_mask_state
                None,                    # inpaint_result_state
                # Smart crop resets
                None,                    # smart_crop_results_state
                [(best_for_downstream, "original")],  # smart_crop_gallery (best available)
            )
        return (None, None, "Select an image from the library.", None, None, None, None, None, None, None,
                gr.Tabs(selected="tab_original"),
                None, [], None, None, [], [], None, [], None,
                None, None)

    def on_gallery_select(evt: gr.SelectData):
        """Handle gallery selection in Step 2."""
        return _select_image_by_index(evt.index)

    def auto_select_first_image():
        """Auto-select first image when Step 2 tab loads."""
        gallery_data = global_state.image_paths
        result = _select_image_by_index(0)
        return (gallery_data,) + result

    def upscale_action(image_path, model_name, target_resolution):
        """Upscale the image with smart resize for training.

        Workflow: Upscale with high-quality model, then downscale to target
        resolution on shortest side. This produces sharp details at practical sizes.
        """
        if not image_path:
            return None, None, "No image selected."
        if not model_name or model_name == "No models found":
            return None, None, "No upscaler model selected. Place .pth files in models/"
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

    def _segment_single_image(seg, source_img, image_path, invert_mask):
        """Segment a single image and return the mask. Helper for generate_mask_action."""
        import tempfile
        if source_img is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                if isinstance(source_img, Image.Image):
                    source_img.save(tmp.name)
                elif isinstance(source_img, np.ndarray):
                    Image.fromarray(source_img).save(tmp.name)
                temp_path = tmp.name
            mask = seg.segment(temp_path)
            os.unlink(temp_path)
        elif image_path in global_state.upscaled:
            mask = seg.segment(global_state.upscaled[image_path])
        else:
            mask = seg.segment(image_path)
        if invert_mask and mask:
            mask = ImageOps.invert(mask.convert("L")).convert("RGB")
        return mask

    def generate_mask_action(image_path, upscaled_img, resized_img, inpaint_result, invert_mask, smart_crop_results):
        """Generate mask from best available image or from smart crop results.
        Priority: inpainted > upscaled > resized > original."""
        if not image_path:
            return None, None, "No image selected."

        try:
            seg = get_segmenter()

            # If smart crops exist, process each crop
            if smart_crop_results:
                gallery_data = []
                state_data = []
                for label, crop_img in smart_crop_results:
                    mask = _segment_single_image(seg, crop_img, image_path, invert_mask)
                    if mask:
                        gallery_data.append((mask, f"{label}_mask"))
                        state_data.append((f"{label}_mask", mask))
                count = len(gallery_data)
                return gallery_data, state_data, f"Mask generated for {count} crop(s). Click 'Save Mask' to save."

            # Single image processing
            processed_img = inpaint_result if inpaint_result is not None else (upscaled_img if upscaled_img is not None else resized_img)

            if processed_img is not None:
                if inpaint_result is not None:
                    source_info = "inpainted image"
                elif upscaled_img is not None:
                    source_info = "upscaled image"
                else:
                    source_info = "resized image"
            elif image_path in global_state.upscaled:
                source_info = "saved upscaled image"
            else:
                source_info = "original image"

            mask = _segment_single_image(seg, processed_img, image_path, invert_mask)
            mask_w, mask_h = mask.size if mask else (0, 0)
            gallery_data = [(mask, "mask")] if mask else None
            state_data = [("mask", mask)] if mask else None
            return gallery_data, state_data, f"Mask generated from {source_info} ({mask_w}x{mask_h}px). Click 'Save Mask' to save."
        except Exception as e:
            return None, None, f"Mask Error: {e}"

    def _segment_single_transparent(seg, source_img, image_path):
        """Segment a single image and return transparent version. Helper for generate_transparent_action."""
        import tempfile
        if source_img is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                if isinstance(source_img, Image.Image):
                    source_img.save(tmp.name)
                elif isinstance(source_img, np.ndarray):
                    Image.fromarray(source_img).save(tmp.name)
                temp_path = tmp.name
            transparent = seg.segment(temp_path, return_transparent=True)
            os.unlink(temp_path)
        elif image_path in global_state.upscaled:
            transparent = seg.segment(global_state.upscaled[image_path], return_transparent=True)
        else:
            transparent = seg.segment(image_path, return_transparent=True)
        return transparent

    def generate_transparent_action(image_path, upscaled_img, resized_img, inpaint_result, alpha_threshold, smart_crop_results):
        """Generate transparent image from best available image or from smart crop results.
        Priority: inpainted > upscaled > resized > original."""
        if not image_path:
            return None, None, "No image selected."

        try:
            seg = get_segmenter()

            # If smart crops exist, process each crop
            if smart_crop_results:
                gallery_data = []
                state_data = []
                for label, crop_img in smart_crop_results:
                    transparent = _segment_single_transparent(seg, crop_img, image_path)
                    if transparent:
                        gallery_data.append((transparent, f"{label}_transparent"))
                        state_data.append((f"{label}_transparent", transparent))
                count = len(gallery_data)
                return gallery_data, state_data, f"Transparent generated for {count} crop(s). Click 'Save Transparent' to save."

            # Single image processing
            processed_img = inpaint_result if inpaint_result is not None else (upscaled_img if upscaled_img is not None else resized_img)

            if processed_img is not None:
                if inpaint_result is not None:
                    source_info = "inpainted image"
                elif upscaled_img is not None:
                    source_info = "upscaled image"
                else:
                    source_info = "resized image"
            elif image_path in global_state.upscaled:
                source_info = "saved upscaled image"
            else:
                source_info = "original image"

            transparent = _segment_single_transparent(seg, processed_img, image_path)
            t_w, t_h = transparent.size if transparent else (0, 0)
            gallery_data = [(transparent, "transparent")] if transparent else None
            state_data = [("transparent", transparent)] if transparent else None
            return gallery_data, state_data, f"Transparent generated from {source_info} ({t_w}x{t_h}px). Click 'Save Transparent' to save."
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

    def propagate_source_to_tabs(upscaled_img, resized_img, mask_img, transparent_img, inpaint_result, image_path, smart_crop_results):
        """Update mask/transparent/inpaint/smart-crop previews with best available source image.
        Only updates tabs that don't already have generated output.
        Priority for mask/transparent/smart-crop: inpaint result > upscaled > resized > original
        Priority for inpaint: upscaled > resized > original"""
        base_best = upscaled_img if upscaled_img is not None else resized_img
        if base_best is None and image_path:
            try:
                base_best = Image.open(image_path)
                base_best = ImageOps.exif_transpose(base_best)
            except:
                pass
        # Mask/transparent/smart-crop should use inpainted image if available
        best_for_downstream = inpaint_result if inpaint_result is not None else base_best
        inpaint_out = inpaint_result if inpaint_result is not None else base_best
        # Smart crop: show best available preview unless crops already generated
        if smart_crop_results:
            smart_crop_out = gr.skip()
        else:
            preview = best_for_downstream if best_for_downstream is not None else base_best
            smart_crop_out = [(preview, "original")] if preview is not None else None
        # Mask/transparent galleries: show crops if available, else single best source
        # Note: mask_img/transparent_img are state lists [(label, img), ...] from generate actions
        if smart_crop_results and mask_img is None:
            mask_out = [(crop_img, label) for label, crop_img in smart_crop_results]
        elif mask_img is not None:
            mask_out = [(img, label) for label, img in mask_img]
        elif best_for_downstream is not None:
            mask_out = [(best_for_downstream, "original")]
        else:
            mask_out = None
        if smart_crop_results and transparent_img is None:
            transparent_out = [(crop_img, label) for label, crop_img in smart_crop_results]
        elif transparent_img is not None:
            transparent_out = [(img, label) for label, img in transparent_img]
        elif best_for_downstream is not None:
            transparent_out = [(best_for_downstream, "original")]
        else:
            transparent_out = None
        return mask_out, transparent_out, inpaint_out, smart_crop_out

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

    def save_mask_action(image_path, mask_state):
        """Save mask(s) as grayscale PNG to the masks/ subdirectory.
        mask_state is a list of (label, mask_img) tuples."""
        if not image_path:
            return "No image selected."
        if not mask_state:
            return "No mask. Create mask first."

        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            masks_dir = os.path.join(base_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)

            saved = []
            for label, mask_img in mask_state:
                if isinstance(mask_img, np.ndarray):
                    mask_img = Image.fromarray(mask_img)
                if mask_img.mode != "L":
                    mask_img = mask_img.convert("L")

                # Single mask uses backward-compatible filename
                if label == "mask":
                    filename = base_name + "_mask.png"
                else:
                    filename = f"{base_name}_{label}.png"
                save_path = os.path.join(masks_dir, filename)
                mask_img.save(save_path, "PNG", optimize=True)
                saved.append(filename)

                # Track first/only mask in global state for backward compat
                if label == "mask" or len(mask_state) == 1:
                    global_state.masks[image_path] = save_path

            return f"Saved {len(saved)} mask(s): {', '.join(saved)}"
        except Exception as e:
            return f"Save Error: {e}"

    def save_transparent_action(image_path, transparent_state):
        """Save transparent image(s) as PNG to the output directory.
        transparent_state is a list of (label, transparent_img) tuples."""
        if not image_path:
            return "No image selected."
        if not transparent_state:
            return "No transparent image. Create transparent first."

        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)

            saved = []
            for label, transparent_img in transparent_state:
                if isinstance(transparent_img, np.ndarray):
                    transparent_img = Image.fromarray(transparent_img)
                if transparent_img.mode != "RGBA":
                    transparent_img = transparent_img.convert("RGBA")

                # Single transparent uses backward-compatible filename
                if label == "transparent":
                    filename = base_name + "_transparent.png"
                else:
                    filename = f"{base_name}_{label}.png"
                save_path = os.path.join(base_dir, filename)
                transparent_img.save(save_path, "PNG", optimize=True)
                saved.append(filename)

                # Track first/only transparent in global state for backward compat
                if label == "transparent" or len(transparent_state) == 1:
                    global_state.transparent[image_path] = save_path

            return f"Saved {len(saved)} transparent(s): {', '.join(saved)}"
        except Exception as e:
            return f"Save Error: {e}"


    # --- SMART CROP HELPERS ---

    def run_smart_crop_action(image_path, min_resolution, upscaled_img, resized_img, inpaint_result):
        """Run face-centric smart crop on the best available source image.
        Priority: inpainted > upscaled > resized > original."""
        if not image_path:
            return None, None, "No image selected."
        try:
            from src.core.smart_crop import process_training_image
            # Use best available source (same priority as mask/transparent tabs)
            img = inpaint_result or upscaled_img or resized_img
            if img is None:
                img = Image.open(image_path).convert("RGB")
                img = ImageOps.exif_transpose(img)
            else:
                img = img.convert("RGB")
            results = process_training_image(img, int(min_resolution))
            gallery_data = [(crop_img, label) for label, crop_img in results]
            count = len(results)
            labels = ", ".join(label for label, _ in results)
            return gallery_data, results, f"Smart Crop: {count} crop(s) generated ({labels})"
        except Exception as e:
            return None, None, f"Smart Crop Error: {e}"

    def save_smart_crop_action(image_path, results):
        """Save all generated crops to the output directory."""
        if not image_path:
            return "No image selected."
        if not results:
            return "No crops to save. Run Smart Crop first."
        try:
            saved = []
            for label, crop_img in results:
                out_path = global_state.get_output_path(image_path, f"_{label}.jpg")
                crop_img.save(out_path, "JPEG", quality=98)
                saved.append(os.path.basename(out_path))
            return f"Saved {len(saved)} crop(s): {', '.join(saved)}"
        except Exception as e:
            return f"Save Error: {e}"

    # --- INPAINTING HELPERS ---

    def composite_masks(image_size, rectangles=None, sam_mask=None, preset_masks=None):
        """Combine all mask sources via binary OR into a single mask.

        Args:
            image_size: (width, height) tuple
            rectangles: List of (x1, y1, x2, y2) tuples
            sam_mask: PIL Image mode 'L' or None
            preset_masks: List of PIL Image mode 'L', or single PIL Image, or None

        Returns:
            PIL Image mode 'L' (composite mask)
        """
        from PIL import ImageDraw
        width, height = image_size
        mask = Image.new("L", (width, height), 0)

        if rectangles:
            draw = ImageDraw.Draw(mask)
            for x1, y1, x2, y2 in rectangles:
                draw.rectangle([x1, y1, x2, y2], fill=255)

        if sam_mask is not None:
            if sam_mask.size != (width, height):
                sam_mask = sam_mask.resize((width, height), Image.Resampling.NEAREST)
            sam_arr = np.array(sam_mask)
            mask_arr = np.array(mask)
            mask = Image.fromarray(np.maximum(mask_arr, sam_arr))

        if preset_masks is not None:
            if isinstance(preset_masks, list):
                for pm in preset_masks:
                    if pm.size != (width, height):
                        pm = pm.resize((width, height), Image.Resampling.NEAREST)
                    mask = Image.fromarray(np.maximum(np.array(mask), np.array(pm)))
            else:
                if preset_masks.size != (width, height):
                    preset_masks = preset_masks.resize((width, height), Image.Resampling.NEAREST)
                mask = Image.fromarray(np.maximum(np.array(mask), np.array(preset_masks)))

        return mask

    def _get_inpaint_source(image_path, upscaled_img=None, resized_img=None, inpaint_result=None):
        """Return the best available source image for inpaint mask operations.
        Priority: inpaint result > upscaled > resized > original from disk."""
        for candidate in (inpaint_result, upscaled_img, resized_img):
            if candidate is not None:
                if isinstance(candidate, Image.Image):
                    return candidate.convert("RGB")
                return Image.fromarray(candidate).convert("RGB")
        if image_path:
            try:
                img = Image.open(image_path).convert("RGB")
                return ImageOps.exif_transpose(img)
            except:
                pass
        return None

    def create_mask_overlay(image_path, mask, source_img=None):
        """Create a red-tinted overlay showing masked areas on the source image.

        Args:
            image_path: Path to original image (fallback)
            mask: PIL Image mode 'L', white=areas to inpaint
            source_img: Best available source PIL Image (optional)

        Returns:
            PIL Image (RGB) with red overlay
        """
        if source_img is not None:
            img = source_img.convert("RGB") if isinstance(source_img, Image.Image) else Image.fromarray(source_img).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)

        if mask is None or np.array(mask).max() == 0:
            return img

        if mask.size != img.size:
            mask = mask.resize(img.size, Image.Resampling.NEAREST)

        # Create red overlay
        red_overlay = Image.new("RGB", img.size, (255, 0, 0))
        mask_alpha = mask.point(lambda x: 128 if x > 128 else 0)
        # Composite: blend red where mask is white
        result = Image.composite(red_overlay, img, mask_alpha.convert("L"))
        # That makes masked areas fully red; instead blend 50/50
        img_arr = np.array(img, dtype=np.float32)
        red_arr = np.array(red_overlay, dtype=np.float32)
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_3d = np.stack([mask_arr] * 3, axis=-1)
        blended = img_arr * (1 - mask_3d * 0.5) + red_arr * (mask_3d * 0.5)
        return Image.fromarray(blended.astype(np.uint8))

    def generate_watermark_preset_mask(image_size, preset_name, width_pct, height_pct):
        """Generate a rectangle mask for common watermark positions.

        Args:
            image_size: (width, height)
            preset_name: One of the preset region names
            width_pct: Width percentage (5-50)
            height_pct: Height percentage (5-30)

        Returns:
            PIL Image mode 'L'
        """
        from PIL import ImageDraw
        w, h = image_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        rw = int(w * width_pct / 100)
        rh = int(h * height_pct / 100)

        if preset_name == "Bottom-Right Corner":
            x1, y1 = w - rw, h - rh
            x2, y2 = w, h
        elif preset_name == "Bottom-Center":
            x1, y1 = (w - rw) // 2, h - rh
            x2, y2 = (w + rw) // 2, h
        elif preset_name == "Bottom Strip":
            x1, y1 = 0, h - rh
            x2, y2 = w, h
        elif preset_name == "Top-Left Corner":
            x1, y1 = 0, 0
            x2, y2 = rw, rh
        elif preset_name == "Top-Right Corner":
            x1, y1 = w - rw, 0
            x2, y2 = w, rh
        elif preset_name == "Top Strip":
            x1, y1 = 0, 0
            x2, y2 = w, rh
        else:
            return mask

        draw.rectangle([x1, y1, x2, y2], fill=255)
        return mask

    def add_rectangle_to_mask(image_path, rect_list, x1, y1, x2, y2, sam_mask, source_img=None):
        """Add a rectangle to the mask and return updated overlay."""
        if not image_path:
            return rect_list, None, None, "No image selected."
        try:
            if source_img is not None:
                img = source_img if isinstance(source_img, Image.Image) else Image.fromarray(source_img)
            else:
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
            img_w, img_h = img.size
        except Exception as e:
            return rect_list, None, None, f"Error: {e}"

        # Clamp coordinates
        rx1 = max(0, min(int(x1), img_w))
        ry1 = max(0, min(int(y1), img_h))
        rx2 = max(0, min(int(x2), img_w))
        ry2 = max(0, min(int(y2), img_h))

        if rx1 >= rx2 or ry1 >= ry2:
            return rect_list, None, None, "Invalid rectangle (zero area)."

        if rect_list is None:
            rect_list = []
        rect_list = rect_list + [(rx1, ry1, rx2, ry2)]

        combined = composite_masks((img_w, img_h), rectangles=rect_list, sam_mask=sam_mask)
        overlay = create_mask_overlay(image_path, combined, source_img=img)
        count = len(rect_list)
        sam_info = " + SAM region" if sam_mask is not None else ""
        return rect_list, combined, overlay, f"{count} rectangle(s){sam_info}"

    def undo_last_rectangle(image_path, rect_list, sam_mask, preset_mask,
                            upscaled_img=None, resized_img=None, inpaint_result=None):
        """Remove last rectangle from list."""
        if not rect_list:
            return rect_list, None, None, "No rectangles to undo."

        rect_list = rect_list[:-1]

        if not image_path:
            return rect_list, None, None, "No image selected."

        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size
        except:
            return rect_list, None, None, "Error loading image."

        combined = composite_masks((img_w, img_h), rectangles=rect_list, sam_mask=sam_mask, preset_masks=preset_mask)
        overlay = create_mask_overlay(image_path, combined, source_img=source)
        return rect_list, combined, overlay, f"{len(rect_list)} rectangle(s)"

    def clear_all_masks(image_path, upscaled_img=None, resized_img=None, inpaint_result=None):
        """Clear all mask data and return clean image."""
        if not image_path:
            return [], None, None, None, None, "No image selected."
        try:
            img = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            return [], None, None, None, img, "All masks cleared."
        except:
            return [], None, None, None, None, "Error loading image."


    def add_watermark_preset(image_path, preset_name, width_pct, height_pct,
                             rect_list, preset_list, sam_mask,
                             upscaled_img=None, resized_img=None, inpaint_result=None):
        """Add a watermark preset region to the mask."""
        if preset_list is None:
            preset_list = []
        if not image_path:
            return preset_list, None, None, "No image selected."
        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size
        except Exception as e:
            return preset_list, None, None, f"Error: {e}"

        preset = generate_watermark_preset_mask((img_w, img_h), preset_name, width_pct, height_pct)
        new_list = preset_list + [preset]
        combined = composite_masks(
            (img_w, img_h),
            rectangles=rect_list,
            sam_mask=sam_mask,
            preset_masks=new_list
        )
        overlay = create_mask_overlay(image_path, combined, source_img=source)
        count = len(new_list)
        return new_list, combined, overlay, f"Added {preset_name} preset. {count} preset region(s)."

    def undo_last_preset(image_path, preset_list, rect_list, sam_mask,
                         upscaled_img=None, resized_img=None, inpaint_result=None):
        """Remove last watermark preset region."""
        if not preset_list:
            return preset_list, None, None, "No preset regions to undo."
        new_list = preset_list[:-1]
        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size
            combined = composite_masks(
                (img_w, img_h), rectangles=rect_list, sam_mask=sam_mask, preset_masks=new_list
            )
            overlay = create_mask_overlay(image_path, combined, source_img=source)
            count = len(new_list)
            return new_list, combined, overlay, f"Undid last preset. {count} preset region(s) remaining."
        except Exception as e:
            return new_list, None, None, f"Preset undo error: {e}"

    def clear_preset_regions(image_path, rect_list, sam_mask,
                             upscaled_img=None, resized_img=None, inpaint_result=None):
        """Clear all watermark preset regions."""
        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size
            combined = composite_masks(
                (img_w, img_h), rectangles=rect_list, sam_mask=sam_mask
            )
            overlay = create_mask_overlay(image_path, combined, source_img=source)
            return [], combined, overlay, "All preset regions cleared."
        except:
            return [], None, None, "All preset regions cleared."

    def sam_click_action(image_path, evt_index, sam_points, sam_labels_list, rect_list, preset_mask,
                         upscaled_img=None, resized_img=None, inpaint_result=None):
        """Handle SAM click on image for segmentation."""
        if not image_path:
            return sam_points, sam_labels_list, None, None, None, "No image selected."
        if evt_index is None:
            return sam_points, sam_labels_list, None, None, None, "Click on the image to segment."

        x, y = evt_index
        label = 1  # Always foreground
        if sam_points is None:
            sam_points = []
        if sam_labels_list is None:
            sam_labels_list = []

        sam_points = sam_points + [(x, y)]
        sam_labels_list = sam_labels_list + [label]

        try:
            sam = get_sam_segmenter()
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)

            # SAM needs a file path — save source to temp file if not original
            if upscaled_img is not None or resized_img is not None or inpaint_result is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    source.save(tmp.name)
                    sam_source_path = tmp.name
                sam_mask = sam.segment_multi_point(sam_source_path, [(px, py) for px, py in sam_points], sam_labels_list)
                os.unlink(sam_source_path)
            else:
                sam_mask = sam.segment_multi_point(image_path, [(px, py) for px, py in sam_points], sam_labels_list)

            img_w, img_h = source.size

            combined = composite_masks(
                (img_w, img_h),
                rectangles=rect_list,
                sam_mask=sam_mask,
                preset_masks=preset_mask
            )
            overlay = create_mask_overlay(image_path, combined, source_img=source)

            fg_count = sum(1 for l in sam_labels_list if l == 1)
            bg_count = sum(1 for l in sam_labels_list if l == 0)
            info = f"SAM: {fg_count} foreground, {bg_count} background points"

            return sam_points, sam_labels_list, sam_mask, combined, overlay, info
        except Exception as e:
            return sam_points, sam_labels_list, None, None, None, f"SAM Error: {e}"

    def undo_last_sam(image_path, sam_points, sam_labels_list, rect_list, preset_mask,
                      upscaled_img=None, resized_img=None, inpaint_result=None):
        """Remove last SAM point and recompute mask."""
        if not sam_points:
            return sam_points, sam_labels_list, None, None, None, "No SAM points to undo."

        sam_points = sam_points[:-1]
        sam_labels_list = sam_labels_list[:-1]

        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size

            sam_mask = None
            if sam_points:
                sam = get_sam_segmenter()
                # SAM needs a file path — save source to temp if not original
                if upscaled_img is not None or resized_img is not None or inpaint_result is not None:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        source.save(tmp.name)
                        sam_source_path = tmp.name
                    sam_mask = sam.segment_multi_point(sam_source_path, sam_points, sam_labels_list)
                    os.unlink(sam_source_path)
                else:
                    sam_mask = sam.segment_multi_point(image_path, sam_points, sam_labels_list)

            combined = composite_masks(
                (img_w, img_h), rectangles=rect_list, sam_mask=sam_mask, preset_masks=preset_mask
            )
            overlay = create_mask_overlay(image_path, combined, source_img=source)
            count = len(sam_points)
            return sam_points, sam_labels_list, sam_mask, combined, overlay, f"Undid last SAM point. {count} point(s) remaining."
        except Exception as e:
            return sam_points, sam_labels_list, None, None, None, f"SAM undo error: {e}"

    def clear_sam_points(image_path, rect_list, preset_mask,
                         upscaled_img=None, resized_img=None, inpaint_result=None):
        """Clear SAM points and mask."""
        if not image_path:
            return [], [], None, None, None, "SAM points cleared."
        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size
            combined = composite_masks(
                (img_w, img_h), rectangles=rect_list, preset_masks=preset_mask
            )
            overlay = create_mask_overlay(image_path, combined, source_img=source)
            return [], [], None, combined, overlay, "SAM points cleared."
        except:
            return [], [], None, None, None, "SAM points cleared."

    def run_inpaint_action(image_path, upscaled_img, resized_img, combined_mask, backend, prompt, neg_prompt,
                           steps, guidance, strength):
        """Run inpainting with selected backend. Uses best available source: upscaled > resized > original."""
        if not image_path:
            return None, None, "No image selected."
        if combined_mask is None or np.array(combined_mask).max() == 0:
            return None, None, "No mask defined. Add mask regions first."

        try:
            # Determine source image path: in-memory upscaled/resized > original on disk
            processed_img = upscaled_img if upscaled_img is not None else resized_img
            temp_path = None

            if processed_img is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    if isinstance(processed_img, Image.Image):
                        processed_img.save(tmp.name)
                    elif isinstance(processed_img, np.ndarray):
                        Image.fromarray(processed_img).save(tmp.name)
                    temp_path = tmp.name
                source_path = temp_path
                source_info = "upscaled" if upscaled_img is not None else "resized"
            else:
                source_path = image_path
                source_info = "original"

            if "LaMa" in backend:
                inpainter = get_lama_inpainter()
                result = inpainter.inpaint(source_path, combined_mask)
            else:
                # SD backend
                model_id = None
                for display_name, mid in SDInpainter.MODELS.items():
                    if display_name in backend:
                        model_id = mid
                        break
                if model_id is None:
                    if temp_path:
                        os.unlink(temp_path)
                    return None, None, f"Unknown backend: {backend}"

                inpainter = get_sd_inpainter()
                inpainter.load_model(model_id)
                result = inpainter.inpaint(
                    source_path, combined_mask,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    strength=float(strength),
                )

            if temp_path:
                os.unlink(temp_path)

            return result, result, f"Inpainting complete ({backend}, {source_info} image)."
        except Exception as e:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            err_msg = str(e).split('\n')[0][:200]
            print(f"Inpaint Error: {e}")
            return None, None, f"Inpaint Error: {err_msg}"

    def save_inpainted_action(image_path, inpaint_result):
        """Save inpainted image to output directory."""
        if not image_path:
            return "No image selected."
        if inpaint_result is None:
            return "No inpainted image. Run inpaint first."

        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            save_path = os.path.join(base_dir, base_name + "_inpainted.jpg")

            if isinstance(inpaint_result, np.ndarray):
                inpaint_result = Image.fromarray(inpaint_result)
            if inpaint_result.mode != "RGB":
                inpaint_result = inpaint_result.convert("RGB")

            inpaint_result.save(save_path, "JPEG", quality=98, optimize=True)
            global_state.inpainted[image_path] = save_path

            w, h = inpaint_result.size
            file_size = os.path.getsize(save_path) / 1024
            return f"Saved {base_name}_inpainted.jpg ({w}x{h}, {file_size:.0f} KB)"
        except Exception as e:
            return f"Save Error: {e}"


    def on_unload_all_models():
        """Unload all models (segmenter, upscaler, SAM, LaMa, SD) to free VRAM."""
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
        try:
            sam = get_sam_segmenter()
            sam.unload_model()
        except:
            pass
        try:
            lama = get_lama_inpainter()
            lama.unload_model()
        except:
            pass
        try:
            sd = get_sd_inpainter()
            sd.unload_model()
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
        - < upscale_threshold: Upscale with Spandrel → Lanczos resize to target
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
            # Skip masks and export subdirectories
            if 'masks' in dirs:
                dirs.remove('masks')
            if 'export' in dirs:
                dirs.remove('export')
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

    def _image_status(path):
        """Build a status string for a selected image showing filename and caption state."""
        basename = os.path.basename(path)
        txt_path = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt_path):
            return f"{basename} — caption saved"
        elif _output_captions.get(path, ""):
            return f"{basename} — caption in memory (unsaved)"
        else:
            return f"{basename} — no caption"

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
            return path, img, caption, full_index, _image_status(path)
        return None, None, "", -1, "Select an image from the gallery."

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
        """Generate captions for all output images (generator for live status updates)."""
        if not _output_images:
            yield "No images in output directory."
            return
        total = len(_output_images)
        success_count = 0
        prefix = prefix_tags.strip().rstrip(",").strip() if prefix_tags else ""
        suffix = suffix_tags.strip().lstrip(",").strip() if suffix_tags else ""

        try:
            yield f"Loading {model_name}..."
            captioner = _get_or_load_captioner(model_name)
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
                    yield f"Captioning image {success_count}/{total}..."
                except Exception as e:
                    print(f"Captioning error for {img_path}: {e}")
        except Exception as e:
            yield f"Model Error: {e}"
            return
        yield f"Completed. {success_count}/{total} images captioned."

    def run_single_captioning_action(img_path, model_name, threshold, prefix_tags, suffix_tags, filter_ratings):
        """Generate caption for a single selected image (generator for live status updates)."""
        if not img_path:
            yield "No image selected.", ""
            return

        prefix = prefix_tags.strip().rstrip(",").strip() if prefix_tags else ""
        suffix = suffix_tags.strip().lstrip(",").strip() if suffix_tags else ""

        try:
            yield f"Loading {model_name}...", ""
            captioner = _get_or_load_captioner(model_name)

            yield "Generating caption...", ""
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

            yield f"Generated caption for {os.path.basename(img_path)}", cap
        except Exception as e:
            yield f"Captioning error: {e}", ""
            return

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
                        lines=2,
                        max_lines=2,
                        placeholder="Ready to scan...",
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
                            interactive=False,
                            lines=2,
                            max_lines=2,
                        )

                    # COLUMN 2: THE WORKBENCH (60%)
                    with gr.Column(scale=60):
                        # Tabs with context-sensitive toolbars
                        with gr.Tabs() as workbench_tabs:
                            # TAB 1: ORIGINAL
                            with gr.TabItem("Resize", id="tab_original"):
                                workbench_original = gr.Image(
                                    show_label=False,
                                    type="pil",
                                    height=700,
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
                                    show_label=False,
                                    type="pil",
                                    height=700,
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
                                    save_original_from_upscale_btn = gr.Button("Save Original", variant="primary")
                                    run_upscale_btn = gr.Button("Upscale", variant="secondary")
                                    save_upscale_btn = gr.Button("Save Upscale", variant="primary")

                            # TAB 3: INPAINT
                            with gr.TabItem("Inpaint", id="tab_inpaint"):
                                # Hidden states for inpainting
                                inpaint_mask_state = gr.State(None)
                                rect_list_state = gr.State([])
                                rect_first_click_state = gr.State(None)
                                sam_points_state = gr.State([])
                                sam_labels_state = gr.State([])
                                sam_mask_state = gr.State(None)
                                preset_mask_state = gr.State([])
                                inpaint_result_state = gr.State(None)

                                workbench_inpaint = gr.Image(
                                    show_label=False,
                                    type="pil",
                                    height=700,
                                    interactive=False,
                                )

                                inpaint_tool_state = gr.State("Manual Mask")
                                with gr.Tabs() as inpaint_tool_tabs:
                                    with gr.TabItem("Manual Mask", id="inpaint_tab_rect") as inpaint_tab_rect:
                                        gr.Markdown("Click two points on the image to define a rectangle.")
                                        with gr.Row():
                                            undo_rect_btn = gr.Button("Undo Last", size="sm")
                                            clear_rects_btn = gr.Button("Clear Rectangles", size="sm")

                                    with gr.TabItem("SAM Click", id="inpaint_tab_sam") as inpaint_tab_sam:
                                        gr.Markdown("Click on the image to segment objects.")
                                        with gr.Row():
                                            undo_sam_btn = gr.Button("Undo Last", size="sm")
                                            clear_sam_btn = gr.Button("Clear SAM Points", size="sm")

                                    with gr.TabItem("Watermark Preset", id="inpaint_tab_watermark") as inpaint_tab_watermark:
                                        watermark_preset_dropdown = gr.Dropdown(
                                            choices=[
                                                "Bottom-Right Corner",
                                                "Bottom-Center",
                                                "Bottom Strip",
                                                "Top-Left Corner",
                                                "Top-Right Corner",
                                                "Top Strip",
                                            ],
                                            value="Bottom-Right Corner",
                                            label="Preset Region",
                                        )
                                        with gr.Row():
                                            watermark_width_slider = gr.Slider(
                                                label="Width %", minimum=5, maximum=50,
                                                value=20, step=1,
                                            )
                                            watermark_height_slider = gr.Slider(
                                                label="Height %", minimum=5, maximum=30,
                                                value=10, step=1,
                                            )
                                        add_preset_btn = gr.Button("Add Preset Region", size="sm")
                                        with gr.Row():
                                            undo_preset_btn = gr.Button("Undo Last", size="sm")
                                            clear_presets_btn = gr.Button("Clear Preset Regions", size="sm")

                                    with gr.TabItem("Generate Inpaint", id="inpaint_tab_generate") as inpaint_tab_generate:
                                        with gr.Accordion("🖥️ Model Information", open=False):
                                            gr.Markdown("""
- **LaMa (Fast):** ~2GB VRAM. Best for removing watermarks, text, and small artifacts. No prompt needed — fast and automatic.
- **SD 1.5 Inpainting:** ~6GB VRAM. Prompt-guided inpainting for replacing masked areas with specific content. Good balance of quality and speed.
- **SDXL Inpainting:** ~10GB VRAM. Highest quality prompt-guided inpainting. Requires 12GB+ GPU (RTX 3060 12GB / RTX 3090 / RTX 4090).
""")
                                        inpaint_backend = gr.Dropdown(
                                            choices=[
                                                "LaMa (Fast, ~2GB)",
                                                "SD 1.5 Inpainting (~6GB)",
                                                "SDXL Inpainting (~10GB)",
                                            ],
                                            value="LaMa (Fast, ~2GB)",
                                            label="Inpainting Backend",
                                        )
                                        with gr.Column(visible=False) as sd_controls_col:
                                            inpaint_prompt = gr.Textbox(
                                                label="Prompt",
                                                placeholder="Describe what should replace the masked area...",
                                                max_lines=2,
                                            )
                                            with gr.Accordion("Advanced Settings", open=False):
                                                inpaint_neg_prompt = gr.Textbox(
                                                    label="Negative Prompt",
                                                    placeholder="e.g., blurry, low quality, watermark",
                                                    max_lines=2,
                                                )
                                                inpaint_steps = gr.Slider(
                                                    label="Steps", minimum=10, maximum=50,
                                                    value=30, step=1,
                                                )
                                                inpaint_guidance = gr.Slider(
                                                    label="Guidance Scale", minimum=1.0, maximum=20.0,
                                                    value=7.5, step=0.5,
                                                )
                                                inpaint_strength = gr.Slider(
                                                    label="Strength", minimum=0.0, maximum=1.0,
                                                    value=0.85, step=0.05,
                                                )
                                        with gr.Row():
                                            run_inpaint_btn = gr.Button("Run Inpaint", variant="primary")
                                            save_inpainted_btn = gr.Button("Save Inpaint", variant="primary")
                                        inpaint_unload_btn = gr.Button("⏏ Unload All Models", variant="secondary")

                            # TAB 4: SMART CROP
                            with gr.TabItem("Smart Crop", id="tab_smart_crop"):
                                smart_crop_gallery = gr.Gallery(
                                    show_label=False,
                                    columns=3,
                                    height=700,
                                    allow_preview=True,
                                    preview=True,
                                )
                                smart_crop_min_res_slider = gr.Slider(
                                    label="Minimum Crop Resolution (px)",
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    info="Crops smaller than this are discarded"
                                )
                                with gr.Row():
                                    run_smart_crop_btn = gr.Button("Run Smart Crop", variant="secondary")
                                    save_smart_crop_btn = gr.Button("Save All Crops", variant="primary")
                                smart_crop_results_state = gr.State(None)

                            # TAB 5: MASK
                            with gr.TabItem("Mask", id="tab_mask"):
                                workbench_mask = gr.Gallery(
                                    show_label=False,
                                    columns=3,
                                    height=700,
                                    allow_preview=True,
                                    preview=True,
                                )
                                with gr.Row():
                                    invert_mask_check = gr.Checkbox(
                                        label="Invert Mask",
                                        value=False
                                    )
                                with gr.Row():
                                    create_mask_btn = gr.Button("Create Mask (BiRefNet)", variant="secondary")
                                    save_mask_btn = gr.Button("Save Mask", variant="primary")

                            # TAB 6: TRANSPARENT
                            with gr.TabItem("Transparent", id="tab_transparent"):
                                workbench_transparent = gr.Gallery(
                                    show_label=False,
                                    columns=3,
                                    height=700,
                                    allow_preview=True,
                                    preview=True,
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
- **Upscale** (below lower, Spandrel)
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

                # Main area: Gallery (left) and Tabbed Workspace (right)
                with gr.Row():
                    # LEFT COLUMN: Gallery (40%)
                    with gr.Column(scale=40):
                        search_filter_box = gr.Textbox(
                            label="Filter",
                            placeholder="Filter images by caption content...",
                            max_lines=1
                        )
                        gallery = gr.Gallery(
                            columns=4,
                            height=700,
                            allow_preview=False,
                            show_label=False
                        )
                        save_status = gr.Textbox(
                            label="Status",
                            value="Select an image from the gallery.",
                            interactive=False,
                            lines=2,
                            max_lines=2,
                        )

                    # RIGHT COLUMN: Tabbed Workspace (60%)
                    with gr.Column(scale=60):
                        with gr.Tabs():
                            # Editor Tab (default)
                            with gr.TabItem("Editor"):
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
                                with gr.Row():
                                    fix_format_btn = gr.Button("Fix Format")
                                    dedup_btn = gr.Button("Dedup Tags")
                                    undo_btn = gr.Button("Undo Changes")
                                with gr.Accordion("What do these buttons do?", open=False):
                                    gr.Markdown("""- **Fix Format**: Normalize comma spacing, collapse extra whitespace, remove empty tags
- **Dedup Tags**: Remove duplicate tags (case-insensitive)
- **Undo Changes**: Reload caption from disk""")
                                with gr.Row():
                                    save_next_btn = gr.Button("Save & Next", variant="primary", size="lg", scale=3)
                                    delete_image_btn = gr.Button("Delete Image", variant="stop", size="lg", scale=1)

                            # Auto-Tagging Tab
                            with gr.TabItem("Auto-Tagging"):
                                tagging_preview = gr.Image(
                                    type="pil",
                                    height=500,
                                    interactive=False,
                                    show_label=False
                                )
                                model_dropdown = gr.Dropdown(
                                    [
                                        "BLIP-Base",
                                        "BLIP-Large",
                                        "JoyCaption (8-bit)",
                                        "JoyCaption (Beta One)",
                                        "Qwen2.5-VL 3B",
                                        "Qwen2.5-VL 7B (4-bit)",
                                        "Qwen2.5-VL 7B (8-bit)",
                                        "SmilingWolf WD ConvNext (v3)",
                                        "SmilingWolf WD ViT (v3)",
                                    ],
                                    label="Model",
                                    value="BLIP-Base"
                                )
                                with gr.Accordion("Model Information", open=False):
                                    gr.Markdown("""
- **BLIP-Base:** ~1GB VRAM. Natural language captions, good general purpose.
- **BLIP-Large:** ~1GB VRAM. More detailed natural language captions.
- **JoyCaption (8-bit):** ~10GB VRAM. High quality captions, requires 12GB+ GPU.
- **JoyCaption (Beta One):** ~14GB VRAM. Full BF16 precision, requires 16GB+ GPU (RTX 4090/3090).
- **Qwen2.5-VL 3B:** ~10GB VRAM. Fast VLM captions in half precision. Good for quick drafts.
- **Qwen2.5-VL 7B (4-bit):** ~9GB VRAM. Best speed/quality tradeoff. Recommended for 12GB+ GPUs.
- **Qwen2.5-VL 7B (8-bit):** ~12GB VRAM. Highest quality Qwen option, slower on mid-range GPUs.
- **SmilingWolf WD14 (ViT/ConvNext):** ~1GB VRAM. Fast Danbooru-style tagging using ONNX runtime. Best for anime/illustration.
""")
                                with gr.Column(visible=False) as wd14_options_col:
                                    threshold_slider = gr.Slider(
                                        label="Threshold",
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
                                        max_lines=1
                                    )
                                    suffix_tags_box = gr.Textbox(
                                        label="Suffix (Append)",
                                        placeholder="e.g., best quality, 4k",
                                        max_lines=1
                                    )
                                with gr.Row():
                                    caption_btn = gr.Button("Generate for All Images", variant="primary")
                                    caption_single_btn = gr.Button("Generate for Selected Image", variant="primary")
                                    caption_unload_btn = gr.Button("⏏ Unload Model", variant="secondary")

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
                gr.Markdown("## Step 4: Export Dataset")

                with gr.Tabs():
                    # AI-Toolkit tab
                    with gr.TabItem("AI-Toolkit"):
                        gr.Markdown("Export for [AI-Toolkit](https://github.com/ostris/ai-toolkit). "
                                    "Replaces your trigger word with `[trigger]` in captions.")
                        with gr.Accordion("About AI-Toolkit", open=False):
                            gr.Markdown(
                                "### AI-Toolkit by Ostris (The Flux Specialist)\n\n"
                                "- **Best For:** Training Flux models. It is currently the "
                                "\"gold standard\" for Flux LoRAs.\n"
                                "- **Key Strength:** Speed & Optimization. It often implements the latest "
                                "research (like specific Flux optimizers) before anyone else. It uses a "
                                "configuration-based system (YAML files) rather than complex GUI tabs.\n"
                                "- **Folder Requirement:** Flexible. It does not care about folder names. "
                                "It relies on a `config.yaml` file to define repeats and trigger words.\n"
                                "- **Weakness:** Less feature-rich for older models (SD 1.5) compared to Kohya."
                            )
                        aitk_trigger = gr.Textbox(label="Trigger Word", placeholder="e.g. ohwx, sks")
                        aitk_prepend = gr.Checkbox(label="Prepend [trigger] if not found in caption", value=True)
                        with gr.Accordion("Browse Export Directory", open=False):
                            gr.Markdown("*Click any file to select its containing folder.*")
                            aitk_explorer = gr.FileExplorer(
                                root_dir="datasets/output",
                                glob="**/*",
                                file_count="single",
                                height=250
                            )
                        aitk_export_dir = gr.Textbox(label="Export Directory")
                        aitk_export_btn = gr.Button("Export", variant="primary")

                    # Kohya_ss tab
                    with gr.TabItem("Kohya_ss"):
                        gr.Markdown("Export for [Kohya_ss](https://github.com/kohya-ss/sd-scripts). "
                                    "Creates `img/<repeats>_<trigger>/` folder structure.")
                        with gr.Accordion("About Kohya_ss", open=False):
                            gr.Markdown(
                                "### Kohya_ss (The Standard)\n\n"
                                "- **Best For:** Users who need maximum compatibility and a \"battle-tested\" "
                                "workflow. It is the industry standard for SD 1.5 and SDXL, with the widest "
                                "community support.\n"
                                "- **Key Strength:** Versatility. If a new feature drops, Kohya usually gets it "
                                "quickly. It supports almost every model type (SD1.5, SDXL, Flux, Pony).\n"
                                "- **Folder Requirement:** Strict. Requires the `Repeat_TriggerWord` naming "
                                "convention (e.g., `10_cat`) to function correctly.\n"
                                "- **Weakness:** The interface can be overwhelming with hundreds of settings. "
                                "It is slower to adopt bleeding-edge research (like new optimizers) compared to Ostris."
                            )
                        kohya_trigger = gr.Textbox(label="Trigger Word", placeholder="e.g. ohwx, sks")
                        kohya_steps = gr.Number(label="Target Steps per Epoch", value=200, precision=0, minimum=1)
                        with gr.Accordion("Browse Export Directory", open=False):
                            gr.Markdown("*Click any file to select its containing folder.*")
                            kohya_explorer = gr.FileExplorer(
                                root_dir="datasets/output",
                                glob="**/*",
                                file_count="single",
                                height=250
                            )
                        kohya_export_dir = gr.Textbox(label="Export Directory")
                        kohya_export_btn = gr.Button("Export", variant="primary")

                    # OneTrainer tab
                    with gr.TabItem("OneTrainer"):
                        gr.Markdown("Export for [OneTrainer](https://github.com/Nerogar/OneTrainer). "
                                    "Supports flat folder or Kohya-style directory structure.")
                        with gr.Accordion("About OneTrainer", open=False):
                            gr.Markdown(
                                "### OneTrainer (The Power User / UI King)\n\n"
                                "- **Best For:** Users who want a professional, desktop-app experience "
                                "with real-time feedback.\n"
                                "- **Key Strength:** Visual Feedback. It features a node-based graph view "
                                "and a real-time dashboard that shows loss curves and generated samples "
                                "during training, not just after.\n"
                                "- **Folder Requirement:** Hybrid. It has its own \"Concept\" system but "
                                "fully supports Kohya-style folder structures for backward compatibility.\n"
                                "- **Weakness:** Development can sometimes pause due to developer burnout; "
                                "it may lag behind on support for brand-new architectures (like the very "
                                "first week of a new model release)."
                            )
                        ot_format = gr.Radio(
                            choices=["Flat Folder", "Kohya-style"],
                            value="Flat Folder",
                            label="Format Mode"
                        )
                        with gr.Column(visible=False) as ot_kohya_options:
                            ot_trigger = gr.Textbox(label="Trigger Word", placeholder="e.g. ohwx, sks")
                            ot_steps = gr.Number(label="Target Steps per Epoch", value=200, precision=0, minimum=1)
                        with gr.Accordion("Browse Export Directory", open=False):
                            gr.Markdown("*Click any file to select its containing folder.*")
                            ot_explorer = gr.FileExplorer(
                                root_dir="datasets/output",
                                glob="**/*",
                                file_count="single",
                                height=250
                            )
                        ot_export_dir = gr.Textbox(label="Export Directory")
                        ot_export_btn = gr.Button("Export", variant="primary")

                    # HuggingFace tab
                    with gr.TabItem("HuggingFace"):
                        gr.Markdown("Export for [HuggingFace](https://huggingface.co) datasets. "
                                    "Creates a flat folder with images and a `metadata.jsonl` file.")
                        with gr.Accordion("About HuggingFace", open=False):
                            gr.Markdown(
                                "### Hugging Face (The Cloud Standard)\n\n"
                                "- **Best For:** Users who want to publish their datasets publicly, share them "
                                "with a team, or use cloud-based training services that pull directly from "
                                "Hugging Face.\n"
                                "- **Key Strength:** Interoperability. The \"ImageFolder\" format is universally "
                                "recognized by modern Python libraries (`datasets`, `diffusers`). Uploading this "
                                "format automatically generates a web-based \"Dataset Viewer,\" allowing users to "
                                "browse images and search captions without downloading the files.\n"
                                "- **Folder Structure:** Flat & Simple. Unlike Kohya, it does not use subfolders "
                                "for repeats. It requires a single flat folder containing all images and a "
                                "`metadata.jsonl` file.\n"
                                "- **Metadata Requirement:** The `metadata.jsonl` file must be a \"JSON Lines\" "
                                "file where every line is a valid JSON object containing at least two keys:\n"
                                "  - `file_name` — The exact filename of the image\n"
                                "  - `text` — The caption or training prompt\n"
                                "  - Example: `{\"file_name\": \"img_01.png\", \"text\": \"photo of a sks dog\"}`"
                            )
                        hf_caption_key = gr.Textbox(label="Caption Key", value="text",
                                                     placeholder="JSONL key for captions")
                        with gr.Accordion("Browse Export Directory", open=False):
                            gr.Markdown("*Click any file to select its containing folder.*")
                            hf_explorer = gr.FileExplorer(
                                root_dir="datasets/output",
                                glob="**/*",
                                file_count="single",
                                height=250
                            )
                        hf_export_dir = gr.Textbox(label="Export Directory")
                        hf_export_btn = gr.Button("Export", variant="primary")

                        with gr.Accordion("Push to HuggingFace Hub", open=False):
                            gr.Markdown(
                                "Push your exported dataset directly to the HuggingFace Hub. "
                                "Requires a write token from [huggingface.co/settings/tokens]"
                                "(https://huggingface.co/settings/tokens). "
                                "If you've already logged in via `huggingface-cli login` or set "
                                "`HF_TOKEN`, you can leave the token field blank."
                            )
                            hf_token = gr.Textbox(
                                label="HF Token (optional if already logged in)",
                                type="password",
                                placeholder="hf_...",
                            )
                            hf_repo_name = gr.Textbox(label="Repository Name")
                            hf_private = gr.Checkbox(label="Private Repository", value=True)
                            hf_push_btn = gr.Button("Push to Hub", variant="primary")

                with gr.Accordion("Training Method Guide", open=False):
                    gr.Markdown(
                        "### LoRA (Low-Rank Adaptation)\n\n"
                        "- **What it is:** A small \"patch\" file (20MB \u2013 300MB) that sits on top of a "
                        "base model. It does not change the original model.\n"
                        "- **Best For:** Characters, specific styles, clothing, or objects.\n"
                        "- **Pros:** Extremely lightweight; you can mix and match multiple LoRAs at once "
                        "(e.g., \"Style LoRA\" + \"Character LoRA\"). Fast to train (15\u201330 mins).\n"
                        "- **Cons:** Slightly less powerful than a full fine-tune for changing the entire "
                        "understanding of a model.\n"
                        "- **Dataset Size:** Small. 15\u201330 images is the sweet spot for a character. "
                        "More than 50 often degrades quality unless the images are perfect.\n\n"
                        "---\n\n"
                        "### DreamBooth (Traditional)\n\n"
                        "- **What it is:** A training technique that updates the entire model (6GB+) to "
                        "learn a new subject, usually bound to a rare token (like `sks`).\n"
                        "- **Best For:** Maximum fidelity when \"Good Enough\" isn't acceptable.\n"
                        "- **Pros:** Theoretical 100% likeness capture. Better at learning difficult "
                        "concepts that LoRA struggles with.\n"
                        "- **Cons:** Creates a massive file (Checkpoints are 4GB\u201310GB). You cannot "
                        "easily mix it with other concepts.\n"
                        "- **Dataset Size:** Similar to LoRA (20\u201350 images), but creates a much "
                        "larger output file.\n"
                        "- **Current Status:** Largely obsolete for characters due to modern LoRA quality.\n\n"
                        "---\n\n"
                        "### Full Fine-Tune\n\n"
                        "- **What it is:** Retraining the entire model's weights on a massive dataset.\n"
                        "- **Best For:** Creating a completely new \"Base Model\" (e.g., Pony Diffusion, "
                        "Juggernaut) or erasing concepts (like nudity or style) from a model.\n"
                        "- **Pros:** Fundamental changes to how the model understands prompts and concepts.\n"
                        "- **Cons:** Requires massive hardware (often 80GB+ VRAM or clusters). "
                        "Extremely slow and expensive.\n"
                        "- **Dataset Size:** Massive. Requires hundreds or thousands of images to be effective."
                    )

                export_status = gr.Textbox(label="Status", lines=3, interactive=False)
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

    # Auto-resume last session on app load
    if resume:
        def _resume_session():
            import json
            try:
                with open(SESSION_FILE) as f:
                    session = json.load(f)
                source = session["source"]
                output = session["output"]
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                return {
                    scan_output: "Resume: No saved session found. Please set up your project manually.",
                }

            if not os.path.isdir(source) or not os.path.isdir(output):
                return {
                    scan_output: f"Resume: Saved directories no longer exist.\n  Source: {source}\n  Output: {output}",
                }

            _, output_caps, source_only_caps = global_state.scan_directory(source, output)
            count = len(global_state.image_paths)
            if count == 0:
                return {
                    input_dir: source,
                    output_dir: output,
                    scan_output: f"Resume: No images found in {source}",
                }

            source_rel = to_relative_path(source)
            output_rel = to_relative_path(output)
            lines = [f"Resumed session — {count} images"]
            lines.append(f"Source: {source_rel}")
            lines.append(f"Output: {output_rel}")
            if output_caps > 0 or source_only_caps > 0:
                cap_parts = []
                if output_caps > 0:
                    cap_parts.append(f"{output_caps} output captions")
                if source_only_caps > 0:
                    cap_parts.append(f"{source_only_caps} source-only captions")
                lines.append(f"Found {', '.join(cap_parts)}")

            return {
                input_dir: source,
                output_dir: output,
                scan_output: "\n".join(lines),
                next_btn_1: gr.update(interactive=True),
                tabs: gr.Tabs(selected=1),
            }

        demo.load(
            _resume_session,
            outputs=[input_dir, output_dir, scan_output, next_btn_1, tabs],
        )

    # Step 2: Image Tools - shared output list for workbench updates
    _step2_workbench_outputs = [
        selected_path_state,
        workbench_original,
        workbench_status,
        workbench_upscaled,
        workbench_mask,
        workbench_transparent,
        upscaled_image_state,
        resized_image_state,
        mask_image_state,
        transparent_image_state,
        workbench_tabs,
        workbench_inpaint,
        rect_list_state,
        rect_first_click_state,
        inpaint_mask_state,
        sam_points_state,
        sam_labels_state,
        sam_mask_state,
        preset_mask_state,
        inpaint_result_state,
        smart_crop_results_state,
        smart_crop_gallery,
    ]

    # Auto-select first image when Step 2 tab loads
    tab_step_2.select(
        auto_select_first_image,
        outputs=[step2_gallery] + _step2_workbench_outputs,
    )

    # Gallery selection - load image into all workbench tabs and reset to Original tab
    step2_gallery.select(
        on_gallery_select,
        outputs=_step2_workbench_outputs,
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
    ).then(
        propagate_source_to_tabs,
        inputs=[upscaled_image_state, resized_image_state, mask_image_state,
                transparent_image_state, inpaint_result_state, selected_path_state, smart_crop_results_state],
        outputs=[workbench_mask, workbench_transparent, workbench_inpaint, smart_crop_gallery],
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
    ).then(
        propagate_source_to_tabs,
        inputs=[upscaled_image_state, resized_image_state, mask_image_state,
                transparent_image_state, inpaint_result_state, selected_path_state, smart_crop_results_state],
        outputs=[workbench_mask, workbench_transparent, workbench_inpaint, smart_crop_gallery],
    )

    save_upscale_btn.click(
        save_upscale_action,
        inputs=[selected_path_state, workbench_upscaled],
        outputs=workbench_status
    )
    save_original_from_upscale_btn.click(
        save_original_action,
        inputs=[selected_path_state, workbench_original],
        outputs=workbench_status
    )

    refresh_models_btn.click(refresh_upscaler_models, outputs=upscaler_model)

    # TAB 4: Smart Crop - Run and Save buttons
    run_smart_crop_btn.click(
        run_smart_crop_action,
        inputs=[selected_path_state, smart_crop_min_res_slider,
                upscaled_image_state, resized_image_state, inpaint_result_state],
        outputs=[smart_crop_gallery, smart_crop_results_state, workbench_status]
    ).then(
        propagate_source_to_tabs,
        inputs=[upscaled_image_state, resized_image_state, mask_image_state,
                transparent_image_state, inpaint_result_state, selected_path_state, smart_crop_results_state],
        outputs=[workbench_mask, workbench_transparent, workbench_inpaint, smart_crop_gallery],
    )

    save_smart_crop_btn.click(
        save_smart_crop_action,
        inputs=[selected_path_state, smart_crop_results_state],
        outputs=workbench_status
    )

    # TAB 5: Mask - Create and Save buttons
    create_mask_btn.click(
        generate_mask_action,
        inputs=[selected_path_state, upscaled_image_state, resized_image_state,
                inpaint_result_state, invert_mask_check, smart_crop_results_state],
        outputs=[workbench_mask, mask_image_state, workbench_status]
    )

    save_mask_btn.click(
        save_mask_action,
        inputs=[selected_path_state, mask_image_state],
        outputs=workbench_status
    )

    # TAB 6: Transparent - Create and Save buttons
    create_transparent_btn.click(
        generate_transparent_action,
        inputs=[selected_path_state, upscaled_image_state, resized_image_state,
                inpaint_result_state, alpha_threshold_slider, smart_crop_results_state],
        outputs=[workbench_transparent, transparent_image_state, workbench_status]
    )

    save_transparent_btn.click(
        save_transparent_action,
        inputs=[selected_path_state, transparent_image_state],
        outputs=workbench_status
    )

    # --- INPAINTING EVENT BINDINGS ---

    # Toggle visibility of mask tool controls based on radio selection
    # Track active inpaint tool tab
    # Track active inpaint tool tab via individual TabItem.select() events
    inpaint_tab_rect.select(lambda: "Manual Mask", outputs=inpaint_tool_state)
    inpaint_tab_sam.select(lambda: "SAM Click", outputs=inpaint_tool_state)
    inpaint_tab_watermark.select(lambda: "Watermark Preset", outputs=inpaint_tool_state)
    inpaint_tab_generate.select(lambda: "Generate Inpaint", outputs=inpaint_tool_state)

    # Toggle SD controls visibility based on backend selection
    def toggle_sd_controls(backend):
        is_sd = "SD" in backend or "SDXL" in backend
        return gr.update(visible=is_sd)

    inpaint_backend.change(
        toggle_sd_controls,
        inputs=inpaint_backend,
        outputs=sd_controls_col,
    )

    # Image click handler: routes to Rectangle or SAM based on mode
    def on_inpaint_image_click(image_path, tool_mode,
                                first_click, rect_list, sam_points, sam_labels,
                                sam_mask, preset_mask,
                                upscaled_img, resized_img, inpaint_result,
                                evt: gr.SelectData):
        """Handle click on the inpainting image canvas."""
        if not image_path or evt is None:
            return (first_click, rect_list, sam_points, sam_labels, sam_mask,
                    None, gr.skip(), "No image selected.")

        x, y = evt.index
        source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)

        if tool_mode == "Manual Mask":
            if first_click is None:
                return (
                    (x, y), rect_list, sam_points, sam_labels, sam_mask,
                    None, gr.skip(), f"First corner at ({x}, {y}). Click second corner."
                )
            else:
                x1, y1 = first_click
                rx1, ry1 = min(x1, x), min(y1, y)
                rx2, ry2 = max(x1, x), max(y1, y)
                new_rect_list, new_mask, overlay, msg = add_rectangle_to_mask(
                    image_path, rect_list, rx1, ry1, rx2, ry2, sam_mask, source_img=source
                )
                return (
                    None, new_rect_list, sam_points, sam_labels, sam_mask,
                    new_mask, overlay if overlay is not None else gr.skip(), msg
                )

        elif tool_mode == "SAM Click":
            new_pts, new_labels, new_sam_mask, new_combined, overlay, msg = sam_click_action(
                image_path, (x, y), sam_points, sam_labels, rect_list, preset_mask,
                upscaled_img=upscaled_img, resized_img=resized_img, inpaint_result=inpaint_result
            )
            return (
                first_click, rect_list, new_pts, new_labels, new_sam_mask,
                new_combined, overlay if overlay is not None else gr.skip(), msg
            )

        return (first_click, rect_list, sam_points, sam_labels, sam_mask,
                None, gr.skip(), "Select a mask tool mode first.")

    workbench_inpaint.select(
        on_inpaint_image_click,
        inputs=[selected_path_state, inpaint_tool_state,
                rect_first_click_state, rect_list_state,
                sam_points_state, sam_labels_state, sam_mask_state,
                preset_mask_state,
                upscaled_image_state, resized_image_state, inpaint_result_state],
        outputs=[rect_first_click_state, rect_list_state,
                 sam_points_state, sam_labels_state, sam_mask_state,
                 inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # Shorthand for the three source states used by all inpaint mask handlers
    _inpaint_src_states = [upscaled_image_state, resized_image_state, inpaint_result_state]

    # Rectangle: Undo last
    undo_rect_btn.click(
        undo_last_rectangle,
        inputs=[selected_path_state, rect_list_state, sam_mask_state, preset_mask_state] + _inpaint_src_states,
        outputs=[rect_list_state, inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # Rectangle: Clear all rectangles
    def clear_rects_only(image_path, sam_mask, preset_mask,
                         upscaled_img=None, resized_img=None, inpaint_result=None):
        if not image_path:
            return [], None, None, "Rectangles cleared."
        try:
            source = _get_inpaint_source(image_path, upscaled_img, resized_img, inpaint_result)
            img_w, img_h = source.size
            combined = composite_masks((img_w, img_h), sam_mask=sam_mask, preset_masks=preset_mask)
            overlay = create_mask_overlay(image_path, combined, source_img=source)
            return [], combined, overlay, "Rectangles cleared."
        except:
            return [], None, None, "Rectangles cleared."

    clear_rects_btn.click(
        clear_rects_only,
        inputs=[selected_path_state, sam_mask_state, preset_mask_state] + _inpaint_src_states,
        outputs=[rect_list_state, inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # SAM: Undo last point
    undo_sam_btn.click(
        undo_last_sam,
        inputs=[selected_path_state, sam_points_state, sam_labels_state,
                rect_list_state, preset_mask_state] + _inpaint_src_states,
        outputs=[sam_points_state, sam_labels_state, sam_mask_state,
                 inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # SAM: Clear points
    clear_sam_btn.click(
        clear_sam_points,
        inputs=[selected_path_state, rect_list_state, preset_mask_state] + _inpaint_src_states,
        outputs=[sam_points_state, sam_labels_state, sam_mask_state,
                 inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # Watermark preset: Add region
    add_preset_btn.click(
        add_watermark_preset,
        inputs=[selected_path_state, watermark_preset_dropdown,
                watermark_width_slider, watermark_height_slider,
                rect_list_state, preset_mask_state, sam_mask_state] + _inpaint_src_states,
        outputs=[preset_mask_state, inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # Watermark preset: Undo last
    undo_preset_btn.click(
        undo_last_preset,
        inputs=[selected_path_state, preset_mask_state, rect_list_state,
                sam_mask_state] + _inpaint_src_states,
        outputs=[preset_mask_state, inpaint_mask_state, workbench_inpaint, workbench_status],
    )

    # Watermark preset: Clear all
    clear_presets_btn.click(
        clear_preset_regions,
        inputs=[selected_path_state, rect_list_state, sam_mask_state] + _inpaint_src_states,
        outputs=[preset_mask_state, inpaint_mask_state, workbench_inpaint, workbench_status],
    )
    # Run inpaint
    run_inpaint_btn.click(
        run_inpaint_action,
        inputs=[selected_path_state, upscaled_image_state, resized_image_state,
                inpaint_mask_state, inpaint_backend,
                inpaint_prompt, inpaint_neg_prompt, inpaint_steps,
                inpaint_guidance, inpaint_strength],
        outputs=[inpaint_result_state, workbench_inpaint, workbench_status],
    ).then(
        propagate_source_to_tabs,
        inputs=[upscaled_image_state, resized_image_state, mask_image_state,
                transparent_image_state, inpaint_result_state, selected_path_state, smart_crop_results_state],
        outputs=[workbench_mask, workbench_transparent, workbench_inpaint, smart_crop_gallery],
    )

    # Save inpainted
    save_inpainted_btn.click(
        save_inpainted_action,
        inputs=[selected_path_state, inpaint_result_state],
        outputs=workbench_status,
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

    # Unload models (on Upscale tab + Inpaint tab)
    unload_btn.click(on_unload_all_models, outputs=workbench_status)
    inpaint_unload_btn.click(on_unload_all_models, outputs=workbench_status)

    def go_to_step3():
        """Navigate to Step 3 and refresh output gallery."""
        images, status = refresh_output_gallery()
        return gr.Tabs(selected=2), images, status

    def auto_select_first_output():
        """Auto-select the first output image into the editor previews."""
        if not _output_images:
            return None, None, None, "", -1, "No images found in output directory."
        first_path = _output_images[0]
        first_caption = _output_captions.get(first_path, "")
        try:
            first_img = Image.open(first_path)
            first_img = ImageOps.exif_transpose(first_img)
        except Exception:
            first_img = None
        status = f"Found {len(_output_images)} images. {_image_status(first_path)}"
        return first_path, first_img, first_img, first_caption, 0, status

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

    caption_unload_btn.click(_unload_captioner, outputs=save_status)

    # Toggle WD14-specific options based on model selection
    model_dropdown.change(
        lambda m: gr.Column(visible=m.startswith("SmilingWolf")),
        inputs=model_dropdown,
        outputs=wd14_options_col
    )

    # When tab is selected, refresh gallery with output images and auto-select first
    tab_step_3.select(
        refresh_output_gallery, outputs=[gallery, save_status]
    ).then(
        auto_select_first_output,
        outputs=[current_path_state, editor_preview, tagging_preview,
                 editor_caption, current_index_state, save_status]
    )

    # Search/filter functionality
    search_filter_box.change(filter_gallery_action, inputs=search_filter_box, outputs=gallery)

    # When gallery image is selected, update editor with preview
    gallery.select(
        on_select_output_image,
        outputs=[current_path_state, editor_preview, editor_caption, current_index_state, save_status]
    ).then(
        lambda img: img, inputs=editor_preview, outputs=tagging_preview
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
    ).then(
        lambda img: img, inputs=editor_preview, outputs=tagging_preview
    )

    # Delete image button
    delete_image_btn.click(
        delete_image_action,
        inputs=current_path_state,
        outputs=[save_status, gallery, editor_preview, editor_caption, current_index_state]
    ).then(
        lambda img: img, inputs=editor_preview, outputs=tagging_preview
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
    def _get_stats_text():
        output_images, output_captions = get_output_images()
        saved_captions = sum(1 for c in output_captions.values() if c)
        # Count from actual output directory contents for accuracy
        mask_count = 0
        transparent_count = 0
        inpainted_count = 0
        upscaled_count = sum(1 for p in global_state.upscaled.values() if os.path.exists(p))
        if global_state.output_directory and os.path.isdir(global_state.output_directory):
            masks_dir = os.path.join(global_state.output_directory, "masks")
            if os.path.isdir(masks_dir):
                mask_count = sum(1 for f in os.listdir(masks_dir)
                                 if f.lower().endswith('.png') and '_mask' in f.lower())
            for f in os.listdir(global_state.output_directory):
                fl = f.lower()
                if fl.endswith('.txt'):
                    continue
                if '_transparent.' in fl:
                    transparent_count += 1
                elif '_inpainted.' in fl:
                    inpainted_count += 1
        parts = [
            f"Source: {len(global_state.image_paths)}",
            f"Output: {len(output_images)}",
            f"Captions: {saved_captions}",
            f"Masks: {mask_count}",
            f"Transparent: {transparent_count}",
            f"Upscaled: {upscaled_count}",
            f"Inpainted: {inpainted_count}",
        ]
        return "Session — " + " | ".join(parts)

    def _get_default_export_dir(format_name):
        if global_state.output_directory:
            return os.path.join(global_state.output_directory, "export", format_name)
        return ""

    def _get_default_trigger_word():
        if global_state.output_directory:
            return os.path.basename(global_state.output_directory.rstrip(os.sep))
        return ""

    def init_export_tab():
        trigger = _get_default_trigger_word()
        return (
            _get_stats_text(),
            trigger,
            _get_default_export_dir("ai_toolkit"),
            trigger,
            _get_default_export_dir("kohya_ss"),
            trigger,
            _get_default_export_dir("onetrainer"),
            _get_default_export_dir("huggingface"),
            trigger,  # hf_repo_name default
        )

    def run_kohya_export(trigger, steps, export_dir, progress=gr.Progress()):
        if not global_state.output_directory:
            return "No project loaded. Please complete Steps 1-3 first."
        steps = int(steps) if steps else 200
        def progress_cb(current, total, msg):
            progress((current + 1) / total, desc=msg)
        status, _ = export_kohya(global_state.output_directory, export_dir, trigger, steps, progress_cb)
        return status

    def run_aitk_export(trigger, prepend, export_dir, progress=gr.Progress()):
        if not global_state.output_directory:
            return "No project loaded. Please complete Steps 1-3 first."
        def progress_cb(current, total, msg):
            progress((current + 1) / total, desc=msg)
        status, _ = export_ai_toolkit(global_state.output_directory, export_dir, trigger, prepend, progress_cb)
        return status

    def run_ot_export(format_mode, trigger, steps, export_dir, progress=gr.Progress()):
        if not global_state.output_directory:
            return "No project loaded. Please complete Steps 1-3 first."
        mode = "kohya" if format_mode == "Kohya-style" else "flat"
        steps = int(steps) if steps else 200
        def progress_cb(current, total, msg):
            progress((current + 1) / total, desc=msg)
        status, _ = export_onetrainer(global_state.output_directory, export_dir, mode, trigger, steps, progress_cb)
        return status

    def run_hf_export(caption_key, export_dir, progress=gr.Progress()):
        if not global_state.output_directory:
            return "No project loaded. Please complete Steps 1-3 first."
        def progress_cb(current, total, msg):
            progress((current + 1) / total, desc=msg)
        status, _ = export_huggingface(global_state.output_directory, export_dir, caption_key, progress_cb)
        return status

    def run_hf_push(token, repo_name, private, export_dir, progress=gr.Progress()):
        if not global_state.output_directory:
            return "No project loaded. Please complete Steps 1-3 first."
        def progress_cb(msg):
            progress(0, desc=msg)
        status, _ = push_to_huggingface(export_dir, repo_name, token, private, progress_cb)
        return status

    tab_step_4.select(
        init_export_tab,
        outputs=[export_status, aitk_trigger, aitk_export_dir, kohya_trigger, kohya_export_dir, ot_trigger, ot_export_dir, hf_export_dir, hf_repo_name]
    )

    # FileExplorer → Textbox updates for export directories
    aitk_explorer.change(on_explorer_select, inputs=aitk_explorer, outputs=aitk_export_dir)
    kohya_explorer.change(on_explorer_select, inputs=kohya_explorer, outputs=kohya_export_dir)
    ot_explorer.change(on_explorer_select, inputs=ot_explorer, outputs=ot_export_dir)
    hf_explorer.change(on_explorer_select, inputs=hf_explorer, outputs=hf_export_dir)

    ot_format.change(
        lambda mode: gr.Column(visible=(mode == "Kohya-style")),
        inputs=ot_format,
        outputs=ot_kohya_options,
    )

    kohya_export_btn.click(
        run_kohya_export,
        inputs=[kohya_trigger, kohya_steps, kohya_export_dir],
        outputs=export_status,
    )
    aitk_export_btn.click(
        run_aitk_export,
        inputs=[aitk_trigger, aitk_prepend, aitk_export_dir],
        outputs=export_status,
    )
    ot_export_btn.click(
        run_ot_export,
        inputs=[ot_format, ot_trigger, ot_steps, ot_export_dir],
        outputs=export_status,
    )
    hf_export_btn.click(
        run_hf_export,
        inputs=[hf_caption_key, hf_export_dir],
        outputs=export_status,
    )
    hf_push_btn.click(
        run_hf_push,
        inputs=[hf_token, hf_repo_name, hf_private, hf_export_dir],
        outputs=export_status,
    )

    back_btn_4.click(lambda: gr.Tabs(selected=2), outputs=tabs)

    return wizard_container