import gradio as gr
import os
import torch
import gc
import numpy as np
from PIL import Image, ImageOps
from src.core.state import global_state
from src.core.captioning import get_captioner
from src.core.segmentation import get_segmenter
from src.core.upscaling import get_upscaler, get_available_models, should_upscale

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
            # Get image info
            try:
                with Image.open(path) as img:
                    width, height = img.size
                info = f"Original: {width}x{height}px"
            except:
                info = "Unknown size"

            # Get processing status
            status = global_state.get_processing_status(path)
            status_parts = []
            if status["has_mask"]:
                status_parts.append("Mask")
            if status["has_transparent"]:
                status_parts.append("Transparent")
            if status["has_upscaled"]:
                status_parts.append("Upscaled")
            status_str = ", ".join(status_parts) if status_parts else "Not yet processed"

            filename = os.path.basename(path)
            # Reset working image when selecting new image
            return path, f"{filename}\n{info} | Saved: {status_str}", None, None, None, "Choose base image: Use Original or Upscale"
        return None, "", None, None, None, ""

    def use_original_action(image_path):
        """Use the original image as the working image."""
        if not image_path:
            return None, None, "No image selected."
        try:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            return img, False, f"Using original ({width}x{height}). Ready for processing."
        except Exception as e:
            return None, None, f"Error: {e}"

    def upscale_action(image_path, model_name):
        """Upscale the image and use as working image."""
        if not image_path:
            return None, None, "No image selected."
        if not model_name or model_name == "No models found":
            return None, None, "No upscaler model selected. Place .pth files in .models/"
        try:
            upscaler = get_upscaler(model_name)
            result = upscaler.upscale(image_path)
            width, height = result.size
            return result, True, f"Upscaled {upscaler.scale}x ({width}x{height}). Ready for processing."
        except Exception as e:
            return None, None, f"Upscale Error: {e}"

    def generate_mask_action(working_image):
        """Generate mask from the working image."""
        if working_image is None:
            return None, "No working image. Click 'Use Original' or 'Upscale' first."
        try:
            # Save working image to temp file for segmenter
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                working_image.save(tmp.name)
                temp_path = tmp.name

            seg = get_segmenter()
            mask = seg.segment(temp_path)

            # Clean up temp file
            os.unlink(temp_path)

            return mask, "Mask generated from working image."
        except Exception as e:
            return None, f"Mask Error: {e}"

    def generate_transparent_action(working_image):
        """Generate transparent image from the working image."""
        if working_image is None:
            return None, "No working image. Click 'Use Original' or 'Upscale' first."
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                working_image.save(tmp.name)
                temp_path = tmp.name

            seg = get_segmenter()
            transparent = seg.segment(temp_path, return_transparent=True)

            os.unlink(temp_path)

            return transparent, "Transparent image generated from working image."
        except Exception as e:
            return None, f"Transparent Error: {e}"

    def save_working_image(image_path, working_image, is_upscaled):
        """Save the working image to output directory."""
        if not image_path or working_image is None:
            return "No working image to save."
        try:
            ext = os.path.splitext(image_path)[1]
            output_path = global_state.get_output_path(image_path, ext)

            # Handle different types that Gradio State might provide
            if isinstance(working_image, Image.Image):
                working_image.save(output_path)
            elif isinstance(working_image, np.ndarray):
                Image.fromarray(working_image).save(output_path)
            elif isinstance(working_image, str):
                # If it's a file path, copy it
                import shutil
                shutil.copy2(working_image, output_path)
            else:
                return f"Error: Unexpected image type {type(working_image)}"

            if is_upscaled:
                global_state.upscaled[image_path] = output_path
            return f"Saved base image to {os.path.basename(output_path)}"
        except Exception as e:
            return f"Save Error: {e}"

    def save_mask_action(image_path, mask_image):
        """Save mask to output folder with _mask suffix."""
        if not image_path or mask_image is None:
            return "No mask to save."
        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            save_path = global_state.get_output_path(image_path, "_mask.png")
            # get_output_path uses the original extension position, so we need to construct properly
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            save_path = os.path.join(base_dir, base_name + "_mask.png")
            mask_image.save(save_path)
            global_state.masks[image_path] = save_path
            return f"Saved {base_name}_mask.png"
        except Exception as e:
            return f"Save Error: {e}"

    def save_transparent_action(image_path, transparent_image):
        """Save transparent image to output folder with _transparent suffix."""
        if not image_path or transparent_image is None:
            return "No transparent image to save."
        try:
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            save_path = os.path.join(base_dir, base_name + "_transparent.png")
            transparent_image.save(save_path)
            global_state.transparent[image_path] = save_path
            return f"Saved {base_name}_transparent.png"
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

    def run_bulk_masks_action(progress=gr.Progress()):
        """Generate masks for all images."""
        if not global_state.image_paths:
            return "No images loaded."

        seg = get_segmenter()
        total = len(global_state.image_paths)
        success = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Generating masks"):
            try:
                mask = seg.segment(img_path)
                if mask:
                    # Save mask to flat output folder
                    dummy_path = global_state.get_output_path(img_path, ".txt")
                    base_dir = os.path.dirname(dummy_path)
                    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
                    save_path = os.path.join(base_dir, base_name + "_mask.png")
                    mask.save(save_path)
                    global_state.masks[img_path] = save_path
                    success += 1
            except Exception as e:
                print(f"Mask error for {img_path}: {e}")

        return f"Generated {success}/{total} masks."

    def run_bulk_transparent_action(progress=gr.Progress()):
        """Generate transparent images for all images."""
        if not global_state.image_paths:
            return "No images loaded."

        seg = get_segmenter()
        total = len(global_state.image_paths)
        success = 0

        for img_path in progress.tqdm(global_state.image_paths, desc="Generating transparent"):
            try:
                img = seg.segment(img_path, return_transparent=True)
                if img:
                    # Save to flat output folder
                    dummy_path = global_state.get_output_path(img_path, ".txt")
                    base_dir = os.path.dirname(dummy_path)
                    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
                    save_path = os.path.join(base_dir, base_name + "_transparent.png")
                    img.save(save_path)
                    global_state.transparent[img_path] = save_path
                    success += 1
            except Exception as e:
                print(f"Transparent error for {img_path}: {e}")

        return f"Generated {success}/{total} transparent images."

    def run_bulk_prepare_output(model_name, min_dim, progress=gr.Progress()):
        """Prepare all images for output - copy or upscale as needed."""
        if not global_state.image_paths:
            return "No images loaded."

        total = len(global_state.image_paths)
        upscaled_count = 0
        copied_count = 0

        upscaler = None
        if model_name:
            try:
                upscaler = get_upscaler(model_name)
            except Exception as e:
                return f"Error loading upscaler: {e}"

        for img_path in progress.tqdm(global_state.image_paths, desc="Preparing output"):
            try:
                # Determine output path (preserve original extension)
                ext = os.path.splitext(img_path)[1]
                output_path = global_state.get_output_path(img_path, ext)

                # Check if already processed
                if img_path in global_state.upscaled:
                    continue

                # Check if needs upscaling
                if upscaler and should_upscale(img_path, min_dim):
                    result = upscaler.upscale(img_path)
                    result.save(output_path)
                    global_state.upscaled[img_path] = output_path
                    upscaled_count += 1
                else:
                    # Just copy the original
                    import shutil
                    shutil.copy2(img_path, output_path)
                    copied_count += 1
            except Exception as e:
                print(f"Output prep error for {img_path}: {e}")

        return f"Prepared {total} images: {upscaled_count} upscaled, {copied_count} copied."

    def refresh_upscaler_models():
        """Refresh the list of available upscaler models."""
        models = get_available_models()
        if not models:
            return gr.update(choices=["No models found"], value=None)
        return gr.update(choices=models, value=models[0])

    def get_output_images():
        """Scan output directory for images (excluding masks, and transparent only if base exists)."""
        if not global_state.output_directory or not os.path.isdir(global_state.output_directory):
            return [], {}

        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        all_files = set()
        images = []
        captions = {}

        # First pass: collect all image filenames
        for root, _, files in os.walk(global_state.output_directory):
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
                # Check if base image exists (remove _transparent suffix)
                base_name = file.replace('_transparent.', '.').replace('_Transparent.', '.')
                base_path = os.path.join(os.path.dirname(img_path), base_name)
                if base_path in all_files:
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

    def refresh_output_gallery():
        """Refresh the output images gallery for Step 3."""
        nonlocal _output_images, _output_captions
        _output_images, _output_captions = get_output_images()
        if not _output_images:
            return [], "No images found in output directory. Process images in Step 2 first."
        return _output_images, f"Found {len(_output_images)} images in output directory."

    def on_select_output_image(evt: gr.SelectData):
        """Handle gallery selection in Step 3 (output images)."""
        index = evt.index
        if index < len(_output_images):
            path = _output_images[index]
            caption = _output_captions.get(path, "")
            return path, caption
        return None, ""

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

    def run_output_captioning_action(model_name, progress=gr.Progress()):
        """Generate captions for all output images."""
        if not _output_images:
            return "No images in output directory."
        captioner = None
        total = len(_output_images)
        success_count = 0
        try:
            captioner = get_captioner(model_name)
            for img_path in progress.tqdm(_output_images):
                try:
                    cap = captioner.generate_caption(img_path)
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

    def bulk_output_replace_action(find_text, replace_text):
        """Replace text in all output captions."""
        if not find_text:
            return "Error: Find text empty."
        count = 0
        for path, cap in _output_captions.items():
            if find_text in cap:
                new_cap = cap.replace(find_text, replace_text)
                _output_captions[path] = new_cap
                try:
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(new_cap)
                    count += 1
                except:
                    pass
        return f"Updated {count} captions."

    def bulk_output_add_action(text, mode):
        """Add text to all output captions."""
        if not text:
            return "Error: Text empty."
        count = 0
        for path, cap in _output_captions.items():
            new_cap = f"{text}{cap}" if mode == "prepend" else f"{cap}{text}"
            _output_captions[path] = new_cap
            try:
                txt_path = os.path.splitext(path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_cap)
                count += 1
            except:
                pass
        return f"{mode.capitalize()}ed text to {count} captions."

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
                gr.Markdown("## Step 2: Masking & Image Tools")

                with gr.Accordion("ℹ️ Workflow Guide", open=False):
                    gr.Markdown("""
                    ### Workflow
                    1. **Select an image** from the gallery
                    2. **Choose base image**: Use original or upscale it
                    3. **Save base image** to output folder
                    4. **Optionally** generate mask or transparent version from the base image

                    ### Upscaling
                    Place Real-ESRGAN `.pth` model files in the `.models/` directory.

                    ### Masks
                    Masks tell the training script which parts of the image are important.
                    - **LoRAs**: Optional but recommended to focus on a subject.
                    - **Full Fine Tune**: Critical to prevent 'background bleeding'.
                    """)

                with gr.Row():
                    # Left: Gallery and Bulk Operations
                    with gr.Column(scale=3):
                        step2_gallery = gr.Gallery(
                            label="Select Image",
                            columns=4,
                            height=800,
                            allow_preview=False,
                            show_label=True
                        )

                        # Status bar - visible position
                        tool_status = gr.Textbox(label="Status", value="Select an image to begin.", interactive=False)

                        # Bulk Operations - collapsible
                        with gr.Accordion("Bulk Operations", open=False):
                            with gr.Row():
                                bulk_masks_btn = gr.Button("All Masks")
                                bulk_transparent_btn = gr.Button("All Transparent")
                                bulk_upscale_btn = gr.Button("All Upscale")
                            min_dim_slider = gr.Slider(
                                label="Min Dimension for Upscaling (px)",
                                minimum=256,
                                maximum=2048,
                                value=1024,
                                step=64,
                                info="Images smaller than this will be upscaled"
                            )
                            bulk_status = gr.Textbox(label="Bulk Status", interactive=False)

                    # Right: Side panel - workflow
                    with gr.Column(scale=2):
                        # Hidden states
                        selected_path_state = gr.State()
                        working_image_state = gr.State()
                        is_upscaled_state = gr.State(False)
                        mask_result_state = gr.State()
                        transparent_result_state = gr.State()

                        # Image info
                        selected_info = gr.Textbox(label="Selected Image", interactive=False, lines=2)

                        # Step 1: Choose base image
                        gr.Markdown("### 1. Choose Base Image")
                        with gr.Row():
                            upscaler_model = gr.Dropdown(
                                label="Upscaler Model",
                                choices=get_available_models() or ["No models found"],
                                value=get_available_models()[0] if get_available_models() else None,
                                scale=2
                            )
                            refresh_models_btn = gr.Button("↻", scale=0, min_width=40)
                        with gr.Row():
                            use_original_btn = gr.Button("Use Original", variant="secondary")
                            upscale_btn = gr.Button("Upscale", variant="primary")

                        # Working image preview
                        gr.Markdown("### 2. Working Image")
                        working_preview = gr.Image(label="Base Image", type="pil", height=280, interactive=False)
                        save_base_btn = gr.Button("Save Base Image to Output", variant="primary")

                        # Step 2: Optional processing
                        gr.Markdown("### 3. Additional Processing (Optional)")
                        with gr.Row():
                            gen_mask_btn = gr.Button("Generate Mask")
                            gen_transparent_btn = gr.Button("Generate Transparent")

                        with gr.Row():
                            mask_preview = gr.Image(label="Mask", type="pil", height=200, interactive=False)
                            transparent_preview = gr.Image(label="Transparent", type="pil", height=200, interactive=False)

                        with gr.Row():
                            save_mask_btn = gr.Button("Save Mask")
                            save_transparent_btn = gr.Button("Save Transparent")

                        # Unload button
                        unload_btn = gr.Button("Unload All Models", variant="secondary")

                with gr.Row():
                    back_btn_2 = gr.Button("< Back")
                    next_btn_2 = gr.Button("Next >")

            # STEP 3: Captioning (moved after image tools)
            with gr.TabItem("Step 3: Captioning", id=2) as tab_step_3:
                gr.Markdown("## Step 3: Captioning & Review")
                gr.Markdown("*Showing images from the output directory. Process images in Step 2 first.*")

                with gr.Accordion("🤖 Auto-Captioning Tools", open=False):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            ["Florence-2-Base", "Florence-2-Large", "BLIP-Base", "BLIP-Large", "JoyCaption (Beta One)", "JoyCaption (4-bit Quantized)", "SmilingWolf WD ViT (v3)", "SmilingWolf WD ConvNext (v3)"],
                            label="Select Model",
                            value="SmilingWolf WD ConvNext (v3)",
                            scale=2
                        )
                        caption_btn = gr.Button("Generate Captions for All Images", scale=1, variant="primary")

                    with gr.Accordion("📊 VRAM Requirements", open=False):
                        gr.Markdown("""
                        - **Florence-2 (Base/Large):** ~1.2GB - 2.0GB
                        - **BLIP (Base/Large):** ~1.0GB - 2.0GB
                        - **WD14 (ViT / ConvNext):** < 1GB
                        - **JoyCaption (4-bit):** ~6GB
                        - **JoyCaption (BF16):** ~16GB
                        *Note: Requirements are estimates and include model loading overhead.*
                        """)

                    progress_bar = gr.Textbox(label="Progress", value="Idle")

                output_gallery_status = gr.Textbox(label="Output Directory Status", interactive=False)

                with gr.Row():
                    # Gallery takes up more space now
                    gallery = gr.Gallery(label="Output Images", columns=6, height=600, allow_preview=False, scale=3)

                    # Editor Side Panel
                    with gr.Column(scale=2):
                        gr.Markdown("### Edit Caption")
                        editor_caption = gr.Textbox(label="Caption", lines=10)
                        save_entry_btn = gr.Button("Save Caption", variant="primary")
                        save_status = gr.Textbox(label="Status", interactive=False)
                        current_path_state = gr.State()

                        with gr.Accordion("🛠️ Bulk Tools", open=False):
                            find_box = gr.Textbox(label="Find", placeholder="Text to find")
                            replace_box = gr.Textbox(label="Replace", placeholder="Replacement text")
                            replace_btn = gr.Button("Replace All")

                            tag_box = gr.Textbox(label="Tag / Text", placeholder="Text to add")
                            with gr.Row():
                                prepend_btn = gr.Button("Prepend to All")
                                append_btn = gr.Button("Append to All")
                            caption_bulk_status = gr.Textbox(label="Bulk Status", interactive=False)

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

    # Gallery selection - resets working state
    step2_gallery.select(
        on_gallery_select,
        outputs=[selected_path_state, selected_info,
                 working_image_state, mask_result_state, transparent_result_state, tool_status]
    ).then(
        lambda: (None, None, None),
        outputs=[working_preview, mask_preview, transparent_preview]
    )

    # Use Original button
    use_original_btn.click(
        use_original_action,
        inputs=selected_path_state,
        outputs=[working_image_state, is_upscaled_state, tool_status]
    ).then(
        lambda img: img,
        inputs=working_image_state,
        outputs=working_preview
    )

    # Upscale button
    upscale_btn.click(
        upscale_action,
        inputs=[selected_path_state, upscaler_model],
        outputs=[working_image_state, is_upscaled_state, tool_status]
    ).then(
        lambda img: img,
        inputs=working_image_state,
        outputs=working_preview
    )

    # Save base image button - use working_preview as source since gr.State may not preserve PIL Images
    save_base_btn.click(
        save_working_image,
        inputs=[selected_path_state, working_preview, is_upscaled_state],
        outputs=tool_status
    )

    # Generate Mask button
    gen_mask_btn.click(
        generate_mask_action,
        inputs=working_image_state,
        outputs=[mask_result_state, tool_status]
    ).then(
        lambda img: img,
        inputs=mask_result_state,
        outputs=mask_preview
    )

    # Generate Transparent button
    gen_transparent_btn.click(
        generate_transparent_action,
        inputs=working_image_state,
        outputs=[transparent_result_state, tool_status]
    ).then(
        lambda img: img,
        inputs=transparent_result_state,
        outputs=transparent_preview
    )

    # Save Mask button
    save_mask_btn.click(
        save_mask_action,
        inputs=[selected_path_state, mask_result_state],
        outputs=tool_status
    )

    # Save Transparent button
    save_transparent_btn.click(
        save_transparent_action,
        inputs=[selected_path_state, transparent_result_state],
        outputs=tool_status
    )

    # Bulk operations
    bulk_masks_btn.click(run_bulk_masks_action, outputs=bulk_status)
    bulk_transparent_btn.click(run_bulk_transparent_action, outputs=bulk_status)
    bulk_upscale_btn.click(
        run_bulk_prepare_output,
        inputs=[upscaler_model, min_dim_slider],
        outputs=bulk_status
    )

    # Upscaler settings
    refresh_models_btn.click(refresh_upscaler_models, outputs=upscaler_model)
    unload_btn.click(on_unload_all_models, outputs=tool_status)

    def go_to_step3():
        """Navigate to Step 3 and refresh output gallery."""
        images, status = refresh_output_gallery()
        return gr.Tabs(selected=2), images, status

    back_btn_2.click(lambda: gr.Tabs(selected=0), outputs=tabs)
    next_btn_2.click(go_to_step3, outputs=[tabs, gallery, output_gallery_status])

    # Step 3: Captioning (uses output directory images)
    caption_btn.click(run_output_captioning_action, inputs=model_dropdown, outputs=progress_bar)

    # When tab is selected, refresh gallery with output images
    tab_step_3.select(refresh_output_gallery, outputs=[gallery, output_gallery_status])

    # When gallery image is selected, update editor
    gallery.select(on_select_output_image, outputs=[current_path_state, editor_caption])

    # Save button
    save_entry_btn.click(save_output_caption_action, inputs=[current_path_state, editor_caption], outputs=save_status)

    # Bulk tools
    replace_btn.click(bulk_output_replace_action, inputs=[find_box, replace_box], outputs=caption_bulk_status)
    prepend_btn.click(lambda t: bulk_output_add_action(t, "prepend"), inputs=tag_box, outputs=caption_bulk_status)
    append_btn.click(lambda t: bulk_output_add_action(t, "append"), inputs=tag_box, outputs=caption_bulk_status)

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