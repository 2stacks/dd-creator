import gradio as gr
import os
import torch
import numpy as np
from PIL import Image, ImageOps
from src.core.state import global_state
from src.core.captioning import get_captioner
from src.core.segmentation import get_segmenter

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

    def run_birefnet_action(image_path):
        if not image_path:
            return None, "No image selected."
        
        try:
            seg = get_segmenter()
            mask = seg.segment(image_path)
            # Create a black background image to show the mask clearly if needed, 
            # but returning the mask directly (L mode or RGB) is usually fine for gr.Image
            return mask, "Auto-mask generated."
        except Exception as e:
            return None, f"BiRefNet Error: {e}"

    def run_transparent_action(image_path):
        if not image_path:
            return None, "No image selected."
        
        try:
            seg = get_segmenter()
            # return_transparent=True is the new argument we added to core/segmentation.py
            img = seg.segment(image_path, return_transparent=True)
            return img, "Transparent image generated."
        except Exception as e:
            return None, f"BiRefNet Error: {e}"

    def on_image_change(path):
        if not path:
            return None, None, "Select an image."
        return None, None, "Image loaded. Ready to process."

    def on_unload_model():
        seg = get_segmenter()
        seg.unload_model()
        return None, None, "Model unloaded from VRAM."

    def on_save_result(image_path, result_image, is_mask):
        if not image_path or result_image is None:
            return "No image to save."
        
        try:
            # Determine folder and filename
            suffix = "_mask.png" if is_mask else "_transparent.png"
            subfolder = "masks" if is_mask else "transparent"
            
            # Use get_output_path to determine where the "base" file would be
            # We use .txt as a dummy extension to find the folder location in the output dir
            dummy_path = global_state.get_output_path(image_path, ".txt")
            base_dir = os.path.dirname(dummy_path)
            
            output_dir = os.path.join(base_dir, subfolder)
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            filename = base_name + suffix
            save_path = os.path.join(output_dir, filename)
            
            result_image.save(save_path)
            
            # Register in state if it's a mask
            if is_mask:
                global_state.masks[image_path] = save_path
                
            return f"Saved to {subfolder}/{filename}"
        except Exception as e:
            return f"Save Error: {e}"

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

            # STEP 2 (MERGED)
            with gr.TabItem("Step 2: Captioning", id=1) as tab_step_2:
                gr.Markdown("## Step 2: Captioning & Review")
                
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

                with gr.Row():
                    # Gallery takes up more space now
                    gallery = gr.Gallery(label="Images", columns=6, height=600, allow_preview=False, scale=3)
                    
                    # Editor Side Panel
                    with gr.Column(scale=2):
                        gr.Markdown("### Edit Tags")
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
                            bulk_status = gr.Textbox(label="Bulk Status")

                with gr.Row():
                    back_btn_2 = gr.Button("< Back")
                    next_btn_2 = gr.Button("Next >")

            # STEP 3 (Old Step 4)
            with gr.TabItem("Step 3: Image Tools", id=2) as tab_step_3:
                gr.Markdown("## Step 3: Masking & Image Tools")
                
                with gr.Accordion("ℹ️ When to use Masks?", open=False):
                    gr.Markdown("""
                    ### Training with Masks
                    Masks are used to tell the training script which parts of the image are important.
                    
                    *   **LoRAs**: Masks are often optional but highly recommended if you want the model to focus strictly on a subject.
                    *   **Full Fine Tune**: Masks are critical to prevent 'background bleeding'.
                    
                    **BiRefNet** is used to automatically detect the subject and generate high-quality masks or transparent images.
                    """)

                with gr.Row():
                    img_selector = gr.Dropdown(label="Select Image to Edit", scale=2)
                
                with gr.Row():
                    # Left: Controls
                    with gr.Column(scale=1):
                        gr.Markdown("### Actions")
                        birefnet_btn = gr.Button("Generate Mask", variant="primary")
                        transparent_btn = gr.Button("Generate Transparent Image")
                        unload_btn = gr.Button("🗑️ Unload Model")
                        
                        gr.Markdown("### Save Result")
                        # Hidden state to track if current result is a mask (True) or transparent image (False)
                        is_mask_state = gr.State(True) 
                        save_result_btn = gr.Button("Save to Disk", variant="primary")
                        tool_status = gr.Textbox(label="Status", value="Ready.")

                    # Right: Preview
                    with gr.Column(scale=2):
                        result_output = gr.Image(label="Generated Result", type="pil", interactive=False)

                with gr.Row():
                    back_btn_3 = gr.Button("< Back")
                    next_btn_3 = gr.Button("Next >")

            # STEP 4 (Old Step 5)
            with gr.TabItem("Step 4: Export", id=3) as tab_step_4:
                gr.Markdown("## Step 4: Project Finalized")
                gr.Markdown("Your files have been saved alongside your images.")
                final_status = gr.JSON(label="Current Session Stats")
                refresh_btn = gr.Button("Refresh Stats")
                back_btn_4 = gr.Button("< Back")

    # --- EVENT BINDINGS (Wired at the end to ensure all components exist) ---
    
    # Step 1
    browse_input_btn.click(browse_directory, inputs=input_dir, outputs=input_dir)
    browse_output_btn.click(browse_directory, inputs=output_dir, outputs=output_dir)
    scan_btn.click(scan_action, inputs=[input_dir, output_dir, project_name], outputs=[scan_output, next_btn_1])
    next_btn_1.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    
    # Step 2 (Merged)
    # Auto-captioning triggers
    caption_btn.click(run_captioning_action, inputs=model_dropdown, outputs=progress_bar)
    
    # Selection triggers
    # When tab is selected, refresh gallery
    tab_step_2.select(lambda: global_state.image_paths, outputs=gallery)
    
    # When gallery image is selected, update editor
    gallery.select(on_select_image, outputs=[current_path_state, editor_caption])
    
    # Save button
    save_entry_btn.click(save_caption_action, inputs=[current_path_state, editor_caption], outputs=save_status)
    
    # Bulk tools
    replace_btn.click(bulk_replace_action, inputs=[find_box, replace_box], outputs=bulk_status)
    prepend_btn.click(lambda t: bulk_add_action(t, "prepend"), inputs=tag_box, outputs=bulk_status)
    append_btn.click(lambda t: bulk_add_action(t, "append"), inputs=tag_box, outputs=bulk_status)
    
    # Nav buttons
    back_btn_2.click(lambda: gr.Tabs(selected=0), outputs=tabs)
    next_btn_2.click(lambda: gr.Tabs(selected=2), outputs=tabs)

    # Step 3
    tab_step_3.select(lambda: gr.update(choices=global_state.image_paths), outputs=img_selector)
    
    img_selector.change(
        on_image_change,
        inputs=img_selector,
        outputs=[result_output, is_mask_state, tool_status]
    )

    birefnet_btn.click(
        run_birefnet_action,
        inputs=img_selector,
        outputs=[result_output, tool_status]
    ).then(
        lambda: True, outputs=is_mask_state
    )

    transparent_btn.click(
        run_transparent_action,
        inputs=img_selector,
        outputs=[result_output, tool_status]
    ).then(
        lambda: False, outputs=is_mask_state
    )

    unload_btn.click(
        on_unload_model,
        outputs=[result_output, is_mask_state, tool_status]
    )
    
    save_result_btn.click(
        on_save_result,
        inputs=[img_selector, result_output, is_mask_state],
        outputs=tool_status
    )

    back_btn_3.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=3), outputs=tabs)

    # Step 4
    def get_stats():
        return {
            "Images Found": len(global_state.image_paths),
            "Captions Loaded/Created": len(global_state.captions),
            "Masks Created": len(global_state.masks)
        }
    tab_step_4.select(get_stats, outputs=final_status)
    refresh_btn.click(get_stats, outputs=final_status)
    back_btn_4.click(lambda: gr.Tabs(selected=2), outputs=tabs)

    return wizard_container