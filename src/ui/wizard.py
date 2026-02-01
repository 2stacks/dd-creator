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
    
    def browse_directory():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            directory = filedialog.askdirectory()
            root.destroy()
            return directory
        except Exception as e:
            print(f"Browse error: {e}")
            return ""

    def scan_action(path):
        if not path or not os.path.isdir(path):
            return "Please select a valid directory.", gr.update(interactive=False)
        msg = global_state.scan_directory(path)
        count = len(global_state.image_paths)
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
                    # Auto-save to .txt
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

    def on_select_image(evt: gr.SelectData):
        # evt.index is the index in the gallery list
        index = evt.index
        if index < len(global_state.image_paths):
            path = global_state.image_paths[index]
            caption = global_state.captions.get(path, "")
            return path, caption, path 
        return None, "", None

    def save_caption_action(path, new_caption):
        if path and path in global_state.captions:
            global_state.captions[path] = new_caption
            txt_path = os.path.splitext(path)[0] + ".txt"
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
                    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f: f.write(new_cap)
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
                with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f: f.write(new_cap)
                count += 1
            except: pass
        return f"{mode.capitalize()}ed text to {count} captions."

    def on_sam_image_change(path):
        if not path:
            return None, None, [], [], "Select an image."
        
        try:
            # Load image for display
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            
            # Initialize/Update Segmenter
            # Note: We don't block UI here, but computation happens. 
            # ideally we'd use a generator or progress, but keeping it simple for now.
            seg = get_segmenter()
            seg.set_image(path)
            
            return img, None, [], [], "Image loaded. Click to segment."
        except Exception as e:
            print(f"SAM Load Error: {e}")
            return None, None, [], [], f"Error: {e}"

    def on_sam_click(image_path, points, labels, bg_mode_active, evt: gr.SelectData):
        if not image_path: return None, points, labels, "No image selected."
        
        x, y = evt.index
        label = 0 if bg_mode_active else 1
        
        new_points = points + [[x, y]]
        new_labels = labels + [label]
        
        seg = get_segmenter()
        try:
            mask = seg.segment_from_points(new_points, new_labels)
            
            # Create overlay for preview (optional, or just show mask)
            # For now, just return the mask. 
            # Ideally we might want to overlay it on the original image in the UI, 
            # but returning it to a separate Image component is easier.
            
            status = f"Points: {len(new_points)} ({sum(new_labels)} FG, {len(new_labels)-sum(new_labels)} BG)"
            return mask, new_points, new_labels, status
        except Exception as e:
            return None, points, labels, f"Segmentation Error: {e}"

    def on_sam_undo(image_path, points, labels):
        if not points: return None, points, labels, "Nothing to undo."
        
        new_points = points[:-1]
        new_labels = labels[:-1]
        
        seg = get_segmenter()
        mask = None
        status = "Cleared."
        
        if new_points:
            try:
                mask = seg.segment_from_points(new_points, new_labels)
                status = f"Points: {len(new_points)}"
            except Exception as e:
                status = f"Error: {e}"
        else:
            status = "Reset."
            
        return mask, new_points, new_labels, status

    def on_sam_reset():
        return None, [], [], "Reset."

    def on_sam_unload():
        seg = get_segmenter()
        seg.unload_model()
        return None, [], [], "Model unloaded from VRAM."

    def on_sam_save(image_path, mask_image):
        if not image_path or mask_image is None:
            return "No mask to save."
        
        try:
            mask_dir = os.path.join(os.path.dirname(image_path), "masks")
            os.makedirs(mask_dir, exist_ok=True)
            mask_name = os.path.basename(image_path).rsplit('.', 1)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            
            mask_image.save(mask_path)
            global_state.masks[image_path] = mask_path
            return f"Saved mask to {mask_name}"
        except Exception as e:
            return f"Save Error: {e}"

    def on_toggle_mode(is_bg):
        # Toggle boolean
        new_mode = not is_bg
        btn_label = "🚫 Switch to Foreground Mode" if new_mode else "🎯 Switch to Background Mode"
        return new_mode, btn_label

    # --- UI LAYOUT ---
    
    with gr.Column() as wizard_container:
        with gr.Tabs(elem_id="wizard_tabs", selected=0) as tabs:
            
            # STEP 1
            with gr.TabItem("Step 1: Import", id=0):
                gr.Markdown("## Step 1: Select Dataset Directory")
                with gr.Row():
                    input_dir = gr.Textbox(label="Directory Path", placeholder="/path/to/images", scale=4)
                    browse_btn = gr.Button("📂 Browse", scale=1)
                scan_btn = gr.Button("Scan Directory", variant="primary")
                scan_output = gr.Textbox(label="Status", interactive=False)
                next_btn_1 = gr.Button("Next >", interactive=False)

            # STEP 2
            with gr.TabItem("Step 2: Auto-Caption", id=1):
                gr.Markdown("## Step 2: Generate Captions")
                model_dropdown = gr.Dropdown(
                    ["Florence-2-Base", "Florence-2-Large", "BLIP-Base", "BLIP-Large", "JoyCaption (Beta One)", "JoyCaption (4-bit Quantized)", "SmilingWolf WD14 (v3 SwinV2)"], 
                    label="Select Model", value="Florence-2-Base"
                )
                gr.Markdown("**VRAM Requirements:** JoyCaption BF16 (~17GB), JoyCaption 4-bit (~8GB), Florence-2 (<4GB).")
                caption_btn = gr.Button("Generate Captions")
                progress_bar = gr.Textbox(label="Progress", value="Idle")
                with gr.Row():
                    back_btn_2 = gr.Button("< Back")
                    next_btn_2 = gr.Button("Next >")

            # STEP 3
            with gr.TabItem("Step 3: Review", id=2) as tab_step_3:
                gr.Markdown("## Step 3: Review and Edit Captions")
                with gr.Row():
                    gallery = gr.Gallery(label="Images", columns=4, height=600, allow_preview=False)
                    with gr.Column():
                        preview_image = gr.Image(label="Selected Image", type="filepath", interactive=False)
                        editor_caption = gr.Textbox(label="Caption", lines=5)
                        save_entry_btn = gr.Button("Save Caption Only", variant="primary")
                        save_status = gr.Textbox(label="Save Status", interactive=False)
                        current_path_state = gr.State()
                
                with gr.Accordion("🛠️ Bulk Caption Tools", open=False):
                    with gr.Row():
                        find_box = gr.Textbox(label="Find", placeholder="Text to find")
                        replace_box = gr.Textbox(label="Replace", placeholder="Replacement text")
                        replace_btn = gr.Button("Replace All")
                    with gr.Row():
                        tag_box = gr.Textbox(label="Tag / Text", placeholder="Text to add")
                        prepend_btn = gr.Button("Prepend to All")
                        append_btn = gr.Button("Append to All")
                    bulk_status = gr.Textbox(label="Bulk Status")
                
                with gr.Row():
                    back_btn_3 = gr.Button("< Back")
                    next_btn_3 = gr.Button("Next >")

            # STEP 4
            with gr.TabItem("Step 4: Image Tools", id=3) as tab_step_4:
                gr.Markdown("## Step 4: Masking (SAM 2.1)")
                img_selector = gr.Dropdown(label="Select Image to Edit")
                
                with gr.Row():
                    # Interactive image for clicking
                    sam_input_image = gr.Image(label="Click Object to Mask", type="filepath", interactive=True)
                    # Mask output preview
                    sam_mask_output = gr.Image(label="Generated Mask", type="pil", interactive=False)
                
                with gr.Row():
                    mode_toggle_btn = gr.Button("🎯 Switch to Background Mode")
                    undo_btn = gr.Button("Undo Last Point")
                    reset_btn = gr.Button("Reset All")
                    unload_btn = gr.Button("🗑️ Unload Model")
                    save_mask_btn = gr.Button("Save Mask", variant="primary")
                
                tool_status = gr.Textbox(label="Status", value="Ready.")
                
                # State for SAM
                points_state = gr.State([])
                labels_state = gr.State([])
                bg_mode_state = gr.State(False) # False = Foreground, True = Background

                with gr.Row():
                    back_btn_4 = gr.Button("< Back")
                    next_btn_4 = gr.Button("Next >")

            # STEP 5
            with gr.TabItem("Step 5: Export", id=4) as tab_step_5:
                gr.Markdown("## Step 5: Project Finalized")
                gr.Markdown("Your files have been saved alongside your images.")
                final_status = gr.JSON(label="Current Session Stats")
                refresh_btn = gr.Button("Refresh Stats")
                back_btn_5 = gr.Button("< Back")

    # --- EVENT BINDINGS (Wired at the end to ensure all components exist) ---
    
    # Step 1
    browse_btn.click(browse_directory, outputs=input_dir)
    scan_btn.click(scan_action, inputs=input_dir, outputs=[scan_output, next_btn_1])
    next_btn_1.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    
    # Step 2
    caption_btn.click(run_captioning_action, inputs=model_dropdown, outputs=progress_bar)
    back_btn_2.click(lambda: gr.Tabs(selected=0), outputs=tabs)
    next_btn_2.click(lambda: (gr.Tabs(selected=2), global_state.image_paths), outputs=[tabs, gallery])
    
    # Step 3
    tab_step_3.select(lambda: global_state.image_paths, outputs=gallery)
    gallery.select(on_select_image, outputs=[preview_image, editor_caption, current_path_state])
    save_entry_btn.click(save_caption_action, inputs=[current_path_state, editor_caption], outputs=save_status)
    replace_btn.click(bulk_replace_action, inputs=[find_box, replace_box], outputs=bulk_status)
    prepend_btn.click(lambda t: bulk_add_action(t, "prepend"), inputs=tag_box, outputs=bulk_status)
    append_btn.click(lambda t: bulk_add_action(t, "append"), inputs=tag_box, outputs=bulk_status)
    back_btn_3.click(lambda: gr.Tabs(selected=1), outputs=tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=3), outputs=tabs)

    # Step 4
    tab_step_4.select(lambda: gr.update(choices=global_state.image_paths), outputs=img_selector)
    
    img_selector.change(
        on_sam_image_change, 
        inputs=img_selector, 
        outputs=[sam_input_image, sam_mask_output, points_state, labels_state, tool_status]
    )
    
    sam_input_image.select(
        on_sam_click,
        inputs=[img_selector, points_state, labels_state, bg_mode_state],
        outputs=[sam_mask_output, points_state, labels_state, tool_status]
    )
    
    mode_toggle_btn.click(
        on_toggle_mode,
        inputs=bg_mode_state,
        outputs=[bg_mode_state, mode_toggle_btn]
    )
    
    undo_btn.click(
        on_sam_undo,
        inputs=[img_selector, points_state, labels_state],
        outputs=[sam_mask_output, points_state, labels_state, tool_status]
    )
    
    reset_btn.click(
        on_sam_reset,
        outputs=[sam_mask_output, points_state, labels_state, tool_status]
    )

    unload_btn.click(
        on_sam_unload,
        outputs=[sam_mask_output, points_state, labels_state, tool_status]
    )
    
    save_mask_btn.click(
        on_sam_save,
        inputs=[img_selector, sam_mask_output],
        outputs=tool_status
    )
    
    back_btn_4.click(lambda: gr.Tabs(selected=2), outputs=tabs)
    next_btn_4.click(lambda: gr.Tabs(selected=4), outputs=tabs)

    # Step 5
    def get_stats():
        return {
            "Images Found": len(global_state.image_paths),
            "Captions Loaded/Created": len(global_state.captions),
            "Masks Created": len(global_state.masks)
        }
    tab_step_5.select(get_stats, outputs=final_status)
    refresh_btn.click(get_stats, outputs=final_status)
    back_btn_5.click(lambda: gr.Tabs(selected=3), outputs=tabs)

    return wizard_container