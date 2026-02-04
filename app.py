from PIL import Image
# Raise decompression bomb limit for large training images (~17,000 x 17,000 px)
Image.MAX_IMAGE_PIXELS = 300_000_000

import gradio as gr
import os
from src.ui.wizard import render_wizard
from src.ui.dashboard import render_dashboard

with gr.Blocks(title="Diffusion Dataset Creator") as app:
    gr.Markdown("# 🎨 Diffusion Dataset Creator")
    
    with gr.Tabs():
        with gr.TabItem("Wizard Mode (Recommended)"):
            render_wizard()
        
        with gr.TabItem("Advanced Dashboard"):
            render_dashboard()

if __name__ == "__main__":
    # Ensure cache directories are local and writable
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # Allow access to the root directory so files from external drives/mnt are also accessible
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        allowed_paths=["/"],
        share=False,
        theme=gr.themes.Soft()
    )
