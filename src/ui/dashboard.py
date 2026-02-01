import gradio as gr
from src.core.state import global_state

def render_dashboard():
    with gr.Column():
        gr.Markdown("# Advanced Dashboard")
        gr.Markdown("Direct access to tools (Coming Soon). For now, please use the Wizard.")
        
        with gr.Tabs():
            with gr.TabItem("Project Inspection"):
                state_json = gr.JSON(label="Current State")
                inspect_btn = gr.Button("Inspect Project")
                
                def get_project_state():
                    return {
                        "directory": global_state.source_directory,
                        "images_found": len(global_state.image_paths),
                        "captions_generated": len(global_state.captions)
                    }
                
                inspect_btn.click(get_project_state, outputs=state_json)
                
            with gr.TabItem("Tools"):
                gr.Button("Coming Soon: Bulk Renamer")
                gr.Button("Coming Soon: Tag Manager")