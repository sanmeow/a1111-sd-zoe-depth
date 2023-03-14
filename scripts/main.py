import gradio as gr
import gc
from gradio_depth_pred import create_demo as create_depth_pred_demo
from gradio_im_to_3d import create_demo as create_im_to_3d_demo
from gradio_pano_to_3d import create_demo as create_pano_to_3d_demo
import torch
import modules.extras
import modules.ui
from modules.shared import opts, cmd_opts
from modules import shared, scripts
from modules import script_callbacks
from modules import extensions

'''
UI part
'''
css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }
    
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
title = "# ZoeDepth"

model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

title = "# ZoeDepth"

def unload_models(model,log: bool = True):
    if log:
        print("Unloading models...")
    del model
    torch.cuda.empty_cache()
    gc.collect
    if log:
        print("Done. Models unloaded")

def add_tab():
    print('add tab')
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(title)
        with gr.Tab("Depth Prediction"):
            create_depth_pred_demo(model)
        with gr.Tab("Image to 3D"):
            create_im_to_3d_demo(model)
        with gr.Tab("360 Panorama to 3D"):
            create_pano_to_3d_demo(model)
        unload_models_btn = gr.Button(value="Unload models", variant="secondary")
        unload_models_btn.click(
        fn=unload_models,
        inputs=[],
        outputs=[],)

    return [(ui, "ZoeDepth", "ZoeDepth")]

script_callbacks.on_ui_tabs(add_tab)