import gradio as gr
from utils import colorize
from PIL import Image
import tempfile
import torch
import gc
def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def create_demo(model):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model == {}:
        model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil', elem_id='img-display-input').style(height="auto")
        depth_image = gr.Image(label="Depth Map", elem_id='img-display-output')
    raw_file = gr.File(label="16-bit raw depth, multiplier:256")
    submit = gr.Button("Submit")

    def on_submit(image):

        depth = predict_depth(model, image)
        colored_depth = colorize(depth, cmap='gray_r')
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth = Image.fromarray((depth*256).astype('uint16'))
        raw_depth.save(tmp.name)
        del depth
        del raw_depth
        torch.cuda.empty_cache()
        gc.collect
        return [colored_depth, tmp.name]
    
    submit.click(on_submit, inputs=[input_image], outputs=[depth_image, raw_file])
#    examples = gr.Examples(examples=["examples/person_1.jpeg", "examples/person_2.jpeg", "examples/person-leaves.png", "examples/living-room.jpeg"],
#                           inputs=[input_image])