import gradio as gr
import numpy as np
import trimesh
from geometry import depth_to_points, create_triangles
from functools import partial
import tempfile


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def get_mesh(model, image, keep_edges=False):
    image.thumbnail((1024,1024))  # limit the size of the input image
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)

    # Create a trimesh mesh from the points
    # Each pixel is connected to its 4 neighbors
    # colors are the RGB values of the image

    verts = pts3d.reshape(-1, 3)
    image = np.array(image)
    if keep_edges:
        triangles = create_triangles(image.shape[0], image.shape[1])
    else:
        triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
    colors = image.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    # Save as glb
    glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    glb_path = glb_file.name
    mesh.export(glb_path)
    return glb_path

def create_demo(model):

    gr.Markdown("### Image to 3D mesh")
    gr.Markdown("Convert a single 2D image to a 3D mesh")

    with gr.Row():
        image = gr.Image(label="Input Image", type='pil')
        result = gr.Model3D(label="3d mesh reconstruction", clear_color=[
                                                 1.0, 1.0, 1.0, 1.0])
    
    checkbox = gr.Checkbox(label="Keep occlusion edges", value=False)
    submit = gr.Button("Submit")
    submit.click(partial(get_mesh, model), inputs=[image, checkbox], outputs=[result])
#    examples = gr.Examples(examples=["examples/aerial_beach.jpeg", "examples/mountains.jpeg", "examples/person_1.jpeg", "examples/ancient-carved.jpeg"],
#                            inputs=[image])

