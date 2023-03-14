import gradio as gr
import numpy as np
import trimesh
from geometry import create_triangles
from functools import partial
import tempfile
import torch
import gc
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


def pano_depth_to_world_points(depth):
    """
    360 depth to world points
    given 2D depth is an equirectangular projection of a spherical image
    Treat depth as radius

    longitude : -pi to pi
    latitude : -pi/2 to pi/2
    """

    # Convert depth to radius
    radius = depth.flatten()

    lon = np.linspace(-np.pi, np.pi, depth.shape[1])
    lat = np.linspace(-np.pi/2, np.pi/2, depth.shape[0])

    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()

    # Convert to cartesian coordinates
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    pts3d = np.stack([x, y, z], axis=1)

    return pts3d

def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def get_mesh(model, image, keep_edges=False):
    image.thumbnail((1024,1024))  # limit the size of the image
    depth = predict_depth(model, image)
    pts3d = pano_depth_to_world_points(depth)

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
    print(glb_path)
    mesh.export(glb_path)
    mesh.export(f'./extensions/a1111-sd-zoe-depth/temp.glb')
    del mesh
    del depth
    del image
    del triangles
    torch.cuda.empty_cache()
    gc.collect
    return glb_path

def convert_mesh():
    temp_mesh = f'./extensions/a1111-sd-zoe-depth/temp.glb'
    return temp_mesh

def create_demo(model):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model == {}:
        model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()
    gr.Markdown("### Panorama to 3D mesh")
    gr.Markdown("Convert a 360 spherical panorama to a 3D mesh")
    gr.Markdown("ZoeDepth was not trained on panoramic images. It doesn't know anything about panoramas or spherical projection. Here, we just treat the estimated depth as radius and some projection errors are expected. Nonetheless, ZoeDepth still works surprisingly well on 360 reconstruction.")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil')
        result = gr.Model3D(label="3d mesh reconstruction", clear_color=[
                                                 1.0, 1.0, 1.0, 1.0])
    checkbox = gr.Checkbox(label="Keep occlusion edges", value=True)
    submit = gr.Button("Submit")
    submit.click(partial(get_mesh, model), inputs=[input_image, checkbox], outputs=[result])
    download = gr.Button("Download")
    download.click(convert_mesh,outputs=[gr.File(label="3d glb")])
#    examples = gr.Examples(examples=["examples/pano_1.jpeg", "examples/pano_2.jpeg", "examples/pano_3.jpeg"],
#                            inputs=[input_image])