import argparse
import math
import os
import time
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

import imageio
import plotly.express as px
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm
import viser
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp
from datasets.colmap import Dataset, Parser, BlenderDataset
from textured_gaussians._helper import load_test_data
from textured_gaussians.distributed import cli
from textured_gaussians.rendering import rasterization, rasterization_2dgs, rasterization_textured_gaussians, rasterization_packed_textured_gaussians
from util_viewer import UtilViewer

import nerfview

@dataclass
class TexturedGaussiansModel:
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    textures: Tensor
    sh0: Tensor
    shN: Tensor
    textures_packed: Tensor
    texture_dims: Tensor
    texture_offsets: Tensor

textured_gaussian_models: List[TexturedGaussiansModel] = []

slider_positions: List[float] = []
slider_callbacks: List = []

plots = {
    'gs_contrib_sum': None,
    'gs_contrib_count': None,
    'gs_weight_sum': None,
    'gs_dx_sum': None,
    'gs_dy_sum': None
}

plots_checkboxes = {
    'gs_contrib_sum': None,
    'gs_contrib_count': None,
    'gs_weight_sum': None,
    'gs_dx_sum': None,
    'gs_dy_sum': None
}

debug_train_image_gt: np.array = None
debug_train_image_render: np.array = None
debug_model_idx = 0
debug_val_image_gt: np.array = None
debug_val_image_render: np.array = None
render_train: callable = None
render_val: callable = None
gt_train: callable = None
gt_val: callable = None

class CustomViewer(UtilViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_rendering_tab(self):
        super()._init_rendering_tab()
        self._custom_render_handles = {}
        self._custom_rendering_folder = self.server.gui.add_folder("Custom Rendering")
        self.train_plot_handle = None
        self.train_plot_handle = None

    def update_frustum_callback(self):
        def train_frustum_render_callback(frustum: viser.CameraFrustumHandle, idx):
            @frustum.on_click
            def _(_) -> None:
                global debug_train_image_gt
                debug_train_image_gt = gt_train(idx) / 255.0
                debug_train_image_gt = np.flip(debug_train_image_gt, axis=(0))

                global debug_train_image_render
                debug_train_image_render = np.clip(render_train(idx), 0.0, 1.0)
                debug_train_image_render = np.flip(debug_train_image_render, axis=(0))

                diff = np.abs(debug_train_image_gt - debug_train_image_render)
                image_stack = np.array([debug_train_image_gt, debug_train_image_render, np.sqrt(diff)])
                # Create subplot figure
                fig = px.imshow(image_stack, facet_col_wrap=2, facet_col=0)

                self.train_plot_handle.figure = fig

        for i, frustum in enumerate(self.train_frustums):
            train_frustum_render_callback(frustum=frustum, idx=i)

        def val_frustum_render_callback(frustum: viser.CameraFrustumHandle, idx):
            @frustum.on_click
            def _(_) -> None:
                global debug_val_image_gt
                debug_val_image_gt = gt_val(idx) / 255.0
                debug_val_image_gt = np.flip(debug_val_image_gt, axis=(0))

                global debug_val_image_render
                debug_val_image_render = np.clip(render_val(idx), 0.0, 1.0)
                debug_val_image_render = np.flip(debug_val_image_render, axis=(0))

                diff = np.abs(debug_val_image_gt - debug_val_image_render)
                image_stack = np.array([debug_val_image_gt, debug_val_image_render, np.sqrt(diff)])
                # Create subplot figure
                fig = px.imshow(image_stack, facet_col_wrap=2, facet_col=0)

                self.test_plot_handle.figure = fig

        for i, frustum in enumerate(self.val_frustums):
            val_frustum_render_callback(frustum=frustum, idx=i)

    def _populate_rendering_tab(self):
        super()._populate_rendering_tab()

        with self._custom_rendering_folder:

            def make_on_update(callback, slider):
                @slider.on_update
                def on_update(_) -> None:
                    callback(slider.value)
                    self.rerender(_)

            def make_on_update_checkbox(checkbox, plot):
                @checkbox.on_update
                def on_update(_) -> None:
                    plot.visible = checkbox.value
                    self.rerender(_)
            x = torch.randn(1000, 1)
            x_np = x.view(-1).numpy()
            fig = px.histogram(x_np, nbins=30, title="Histogram of Tensor Values")
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))

            self.server.gui.add_markdown("### Plots")
            for name in list(plots.keys()):
                checkbox = self.server.gui.add_checkbox(label=name, initial_value=False)
                plot = self.server.gui.add_plotly(figure=fig, aspect=1, visible=False)
                make_on_update_checkbox(checkbox=checkbox, plot=plot)
                plots[name] = plot
                plots_checkboxes[name] = checkbox

            for i, (callback, initial_value) in enumerate(zip(slider_callbacks, slider_positions)):
                slider = self.server.gui.add_slider(
                    f"Slider {i+1}",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    initial_value=initial_value,
                    hint=f"Adjust the position of segment {i+1}"
                )

                make_on_update(callback, slider)

                self._custom_render_handles[f"Slider {i+1}"] = slider

            self.train_plot_handle = self.server.gui.add_plotly(
                figure=go.Figure()
            )
            self.val_plot_handle = self.server.gui.add_plotly(
                figure=go.Figure()
            )


def main(local_rank: int, world_rank, world_size: int, args):
    global textured_gaussian_models, slider_positions, slider_callbacks
    torch.manual_seed(42)

    device = torch.device("cuda", local_rank)
    
    num_ckpts = len(args.ckpt)

    slider_positions.clear()
    slider_positions.extend([(i+1)*(1/num_ckpts) for i in range(num_ckpts-1)])
    parser = None

    trainset = None
    valset = None

    # Load data: Training data should contain initial points and colors.
    if args.dataset == "colmap":
        parser = Parser(
            data_dir=args.data_dir,
            factor=args.data_factor,
            normalize=True,
            test_every=args.test_every,
        )
        trainset = Dataset(
            parser,
            split="train",
        )
        valset = Dataset(parser, split="val")
    elif args.dataset == "blender":
        parser = None
        trainset = BlenderDataset(data_dir=args.data_dir, split="train")
        valset = BlenderDataset(data_dir=args.data_dir, split="val")
    else:
        raise ValueError(f"Dataset mode {args.dataset} not supported!")

    # Clear any existing models
    textured_gaussian_models.clear()
    
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means = ckpt["means"]
        quats = F.normalize(ckpt["quats"], p=2, dim=-1)
        scales = torch.exp(ckpt["scales"])
        opacities = torch.sigmoid(ckpt["opacities"])
        sh0 = ckpt["sh0"]
        shN = ckpt["shN"]
        textures = ckpt["textures"]

        # Convert textures to packed format
        # [N, W, H, C] -> [C, N*W*H]
        # Construct the corresponding texture dimensions [N, 2] and offsets [N, 1]
        textures_packed = textures.permute(3, 0, 1, 2).reshape(textures.shape[3], -1)

        N, W, H, C = textures.shape

        # Texture dimensions: [N, 2] with [W, H]
        texture_dims = torch.tensor([[W, H]] * N, device=textures.device, dtype=torch.int32)

        # Offsets: [N, 1] where each is cumulative sum of previous W*H
        areas = texture_dims[:, 0] * texture_dims[:, 1]  # W * H for each texture -> [N]
        texture_offsets = torch.zeros_like(areas)
        texture_offsets[1:] = torch.cumsum(areas, dim=0)[:-1]
        texture_offsets = texture_offsets.unsqueeze(1)  # Make shape [N, 1]

        colors = torch.cat([sh0, shN], dim=-2)
        
        # Create TexturedGaussiansModel instance and add to global list
        model = TexturedGaussiansModel(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            textures=textures,
            sh0=sh0,
            shN=shN,
            textures_packed=textures_packed,
            texture_dims=texture_dims,
            texture_offsets=texture_offsets
        )
        textured_gaussian_models.append(model)

    sh_degree = int(math.sqrt(textured_gaussian_models[0].colors.shape[-2]) - 1)
    
    print("Number of Gaussians:", len(textured_gaussian_models))

    # register and open viewer
    def viewer_render_fn(
        camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState
    ):
        """Callable function for the viewer."""
        # if render_tab_state.preview_render:
        #     width = render_tab_state.render_width
        #     height = render_tab_state.render_height
        # else:
            # width = render_tab_state.viewer_width
            # height = render_tab_state.viewer_height

        width = render_tab_state.render_width
        height = render_tab_state.render_height
        
        # convert to float32
        c2w = torch.tensor(camera_state.c2w).to(device, dtype=torch.float32)
        K = torch.tensor(camera_state.get_K([width, height])).to(device, dtype=torch.float32)

        render_images = [None] * num_ckpts
        metrics = {
            "gs_contrib_sum": [None] * num_ckpts,
            "gs_contrib_count": [None] * num_ckpts,
            "gs_weight_sum": [None] * num_ckpts,
            "gs_dx_sum": [None] * num_ckpts,
            "gs_dy_sum": [None] * num_ckpts
        }
        for i in range(num_ckpts):
            model = textured_gaussian_models[i]
            render_colors, *_, gs_contrib_sum, gs_contrib_count, gs_weight_sum, gs_dx_sum, gs_dy_sum, meta, = rasterization_packed_textured_gaussians(
                means=model.means,
                quats=model.quats,
                scales=model.scales,
                opacities=model.opacities,
                colors=model.colors,
                textures=None,
                textures_packed=model.textures_packed,
                texture_dims=model.texture_dims,
                texture_offsets=model.texture_offsets,
                viewmats=torch.linalg.inv(c2w[None]),
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=sh_degree
            )
            render_images[i] = render_colors
            metrics['gs_contrib_count'][i] = gs_contrib_count
            metrics['gs_contrib_sum'][i] = gs_contrib_sum
            metrics['gs_weight_sum'][i] = gs_weight_sum
            metrics['gs_dx_sum'][i] = gs_dx_sum
            metrics['gs_dy_sum'][i] = gs_dy_sum

        def get_contrib_sum_plot(histc_input, title="gs_contrib_sum", min_val=1.0, max_val=5000, bins=1000):
            with torch.no_grad():
                hist = torch.histc(histc_input, bins=bins, min=min_val, max=max_val).cpu().detach().numpy()

            bin_edges = torch.linspace(min_val, max_val, steps=bins + 1).cpu().numpy()
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            fig = px.histogram(
                x=bin_centers,
                y=hist,
                nbins=bins,
                labels={'x': 'Value', 'y': 'Count'},
                title=title
            )
            return fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        
        for name, plot in plots.items():
            value = plots_checkboxes[name].value
            if plots_checkboxes[name].value:
                plot.figure = get_contrib_sum_plot(metrics[name][0], name)

        # All images must have the same shape
        H, W, C = render_images[0][0].shape
        final_image = torch.zeros_like(render_images[0][0])

        # Compute width per segment

        for i in range(num_ckpts):
            # Calculate the start and end indices based on the slider position
            start = int(slider_positions[i-1] * W) if i != 0 else 0
            end = int(slider_positions[i] * W) if i != num_ckpts - 1 else W
            final_image[:, start:end] = render_images[i][0][:, start:end]

        # Optional: draw separator lines between segments
        for pos in slider_positions:
            x_pos = int(pos * W)
            final_image[:, x_pos - 1:x_pos + 1] = 1.0

        return final_image.cpu().detach().numpy()

    def train_render_fn(idx):
        data = trainset[idx]
        K = data['K'].cuda()
        c2w = data['camtoworld'].cuda()
        image = data['image'].cuda()
        h, w = image.shape[0], image.shape[1]
        model = textured_gaussian_models[debug_model_idx]
        render_colors, *_ = rasterization_packed_textured_gaussians(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.colors,
            textures=None,
            textures_packed=model.textures_packed,
            texture_dims=model.texture_dims,
            texture_offsets=model.texture_offsets,
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=w,
            height=h,
            sh_degree=sh_degree
        )
        return render_colors.squeeze(0).detach().cpu().numpy()
    
    global render_train
    render_train = train_render_fn

    def val_render_fn(idx):
        data = valset[idx]
        K = data['K']
        c2w = data['camtoworld']
        image = data['image']
        h , w = image.shape[0], image.shape[1]
        model = textured_gaussian_models[debug_model_idx]
        render_colors, *_ = rasterization_packed_textured_gaussians(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.colors,
            textures=None,
            textures_packed=model.textures_packed,
            texture_dims=model.texture_dims,
            texture_offsets=model.texture_offsets,
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=w,
            height=h,
            sh_degree=sh_degree
        )
        return render_colors.squeeze(0).detach().cpu().numpy()
    global render_val
    render_val = val_render_fn

    def train_gt_fn(idx):
        return trainset[idx]['image'].detach().cpu().numpy()
    global gt_train
    gt_train = train_gt_fn

    def val_gt_fn(idx):
        return valset[idx]['image'].detach().cpu().numpy()
    global gt_val
    gt_val = val_gt_fn

    def make_update_slider(idx: int):
        def update_slider(value: float):
            slider_positions[idx] = value
        return update_slider

    slider_callbacks.clear()
    slider_callbacks.extend([make_update_slider(i) for i in range(num_ckpts-1)])

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = CustomViewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    viewer.custom_update(train_dataset=trainset, val_dataset=valset)
    viewer.update_frustum_callback()
    # cam0 = trainset[0]
    # K = cam0["K"]
    # image = cam0['image']
    # fx = K[0, 0].detach().cpu().numpy()
    # fy = K[1, 1].detach().cpu().numpy()
    # fov_x = 2 * np.arctan(image.shape[0] / (2 * fx))
    # fov_y = 2 * np.arctan(image.shape[1] / (2 * fy))
    # print(f"fov_x: {fov_x}")
    # print(f"fov_y: {fov_y}")
    # for client in viewer.server.get_clients().values():
    #     client.camera.fov = fov_x
    while True:
        time.sleep(1e-3)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    parser.add_argument(
        "--dataset",
        type=str,
        default="none",
        choices=["colmap", "blender", "none"],
        help="dataset to use",
    )
    parser.add_argument(
        "--data_dir", type=str, default="", help="path to the dataset"
    )
    parser.add_argument(
        "--data_factor", type=int, default=1, help="factor to downsample the dataset"
    )
    parser.add_argument(
        "--test_every", type=int, default=8, help="split dataset"
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
