import argparse
import math
import os
import time
import random
from typing import Tuple, Dict

import imageio
import plotly.express as px
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm
import viser
from pathlib import Path
from textured_gaussians._helper import load_test_data
from textured_gaussians.distributed import cli
from textured_gaussians.rendering import rasterization, rasterization_2dgs, rasterization_textured_gaussians, rasterization_packed_textured_gaussians

import nerfview

class CustomViewer(nerfview.Viewer):
    def __init__(self, init_dict, callback_dict, plots, plots_checkboxes, *args, **kwargs):
        self.callback_dict = callback_dict
        self.init_dict = init_dict
        self.plots = plots
        self.plots_checkboxes = plots_checkboxes
        super().__init__(*args, **kwargs)

    def _init_rendering_tab(self):
        super()._init_rendering_tab()
        self._custom_render_handles = {}
        self._custom_rendering_folder = self.server.gui.add_folder("Custom Rendering")

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
            for name in list(self.plots.keys()):
                checkbox = self.server.gui.add_checkbox(label=name, initial_value=False)
                plot = self.server.gui.add_plotly(figure=fig, aspect=1, visible=False)
                make_on_update_checkbox(checkbox=checkbox, plot=plot)
                self.plots[name] = plot
                self.plots_checkboxes[name] = checkbox

            for i, (callback, initial_value) in enumerate(zip(self.callback_dict["sliders"], self.init_dict["sliders"])):
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

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)

    device = torch.device("cuda", local_rank)
    
    num_ckpts = len(args.ckpt)

    slider_positions = [(i+1)*(1/num_ckpts) for i in range(num_ckpts-1)]

    means, quats, scales, opacities, sh0, shN, textures = [], [], [], [], [], [], []
    textures_packed, texture_dims_list, texture_offsets_list = [], [], []
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
        texture = ckpt["textures"]
        textures.append(texture)

        # Convert textures to packed format
        # [N, W, H, C] -> [C, N*W*H]
        # Construct the corresponding texture dimensions [N, 2] and offsets [N, 1]
        texture_packed = texture.permute(3, 0, 1, 2).reshape(texture.shape[3], -1)
        textures_packed.append(texture_packed)

        N, W, H, C = texture.shape

        # Texture dimensions: [N, 2] with [W, H]
        dims = torch.tensor([[W, H]] * N, device=texture.device, dtype=torch.int32)
        texture_dims_list.append(dims)

        # Offsets: [N, 1] where each is cumulative sum of previous W*H
        areas = dims[:, 0] * dims[:, 1]  # W * H for each texture -> [N]
        offsets = torch.zeros_like(areas)
        offsets[1:] = torch.cumsum(areas, dim=0)[:-1]
        texture_offsets_list.append(offsets.unsqueeze(1))  # Make shape [N, 1]

        

    colors = [None] * num_ckpts
    for i in range(num_ckpts):
        colors[i] = torch.cat([sh0[i], shN[i]], dim=-2)
        sh_degree = int(math.sqrt(colors[i].shape[-2]) - 1)
    
    print("Number of Gaussians:", len(means))

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
            # render_colors, _, _, _, _, _, _, _, _ = rasterization_textured_gaussians(
            #     means=means[i],
            #     quats=quats[i],
            #     scales=scales[i],
            #     opacities=opacities[i],
            #     colors=colors[i],
            #     textures=textures[i],
            #     viewmats=torch.linalg.inv(c2w[None]),
            #     Ks=K[None],
            #     width=width,
            #     height=height,
            #     sh_degree=sh_degree
            # )
            render_colors, *_, gs_contrib_sum, gs_contrib_count, gs_weight_sum, gs_dx_sum, gs_dy_sum, meta, = rasterization_packed_textured_gaussians(
                means=means[i],
                quats=quats[i],
                scales=scales[i],
                opacities=opacities[i],
                colors=colors[i],
                textures=None,
                textures_packed=textures_packed[i],
                texture_dims=texture_dims_list[i],
                texture_offsets=texture_offsets_list[i],
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

        def get_contrib_sum_plot(histc_input, title="gs_contrib_sum", min_val=1.0, max_val=5000, bins=100):
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

    def make_update_slider(idx: int):
        def update_slider(value: float):
            slider_positions[idx] = value
        return update_slider

    slider_callbacks = [make_update_slider(i) for i in range(num_ckpts-1)]
    slider_inits = slider_positions

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = CustomViewer(
        init_dict={
            "sliders": slider_inits,
        },
        callback_dict={
            "sliders": slider_callbacks,
        },
        plots=plots,
        plots_checkboxes=plots_checkboxes,
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


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
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
