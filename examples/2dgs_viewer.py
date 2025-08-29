import argparse
import math
import os
import time
from typing import Tuple, Dict

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm
import viser
from pathlib import Path
from textured_gaussians._helper import load_test_data
from textured_gaussians.distributed import cli
from textured_gaussians.rendering import rasterization, rasterization_2dgs, rasterization_textured_gaussians
from util_viewer import UtilViewer

from nerfview import CameraState
import nerfview


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    
    means, quats, scales, opacities, sh0, shN, textures = [], [], [], [], [], [], []
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
        # textures.append(ckpt["textures"])
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    # textures = torch.cat(textures, dim=0)
    print("Number of Gaussians:", len(means))

    # register and open viewer
    def viewer_render_fn(
        camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState
    ):
        """Callable function for the viewer."""
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        # convert to float32
        c2w = torch.tensor(camera_state.c2w).to(device, dtype=torch.float32)
        K = torch.tensor(camera_state.get_K([width, height])).to(device, dtype=torch.float32)

        render_colors, *_ = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=sh_degree
        )

        return render_colors[0].cpu().detach().numpy()

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = UtilViewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    server.gui.set_panel_label("2dgs viewer")

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
