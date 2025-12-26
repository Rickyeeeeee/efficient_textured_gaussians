import argparse
import math
import os
import time
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import shutil
import seaborn as sns

import imageio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm
import viser
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp
from utils import rgb_to_sh
from datasets.colmap import Dataset, Parser, BlenderDataset
from textured_gaussians._helper import load_test_data
from textured_gaussians.distributed import cli
from textured_gaussians.rendering import rasterization, rasterization_2dgs, rasterization_textured_gaussians, rasterization_packed_textured_gaussians
from util_viewer import UtilViewer

import nerfview

def plot_texture_shape_barchart_pct(
        data: np.ndarray,
        save_path: str,                   # ".png" / ".svg" / ".pdf"
        sort_mode: str = "area_then_ratio",
        # size control
        width_px: int | None = None,
        height_px: int | None = None,
        dpi: int = 200,
        figsize=(11, 5),                  # inches if width_px/height_px not given
        # LaTeX styling
        use_latex: str | bool = "auto",   # "auto" | True | False
        latex_preamble: str | None = None,
        show_legend: bool = True,
        decimals: int = 1
    ):
    """
    Save a LaTeX-styled 2D bar chart (matplotlib + seaborn) of PERCENTAGES of texture shapes (W×H),
    merging (w,h) and (h,w). Squares are blue; non-squares are red. The 1×1 bar is hatched.

    Assumes these are already imported:
      numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
      from matplotlib.ticker import PercentFormatter, and shutil
    """
    # near the top of the function, after rc_context:
    mpl.rcParams.update({"font.size": 142, "axes.titlesize": 14, "axes.labelsize": 14})

    # ----- input checks -----
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")
    if not save_path:
        raise ValueError("save_path must be provided")

    # ----- figure size (inches) -----
    if (width_px is not None) and (height_px is not None):
        figsize_in = (width_px / dpi, height_px / dpi)
    elif (width_px is not None) != (height_px is not None):
        raise ValueError("Provide both width_px and height_px, or neither.")
    else:
        figsize_in = figsize

    # ----- canonicalize (w,h) == (h,w), count, % -----
    W = np.rint(data[:, 0]).astype(int)
    H = np.rint(data[:, 1]).astype(int)
    A = np.minimum(W, H); B = np.maximum(W, H)
    uniq, counts = np.unique(np.stack([A, B], 1), axis=0, return_counts=True)
    total = counts.sum()
    if total == 0:
        raise ValueError("No data to plot (total count is zero).")

    # sort
    if sort_mode == "area_then_ratio":
        areas  = (uniq[:, 0] * uniq[:, 1]).astype(int)
        ratios = uniq[:, 1] / np.maximum(1, uniq[:, 0])
        order = np.lexsort((ratios, areas))
    else:
        order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    uniq, counts = uniq[order], counts[order]

    pct = counts / total * 100.0
    is_square = (uniq[:, 0] == uniq[:, 1])

    # math labels; try bold math via \boldsymbol (no \mathbf)
    labels_math = [rf"$\boldsymbol{{{w}\times{h}}}$" for w, h in uniq]

    df = pd.DataFrame({
        "shape": labels_math,
        "percentage": pct,
        "kind": np.where(is_square, "square", "non-square")
    })

    # ----- LaTeX / MathText style (bold, serif text) -----
    if use_latex == "auto":
        has_latex = shutil.which("latex") is not None
        use_tex = has_latex
    else:
        use_tex = bool(use_latex)

    if latex_preamble is None:
        latex_preamble = r"\usepackage{amsmath}\usepackage{bm}"

    rc_tex = {
        # Non-equation text: bold serif
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.unicode_minus": False,
    }
    if use_tex:
        rc_tex.update({
            "text.usetex": True,
            "text.latex.preamble": latex_preamble,
        })
    else:
        rc_tex.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",   # Computer Modern math
            "mathtext.rm": "serif",
        })

    with mpl.rc_context(rc_tex):
        # Use a clean background (no whitegrid); we'll also hard-disable grid lines.
        sns.set_theme(style="white", rc={
            "font.family": "serif",
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        })

        fig, ax = plt.subplots(figsize=figsize_in, dpi=dpi)

        palette = {"square": "blue", "non-square": "red"}
        sns.barplot(
            data=df,
            x="shape", y="percentage", hue="kind",
            dodge=False, palette=palette, ax=ax
        )
        xticks = ax.get_xticks()

        # --- Remove background grid/level lines (y-percentage guide lines) ---
        ax.grid(False)                        # fully disable grid
        ax.yaxis.grid(False)                  # just to be explicit
        ax.set_axisbelow(False)               # make sure nothing sneaks behind

        # percent axis and limits
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=decimals))
        ymax = float(df["percentage"].max())
        ax.set_ylim(0, min(100.0, ymax * 1.15 if ymax > 0 else 5.0))

        # Titles/labels (bold by rcParams)
        fig.subplots_adjust(top=0.98)
        ax.set_xlabel(r"Texture Shape ($W\times H$)")
        ax.set_ylabel(r"Percentage")

        # xtick labels are math strings; rotate and make sure they appear bold
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels_math, rotation=30, ha="right")
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")

        # ytick labels bold too
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")

        # Legend
        if show_legend:
            leg = ax.legend(title="", frameon=False)
            for txt in leg.get_texts():
                txt.set_fontweight("bold")
        else:
            ax.get_legend().remove()

        # annotate bars with bold percentages (no \mathbf)
        for p in ax.patches:
            y = p.get_height()
            if y > 0:
                ax.annotate(
                    rf"{y:.{decimals}f}%",
                    (p.get_x() + p.get_width() / 2, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold"
                )

    # --- Special styling for the 1×1 bar: hatch slashes with white face (looks unfilled) ---
    one_by_one_label = rf"$\boldsymbol{{1\times1}}$"
    xticks = ax.get_xticks()
    xlabels_now = [t.get_text() for t in ax.get_xticklabels()]
    if one_by_one_label in xlabels_now:
        idx = xlabels_now.index(one_by_one_label)
        target_x = xticks[idx]
        target_patch = None
        for p in ax.patches:
            xc = p.get_x() + p.get_width() / 2.0
            if abs(xc - target_x) < 1e-7:
                target_patch = p
                break
        if target_patch is not None:
            # white face = visually "no fill" on white background; hatch uses edgecolor
            target_patch.set_facecolor("white")
            target_patch.set_alpha(1.0)
            target_patch.set_edgecolor("blue")      # or palette["square"]
            target_patch.set_hatch("///")
            target_patch.set_linewidth(1.5)


        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


@dataclass
class TexturedGaussiansModel:
    """
    Dataclass to hold all the components of a textured Gaussian model.
    """
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

# Constants for histogram plotting for better readability and maintainability
HIST_MIN_VAL = 1.0
HIST_MAX_VAL = 5000
HIST_BINS = 1000

class RenderMode(Enum):
    RGB='RGB'
    GRAD='grad'
    TEX_SIZE='texture size'
    SH='sh'
    NO_TEX='no texture'
    NO_BASE='no base'
    SCALE='scale'
    TEX_ONLY='texture only'

class CustomViewer(UtilViewer):
    """
    Custom viewer class extending UtilViewer to integrate specific UI elements
    and rendering debug functionalities for textured Gaussians.
    """
    def __init__(self, *args, **kwargs):
        # Initialize instance-specific plot handles and debug image storage
        self.train_plot_handle: viser.gui.PlotlyHandle = None
        self.val_plot_handle: viser.gui.PlotlyHandle = None
        self.plots: Dict[str, viser.gui.PlotlyHandle] = {}
        self.plots_checkboxes: Dict[str, viser.gui.CheckboxHandle] = {}

        # Debug image storage for ground truth and rendered images
        self.debug_train_image_gt: np.array = None
        self.debug_train_image_render: np.array = None
        self.debug_val_image_gt: np.array = None
        self.debug_val_image_render: np.array = None
        self.debug_model_idx = 0 # Index of the model to use for debug image rendering

        # References to rendering functions, to be set by the main application logic
        self._render_train_fn: callable = None
        self._render_val_fn: callable = None
        self._gt_train_fn: callable = None
        self._gt_val_fn: callable = None

        self.render_mode = RenderMode.RGB
        self.tex_value: int = 0
        self.scale_x: bool = True
        self.scale_ratio = 10.0
        self.min_scale_value = 10.0
        self.texture_index_slider: viser.gui.SliderHandle | None = None
        self._texture_index_callback: callable | None = None

        self.scale_modifier_value = 1.0
        self.opacity_modifier_value = 1.0

        super().__init__(*args, **kwargs)

    def _init_rendering_tab(self):
        """Initializes the rendering tab and custom rendering folder."""
        super()._init_rendering_tab()
        self._custom_render_handles = {}
        self._visualization_folder = self.server.gui.add_folder("Visualization")
        self._custom_rendering_folder = self.server.gui.add_folder("Custom Rendering")

    def _populate_rendering_tab(self):
        """Populates the custom rendering tab with plot checkboxes and placeholders for sliders."""
        super()._populate_rendering_tab()
        with self._custom_rendering_folder:
            # Helper function to create an on_update callback for sliders
            def make_on_update(callback, slider):
                @slider.on_update
                def on_update_callback(_) -> None:
                    callback(slider.value)
                    self.rerender(_) # Trigger a main scene rerender when slider changes

            # Helper function to create an on_update callback for checkboxes
            def make_on_update_checkbox(checkbox, plot_handle):
                @checkbox.on_update
                def on_update_checkbox_callback(_) -> None:
                    plot_handle.visible = checkbox.value
                    # No main scene rerender needed for just plot visibility change

            # Initial dummy figure for plotly plots to be replaced later
            x = torch.randn(1000, 1)
            x_np = x.view(-1).numpy()
            dummy_fig = px.histogram(x_np, nbins=30, title="Histogram of Tensor Values")
            dummy_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))

            self.server.gui.add_markdown("### Plots")
            plot_names = ['gs_contrib_sum', 'gs_contrib_count', 'gs_weight_sum', 'gs_dx_sum', 'gs_dy_sum']
            for name in plot_names:
                checkbox = self.server.gui.add_checkbox(label=name, initial_value=False)
                plot = self.server.gui.add_plotly(figure=dummy_fig, aspect=1, visible=False)
                make_on_update_checkbox(checkbox=checkbox, plot_handle=plot)
                self.plots[name] = plot
                self.plots_checkboxes[name] = checkbox

            # Placeholders for train/val image plots
            self.train_plot_handle = self.server.gui.add_plotly(
                figure=go.Figure(),
                aspect=1.0, # Maintain aspect ratio
                visible=True # Initially visible for debugging
            )
            self.val_plot_handle = self.server.gui.add_plotly(
                figure=go.Figure(),
                aspect=1.0, # Maintain aspect ratio
                visible=True # Initially visible for debugging
            )
            self.texture_plot_handle = self.server.gui.add_plotly(
                figure=go.Figure(),
                aspect=1.0,
                visible=True
            )
            self.texture_show_handle = self.server.gui.add_plotly(
                figure=go.Figure(),
                aspect=1.0,
                visible=True
            )

        with self._visualization_folder:
            scale_modifier_slider = self.server.gui.add_slider(
                label='scale modifier',
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=1.0,
                marks=(1.0,)
            )
            @scale_modifier_slider.on_update
            def _(_):
                self.scale_modifier_value = scale_modifier_slider.value
                self.rerender(_)

            opacity_modifier_slider = self.server.gui.add_slider(
                label='opacity moidifier',
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=1.0,
                marks=(1.0,)
            )
            @opacity_modifier_slider.on_update
            def _(_):
                self.opacity_modifier_value = opacity_modifier_slider.value
                self.rerender(_)
            render_mode_dropdown = self.server.gui.add_dropdown(
                label='render mode',
                options=[
                    RenderMode.RGB,
                    RenderMode.NO_TEX,
                    RenderMode.NO_BASE,
                    RenderMode.TEX_SIZE,
                    RenderMode.TEX_ONLY,
                    RenderMode.GRAD,
                    RenderMode.SH,
                    RenderMode.SCALE
                ],
                initial_value=RenderMode.RGB
            )

            tex_slider = self.server.gui.add_slider(
                label='texture size',
                min=0,
                max=6,
                step=1,
                initial_value=0,
                visible=False
            )

            @tex_slider.on_update
            def _(_):
                self.tex_value = tex_slider.value
                self.rerender(_)

            scale_ratio_slider = self.server.gui.add_slider(
                label='scale porpotion',
                min=1.,
                max=100.,
                step=0.01,
                initial_value=3.,
                visible=False
            )

            @scale_ratio_slider.on_update
            def _(_):
                self.scale_ratio = scale_ratio_slider.value
                self.rerender(_)

            min_scale_slider = self.server.gui.add_slider(
                label='min scale',
                min=0.,
                max=1.,
                step=0.01,
                initial_value=0.1,
                visible=False
            )

            @min_scale_slider.on_update
            def _(_):
                self.min_scale_value = min_scale_slider.value
                self.rerender(_)

            @render_mode_dropdown.on_update
            def _(_):
                self.render_mode = render_mode_dropdown.value
                tex_slider.visible = (self.render_mode is RenderMode.TEX_SIZE) or (self.render_mode is RenderMode.TEX_ONLY)
                scale_ratio_slider.visible = self.render_mode is RenderMode.SCALE
                min_scale_slider.visible = self.render_mode is RenderMode.SCALE
                self.rerender(_)

            texture_index_slider = self.server.gui.add_slider(
                label='texture gaussian index',
                min=0,
                max=1,
                step=1,
                initial_value=0,
                visible=False
            )
            self.texture_index_slider = texture_index_slider

            @texture_index_slider.on_update
            def _(_):
                if self._texture_index_callback is not None:
                    self._texture_index_callback(int(texture_index_slider.value))

    def configure_texture_index_control(self, max_index: int, callback):
        """Configure visibility and behaviour of the gaussian texture index slider."""
        if self.texture_index_slider is None:
            raise RuntimeError("Texture index slider has not been initialized.")

        if max_index is None or max_index < 0:
            self.texture_index_slider.visible = False
            self._texture_index_callback = None
            return

        self._texture_index_callback = callback
        self.texture_index_slider.min = 0
        self.texture_index_slider.max = max_index
        self.texture_index_slider.step = 1

        current_value = int(self.texture_index_slider.value)
        if current_value < 0 or current_value > max_index:
            current_value = 0
            self.texture_index_slider.value = current_value

        self.texture_index_slider.visible = True
        if callback is not None:
            callback(int(self.texture_index_slider.value))


    def set_rendering_functions(self, render_train_fn: callable, render_val_fn: callable,
                                gt_train_fn: callable, gt_val_fn: callable):
        """Sets the external rendering and ground truth functions for the viewer."""
        self._render_train_fn = render_train_fn
        self._render_val_fn = render_val_fn
        self._gt_train_fn = gt_train_fn
        self._gt_val_fn = gt_val_fn

    def add_slider_to_gui(self, label: str, initial_value: float, callback: callable):
        """Adds a slider to the custom rendering folder in the GUI."""
        with self._custom_rendering_folder:
            slider = self.server.gui.add_slider(
                label,
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=initial_value,
                hint=f"Adjust the position of {label}"
            )
            # Attach the callback to the slider's update event
            @slider.on_update
            def on_update(_) -> None:
                callback(slider.value)
                self.rerender(_) # Trigger a main scene rerender when slider changes

    def _update_image_plot(self, frustum_idx: int, is_train: bool):
        """
        Helper function to render images and update the corresponding Plotly figure
        for either train or validation frustums.
        """
        if is_train:
            gt_image = self._gt_train_fn(frustum_idx) / 255.0
            render_image = np.clip(self._render_train_fn(frustum_idx), 0.0, 1.0)
            plot_handle = self.train_plot_handle
        else:
            gt_image = self._gt_val_fn(frustum_idx) / 255.0
            render_image = np.clip(self._render_val_fn(frustum_idx), 0.0, 1.0)
            plot_handle = self.val_plot_handle

        # Flip images vertically for correct display in Plotly
        gt_image = np.flip(gt_image, axis=(0))
        render_image = np.flip(render_image, axis=(0))

        diff = np.abs(gt_image - render_image)
        # Stack images: Ground Truth, Rendered, and the square root of their difference
        image_stack = np.array([gt_image, render_image, np.sqrt(diff)])

        fig = px.imshow(
            image_stack,
            facet_col=0,
            facet_col_wrap=2,
            facet_col_spacing=0.0,
            facet_row_spacing=0.0,
            labels={
                "facet_col": "Image Type",
                "x": "Width",
                "y": "Height",
                "color": "Pixel Value"
            },
            title="Ground Truth | Rendered | Difference" # This title will be overridden by update_layout
        )
        fig.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> color: %{color}")
        fig.for_each_annotation(lambda a: a.update(text='')) # Remove default facet titles for cleaner look
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            title_text="Ground Truth vs. Rendered Image (and Difference)",
            title_x=0.5 # Center the main title
        )
        plot_handle.figure = fig

    def _update_texture_show(self, gaussian_idx: int):
        """Load a single gaussian texture and update the Plotly preview."""
        if not self.textured_gaussian_models:
            return

        model_idx = max(0, min(self.viewer.debug_model_idx, len(self.textured_gaussian_models) - 1))
        model = self.textured_gaussian_models[model_idx]
        num_textures = int(model.texture_dims.shape[0])
        idx = int(gaussian_idx)
        if idx < 0 or idx >= num_textures:
            return

        dims_tensor = model.texture_dims[idx]
        width, height = [int(v) for v in dims_tensor.tolist()]
        if width <= 0 or height <= 0:
            return

        offset_tensor = model.texture_offsets[idx]
        offset = int(offset_tensor.view(-1)[0].item())
        texel_count = width * height
        total_texel_count = int(model.textures_packed.shape[1])
        if offset + texel_count > total_texel_count:
            print(f"[Texture Viewer] Texture index {idx} exceeds packed range.", flush=True)
            return

        channels = model.textures_packed[:, offset: offset + texel_count]
        channel_count = channels.shape[0]
        if channel_count < 4:
            pad = torch.ones(4 - channel_count, texel_count, device=channels.device, dtype=channels.dtype)
            channels = torch.cat([channels, pad], dim=0)
        elif channel_count > 4:
            channels = channels[:4]

        texture_flat = channels.detach().cpu().numpy()
        dims_array = np.array([width, height], dtype=np.int32)
        if (
            self.viewer.texture_index_slider is not None
            and hasattr(self.viewer.texture_index_slider, "hint")
        ):
            self.viewer.texture_index_slider.hint = f"{width}x{height}"
        self.viewer.load_texture_show(data=texture_flat, dim=dims_array)


    def update_frustum_callback(self):
        """Attaches click callbacks to train and validation frustums to update image plots."""
        # Attach callback for training frustums
        for i, frustum in enumerate(self.train_frustums):
            @frustum.on_click
            def _(event, frustum_idx=i) -> None:
                self._update_image_plot(frustum_idx, is_train=True)
                # self.debug_model_idx = frustum_idx # Update the debug model index

        # Attach callback for validation frustums
        for i, frustum in enumerate(self.val_frustums):
            @frustum.on_click
            def _(event, frustum_idx=i) -> None:
                self._update_image_plot(frustum_idx, is_train=False)
                # self.debug_model_idx = frustum_idx # Update the debug model index

    def get_contrib_sum_plot(self, histc_input: torch.Tensor, title: str) -> go.Figure:
        """Generates a histogram plot for given tensor data, using predefined constants."""
        with torch.no_grad():
            hist = torch.histc(histc_input, bins=HIST_BINS, min=HIST_MIN_VAL, max=HIST_MAX_VAL).cpu().detach().numpy()

        bin_edges = torch.linspace(HIST_MIN_VAL, HIST_MAX_VAL, steps=HIST_BINS + 1).cpu().numpy()
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig = px.histogram(
            x=bin_centers,
            y=hist,
            nbins=HIST_BINS,
            labels={'x': 'Value', 'y': 'Count'},
            title=title
        )
        return fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    
    def set_texture_plot(
        self,
        data: np.ndarray,
        bins=(20, 20),
        bar_scale=0.9,
        colorscale="Viridis",
        renderer=None,          # e.g. "browser", "vscode", "notebook_connected"
        save_html=None          # e.g. "3d_hist.html"
    ):
        """
        Plot a 3D bar-chart histogram for 2D points using Plotly Mesh3d.

        data: (n,2) array of (x,y) points
        bins: (bx, by) or [edges_x, edges_y]
        bar_scale: 0..1, shrink bars inside each bin so gaps are visible
        """
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must have shape (n, 2)")

        # 2D histogram
        H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=bins)

        # bin centers and widths (per-bin to be safe)
        x_cent = 0.5*(xedges[:-1] + xedges[1:])
        y_cent = 0.5*(yedges[:-1] + yedges[1:])
        x_w = np.diff(xedges)
        y_w = np.diff(yedges)

        # Collect vertices and faces for all cuboids
        X, Y, Z = [], [], []
        I, J, K = [], [], []
        intensity = []

        def add_bar(x0, x1, y0, y1, z):
            """Add one cuboid [x0,x1]x[y0,y1]x[0,z] as 8 verts + 12 triangles."""
            base = len(X)
            # order: 0..3 bottom, 4..7 top (see diagram in code comments)
            xs = [x0, x1, x1, x0, x0, x1, x1, x0]
            ys = [y0, y0, y1, y1, y0, y0, y1, y1]
            zs = [0, 0, 0, 0, z, z, z, z]
            X.extend(xs); Y.extend(ys); Z.extend(zs)
            intensity.extend([z]*8)

            # 12 triangles (two per face)
            faces = [
                (0,1,2),(0,2,3),       # bottom
                (4,6,5),(4,7,6),       # top
                (0,5,1),(0,4,5),       # side x+
                (1,6,2),(1,5,6),       # side y+
                (2,7,3),(2,6,7),       # side x-
                (3,4,0),(3,7,4)        # side y-
            ]
            for a,b,c in faces:
                I.append(base+a); J.append(base+b); K.append(base+c)

        # build bars
        for ix, xc in enumerate(x_cent):
            dx = x_w[ix]*bar_scale
            x0, x1 = xc - dx/2, xc + dx/2
            for iy, yc in enumerate(y_cent):
                count = int(H[ix, iy])  # H is (len(xedges)-1, len(yedges)-1)
                if count <= 0:
                    continue
                dy = y_w[iy]*bar_scale
                y0, y1 = yc - dy/2, yc + dy/2
                add_bar(x0, x1, y0, y1, count)

        fig = go.Figure(go.Mesh3d(
            x=X, y=Y, z=Z,
            i=I, j=J, k=K,
            intensity=intensity, colorscale=colorscale, showscale=True,
            flatshading=True, opacity=1.0,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1),
            lightposition=dict(x=100, y=200, z=0)
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Texture width",
                yaxis_title="Texture height",
                zaxis_title="Count",
                aspectmode="cube"
            ),
            title="3D Histogram (bars) from 2D data"
        )

        self.texture_plot_handle.figure = fig

    def load_texture_show(
            self,
            data: np.ndarray,
            dim: np.array
        ):
        """
        Display a single flattened RGBA texture as a 2D image inside the GUI.

        Parameters
        ----------
        data:
            Flattened texture array with shape (4, width*height) or (width*height, 4).
        dim:
            Iterable containing (width, height).
        """
        if data is None or dim is None:
            raise ValueError("Both data and dim must be provided.")

        texture_flat = np.asarray(data)
        dims = np.asarray(dim).reshape(-1)
        if dims.size != 2:
            raise ValueError(f"'dim' must contain exactly two elements (width, height), got {dims}.")

        width, height = map(int, dims.tolist())
        if width <= 0 or height <= 0:
            raise ValueError(f"Texture dimensions must be positive, got ({width}, {height}).")

        if texture_flat.ndim != 2:
            raise ValueError(f"Texture array must be 2D with 4 channels, got shape {texture_flat.shape}.")

        if texture_flat.shape[0] == 4:
            flat_rgba = texture_flat
        elif texture_flat.shape[1] == 4:
            flat_rgba = texture_flat.T
        else:
            raise ValueError(f"Texture array must have 4 channels; received shape {texture_flat.shape}.")

        expected_size = width * height
        if flat_rgba.shape[1] != expected_size:
            raise ValueError(
                f"Texture data length ({flat_rgba.shape[1]}) does not match width*height ({expected_size})."
            )

        # Reshape into image tensor (H, W, 4) and convert to uint8 for Plotly.
        texture_rgba = flat_rgba.reshape(4, height, width).transpose(1, 2, 0)
        texture_rgba = np.flip(texture_rgba, axis=0)  # match previous top-left origin view
        texture_rgba = np.clip(texture_rgba, 0.0, 1.0)
        texture_rgba_uint8 = (texture_rgba * 255).astype(np.uint8)

        fig = go.Figure(
            data=go.Image(
                z=texture_rgba_uint8,
                colormodel="rgba"
            )
        )
        fig.update_layout(
            title="Texture Preview",
            xaxis=dict(title="Width (pixels)", constrain="domain"),
            yaxis=dict(title="Height (pixels)"),
            margin=dict(l=10, r=10, t=40, b=10)
        )

        self.texture_show_handle.figure = fig


class GaussianViewerApp:
    """
    Main application class to manage the textured Gaussian models, datasets,
    and the Viser viewer.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda", args.local_rank) # Use local_rank for device assignment
        self.textured_gaussian_models: List[TexturedGaussiansModel] = []
        self.slider_positions: List[float] = []
        self.slider_callbacks: List[callable] = []
        self.sh_degree: int = 0 # Spherical harmonics degree

        self.trainset = None
        self.valset = None

        # Initialize the Viser server and CustomViewer
        self.server = viser.ViserServer(port=self.args.port, verbose=False)
        self.viewer = CustomViewer(
            server=self.server,
            render_fn=self._viewer_render_fn, # Main rendering function for the 3D scene
            mode="rendering",
        )
        self.server.gui.set_panel_label("ntex viewer")

    def _load_datasets(self):
        """Loads the training and validation datasets based on command-line arguments."""
        if self.args.dataset == "colmap":
            parser = Parser(
                data_dir=self.args.data_dir,
                factor=self.args.data_factor,
                normalize=True,
                test_every=self.args.test_every,
            )
            self.trainset = Dataset(parser, split="train")
            self.valset = Dataset(parser, split="val")
        elif self.args.dataset == "blender":
            self.trainset = BlenderDataset(data_dir=self.args.data_dir, split="train")
            self.valset = BlenderDataset(data_dir=self.args.data_dir, split="val")
        else:
            raise ValueError(f"Dataset mode {self.args.dataset} not supported!")

    def _load_models(self):
        """
        Loads TexturedGaussiansModel instances from specified checkpoint paths.
        Converts textures to a packed format for efficient rendering.
        """
        self.textured_gaussian_models.clear() # Clear existing models before loading new ones

        def _load_checkpoint(path: str) -> Dict[str, Any]:
            try:
                return torch.load(path, map_location=self.device, weights_only=True)
            except TypeError: # PyTorch < 2.1 fallback
                return torch.load(path, map_location=self.device)

        for ckpt_path in self.args.ckpt:
            ckpt_full = _load_checkpoint(ckpt_path)
            ckpt = ckpt_full["splats"]
            means = ckpt["means"]
            quats = ckpt["quats"]
            scales = torch.exp(ckpt["scales"])
            opacities = torch.sigmoid(ckpt["opacities"])
            sh0 = ckpt["sh0"]
            shN = ckpt["shN"]
            # textures = ckpt["textures"]

            # Convert textures from [N, W, H, C] to packed format [C, N*W*H]
            textures_packed = ckpt['textures_packed']
            rgb_textures = textures_packed[:3, :]
            alpha_textures = textures_packed[3:4, :]
            alpha_textures = alpha_textures / (alpha_textures.amax(dim=1, keepdim=True) + 1e-6) # normalize so that the max is 1
            textures_packed = torch.cat([rgb_textures, alpha_textures], dim=0) # [4, \sum(N_i * H_i * W_i)]
            textures_packed = textures_packed.clamp(0.0, 1.0)

            # N, W, H, C = textures.shape

            # Texture dimensions: [N, 2] with [W, H] for each texture
            # texture_dims = torch.tensor([[W, H]] * N, device=textures.device, dtype=torch.int32)
            texture_dims = ckpt_full['texture_dims']
            texture_offsets = ckpt_full['texture_offsets']

            # Offsets: [N, 1] where each is the cumulative sum of previous W*H areas
            # areas = texture_dims[:, 0] * texture_dims[:, 1]  # W * H for each texture -> [N]
            # texture_offsets = torch.zeros_like(areas)
            # Calculate cumulative sum for offsets, excluding the last element
            # texture_offsets[1:] = torch.cumsum(areas, dim=0)[:-1]
            # texture_offsets = texture_offsets.unsqueeze(1)  # Reshape to [N, 1]

            colors = torch.cat([sh0, shN], dim=-2) # Concatenate SH coefficients

            # Create and store the TexturedGaussiansModel instance
            model = TexturedGaussiansModel(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures=None,
                sh0=sh0,
                shN=shN,
                textures_packed=textures_packed,
                texture_dims=texture_dims,
                texture_offsets=texture_offsets
            )
            self.textured_gaussian_models.append(model)

        # Determine the spherical harmonics degree from the first loaded model
        if self.textured_gaussian_models:
            self.sh_degree = int(math.sqrt(self.textured_gaussian_models[0].colors.shape[-2]) - 1)
            print(f"Number of Gaussian models loaded: {len(self.textured_gaussian_models)}")
        else:
            print("No Gaussian models loaded.")

    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState
    ) -> np.array:
        """
        The main rendering function called by the Viser viewer.
        It renders a composite image by combining outputs from multiple Gaussian models
        based on slider positions.
        """
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        # Convert camera state to PyTorch tensors and move to device
        c2w = torch.tensor(camera_state.c2w).to(self.device, dtype=torch.float32)
        K = torch.tensor(camera_state.get_K([width, height])).to(self.device, dtype=torch.float32)

        num_ckpts = len(self.textured_gaussian_models)
        render_images = [None] * num_ckpts
        metrics = {
            "gs_contrib_sum": [None] * num_ckpts,
            "gs_contrib_count": [None] * num_ckpts,
            "gs_weight_sum": [None] * num_ckpts,
            "gs_dx_sum": [None] * num_ckpts,
            "gs_dy_sum": [None] * num_ckpts
        }

        # Render each Gaussian model
        for i in range(num_ckpts):
            model = self.textured_gaussian_models[i]
            means = model.means
            quats = model.quats
            scales = model.scales * self.viewer.scale_modifier_value
            colors = model.colors.clone()
            opacities = model.opacities * self.viewer.opacity_modifier_value
            textures_packed = model.textures_packed.clone()
            match self.viewer.render_mode:
                case RenderMode.TEX_SIZE:
                    # Pull dims
                    tex_dims = model.texture_dims    # shape [..., 2]
                    u = tex_dims[..., 0]
                    v = tex_dims[..., 1]

                    # Select all tex with both dims < tex_value and not 1x1
                    tv = 2**int(self.viewer.tex_value)
                    if tv > 1:
                        valid = (u <= tv) & (v <= tv) & ~((u == 1) & (v == 1))

                        # Split into square vs non-square
                        square_mask    = valid & (u == v)
                        nonsquare_mask = valid & (u != v)

                        # Colors on the same device/dtype as 'colors'
                        blue = torch.tensor([0.0, 0.0, 1.0], device=colors.device, dtype=colors.dtype)
                        red  = torch.tensor([1.0, 0.0, 0.0], device=colors.device, dtype=colors.dtype)
                        black = torch.tensor([0.0, 0.0, 0.0], device=colors.device, dtype=colors.dtype)

                        # Convert to SH once (assumes rgb_to_sh returns a tensor of proper shape)
                        blue_sh = rgb_to_sh(blue)
                        red_sh  = rgb_to_sh(red)

                        # Apply
                        if square_mask.any():
                            colors[square_mask, 0, :] = blue_sh
                            textures_packed[:3, ::2] = blue.unsqueeze(-1)
                            textures_packed[:3, 1::2] = black.unsqueeze(-1)
                        if nonsquare_mask.any():
                            colors[nonsquare_mask, 0, :] = red_sh
                            textures_packed[:3, ::2] = red.unsqueeze(-1)
                            textures_packed[:3, 1::2] = black.unsqueeze(-1)
                    # else: tv <= 1 → nothing to highlight

                case RenderMode.NO_TEX:
                    textures_packed[:3, ...] = torch.zeros_like(textures_packed[:3, ...])
                    textures_packed[-1, ...] = torch.ones_like(textures_packed[-1, ...])

                case RenderMode.NO_BASE:
                    black = torch.tensor([0.0, 0.0, 0.0], device=colors.device, dtype=colors.dtype)
                    black_sh  = rgb_to_sh(black)
                    colors[:,0, :] = black_sh

                case RenderMode.TEX_ONLY:
                    tex_dims = model.texture_dims    # shape [..., 2]
                    u = tex_dims[..., 0]
                    v = tex_dims[..., 1]

                    # Select all tex with both dims < tex_value and not 1x1
                    tv = 2**int(self.viewer.tex_value)
                    if tv > 1:
                        valid = (u <= tv) & (v <= tv) & ~((u == 1) & (v == 1))

                        opacities[~valid] = 0.0
                    
                case RenderMode.SCALE:
                    mask = (model.scales[:,0] / model.scales[:,1]) > self.viewer.scale_ratio
                    mask |= (model.scales[:,1] / model.scales[:,0]) > self.viewer.scale_ratio
                    mask &= model.scales[:,0] < self.viewer.min_scale_value
                    mask &= model.scales[:,1] < self.viewer.min_scale_value
                    colors[mask,0,:] = rgb_to_sh(torch.Tensor([1.0, 0.0, 0.0]).cuda())

            render_colors, *_, gs_contrib_sum, gs_contrib_count, gs_weight_sum, gs_dx_sum, gs_dy_sum, meta, = rasterization_packed_textured_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures_packed=textures_packed,
                texture_dims=model.texture_dims,
                texture_offsets=model.texture_offsets,
                viewmats=torch.linalg.inv(c2w[None]), # Inverse of camera-to-world matrix
                Ks=K[None], # Camera intrinsics
                width=width,
                height=height,
                sh_degree=self.sh_degree
            )
            render_images[i] = render_colors
            metrics['gs_contrib_count'][i] = gs_contrib_count
            metrics['gs_contrib_sum'][i] = gs_contrib_sum
            metrics['gs_weight_sum'][i] = gs_weight_sum
            metrics['gs_dx_sum'][i] = gs_dx_sum
            metrics['gs_dy_sum'][i] = gs_dy_sum

        # Update plots in the GUI if their checkboxes are active
        for name, plot_handle in self.viewer.plots.items():
            if self.viewer.plots_checkboxes[name].value:
                # Assuming we only plot for the first model's metrics for simplicity
                plot_handle.figure = self.viewer.get_contrib_sum_plot(metrics[name][0], name)

        # Return a black image if no models are loaded
        if not render_images:
            return np.zeros((height, width, 3), dtype=np.float32)

        # Prepare final composite image
        H, W, C = render_images[0][0].shape
        final_image = torch.zeros_like(render_images[0][0])

        # Composite images based on slider positions
        for i in range(num_ckpts):
            # Calculate start and end X-coordinates for the current segment
            start_x = int(self.slider_positions[i-1] * W) if i != 0 else 0
            end_x = int(self.slider_positions[i] * W) if i != num_ckpts - 1 else W
            
            # Ensure the last segment covers the remaining width
            if i == num_ckpts - 1:
                end_x = W 
            
            # Copy the segment from the current rendered image to the final image
            final_image[:, start_x:end_x] = render_images[i][0][:, start_x:end_x]

        # Draw separator lines between segments for visual clarity
        for pos in self.slider_positions:
            x_pos = int(pos * W)
            # Draw a 2-pixel wide white line
            final_image[:, max(0, x_pos - 1):min(W, x_pos + 1)] = 1.0

        return final_image.cpu().detach().numpy()

    def _train_render_fn(self, idx: int) -> np.array:
        """Renders a single image from the training set using the currently debugged model."""
        data = self.trainset[idx]
        K = data['K'].to(self.device)
        c2w = data['camtoworld'].to(self.device)
        image = data['image'].to(self.device) # Not used for rendering, but for shape
        h, w = image.shape[0], image.shape[1]
        
        # Use the model selected by the debug_model_idx in the viewer
        model = self.textured_gaussian_models[self.viewer.debug_model_idx] 
        render_colors, *_ = rasterization_packed_textured_gaussians(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.colors,
            textures_packed=model.textures_packed,
            texture_dims=model.texture_dims,
            texture_offsets=model.texture_offsets,
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=w,
            height=h,
            sh_degree=self.sh_degree
        )
        return render_colors.squeeze(0).detach().cpu().numpy()

    def _val_render_fn(self, idx: int) -> np.array:
        """Renders a single image from the validation set using the currently debugged model."""
        data = self.valset[idx]
        K = data['K'].to(self.device)
        c2w = data['camtoworld'].to(self.device)
        image = data['image'].to(self.device) # Not used for rendering, but for shape
        h , w = image.shape[0], image.shape[1]
        
        # Use the model selected by the debug_model_idx in the viewer
        model = self.textured_gaussian_models[self.viewer.debug_model_idx] 
        render_colors, *_ = rasterization_packed_textured_gaussians(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.colors,
            textures_packed=model.textures_packed,
            texture_dims=model.texture_dims,
            texture_offsets=model.texture_offsets,
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=w,
            height=h,
            sh_degree=self.sh_degree
        )
        return render_colors.squeeze(0).detach().cpu().numpy()

    def _train_gt_fn(self, idx: int) -> np.array:
        """Retrieves the ground truth image from the training set."""
        return self.trainset[idx]['image'].detach().cpu().numpy()

    def _val_gt_fn(self, idx: int) -> np.array:
        """Retrieves the ground truth image from the validation set."""
        return self.valset[idx]['image'].detach().cpu().numpy()

    def _make_update_slider_callback(self, idx: int) -> callable:
        """Factory function to create a callback for a specific slider."""
        def update_slider(value: float):
            self.slider_positions[idx] = value
        return update_slider

    def run(self):
        """Main method to set up and run the Gaussian viewer application."""
        torch.manual_seed(42) # Ensure reproducibility

        self._load_datasets()
        self._load_models()
        plot_texture_shape_barchart_pct(
            data=self.textured_gaussian_models[0].texture_dims.detach().cpu().numpy(),
            save_path="./chart.pdf",
            figsize=(8.0, 4.0), dpi=300,   # or use width_px/height_px for PNG
        )

        num_ckpts = len(self.textured_gaussian_models)
        # Initialize slider positions for blending multiple checkpoints
        self.slider_positions = [(i + 1) * (1 / num_ckpts) for i in range(num_ckpts - 1)]
        # Create callbacks for each slider
        self.slider_callbacks = [self._make_update_slider_callback(i) for i in range(num_ckpts - 1)]

        # Pass the rendering and ground truth functions to the viewer
        self.viewer.set_rendering_functions(
            render_train_fn=self._train_render_fn,
            render_val_fn=self._val_render_fn,
            gt_train_fn=self._train_gt_fn,
            gt_val_fn=self._val_gt_fn
        )

        # Add sliders to the GUI based on the number of checkpoints
        for i, (callback, initial_value) in enumerate(zip(self.slider_callbacks, self.slider_positions)):
            self.viewer.add_slider_to_gui(f"Slider {i+1}", initial_value, callback)

        # Update frustums and attach click callbacks in the viewer
        self.viewer.custom_update(train_dataset=self.trainset, val_dataset=self.valset)
        self.viewer.update_frustum_callback()


        current_tg_model = self.textured_gaussian_models[0]

        # texture_plot = plotl
        self.viewer.set_texture_plot(
            data=current_tg_model.texture_dims.detach().cpu().numpy(),
            bins=(3,3)
        )

        mask = current_tg_model.texture_dims[:,0] >= 4
        indices = torch.nonzero(mask)
        tindx = indices[3]
        print(f'tindx: {tindx}')
        tdim = current_tg_model.texture_dims[tindx].squeeze()
        toffset = self.textured_gaussian_models[0].texture_offsets[tindx]
        tsize = tdim[0] * tdim[1]
        self.viewer.load_texture_show(
            self.textured_gaussian_models[0].textures_packed[:,toffset:toffset+tsize].detach().cpu().numpy(),
            tdim.detach().cpu().numpy()
        )


        print("Viewer running... Ctrl+C to exit.")
        # Keep the server running indefinitely
        while True:
            time.sleep(1e-3)

def main(local_rank: int, world_rank, world_size: int, args):
    """
    Entry point for the distributed CLI.
    Initializes and runs the GaussianViewerApp.
    """
    # Attach local_rank to args for the app to use
    args.local_rank = local_rank 
    app = GaussianViewerApp(args)
    app.run()

if __name__ == "__main__":
    """
    Command-line interface setup for running the Gaussian viewer.
    Example usage:
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

    # Use the distributed CLI entry point
    cli(main, args, verbose=True)
