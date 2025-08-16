from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops import rescale_texture
from typing_extensions import Literal


@dataclass
class TextureStrategy(Strategy):

    min_tex_res: int = 1
    max_tex_res: int = 16
    upscale_grad2d: float = 0.0002
    upscale_start_iter: int = 500
    upscale_stop_iter: int = 15000
    upscale_every: int = 100
    reset_every: int = 3000
    absgrad: bool = True
    verbose: bool = True
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        state = {
            "grad2d": None,
            "count": None
        }
        return state

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        constants: Dict[str, torch.tensor],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.upscale_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.upscale_start_iter
            and step % self.upscale_every == 0
        ):
            count = state["count"]
            grads = state["grad2d"] / count.clamp_min(1)
            device = grads.device

            is_grad_high = grads > self.upscale_grad2d
            print(f'grads.min(): {grads.min()}')
            print(f'grads.max(): {grads.max()}')
            print(f'grads.mean(): {grads.mean()}')
            print(f'Upscale points: {is_grad_high.sum()}')

            texture_dims_dst = constants['texture_dims'].clone()
            texture_dims_dst[is_grad_high] *= 2

            # upscale textures
            textures_packed, texture_offsets = rescale_texture(
                textures_packed=params['textures_packed'],
                texture_offsets_src=constants['texture_offsets'],
                texture_dims_src=constants['texture_dims'],
                texture_dims_dst=texture_dims_dst
            )
            params['textures_packed'] = textures_packed
            constants["texture_offsets"] = texture_offsets
            constants["texture_dims"] = texture_dims_dst


            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            pass

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
