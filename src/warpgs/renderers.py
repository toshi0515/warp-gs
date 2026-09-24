from dataclasses import dataclass
from typing import Any

import torch

from warpgs.model import GaussianModel


@dataclass
class RenderOutput:
    image: torch.Tensor
    alpha: torch.Tensor | None
    info: dict[str, Any]


class WarpRenderer:
    name = "warp"

    def __init__(self) -> None:
        import warp as wp

        wp.init()

    def render(
        self,
        model: GaussianModel,
        w2c: torch.Tensor,
        K: torch.Tensor,
        height: int,
        width: int,
        background: torch.Tensor,
    ) -> RenderOutput:
        from warpgs.my_warp_rasterizer import WarpRasterizeFunction, project_means_to_screen

        means2d = project_means_to_screen(model.means, w2c, K)
        image, radii = WarpRasterizeFunction.apply(
            model.means,
            model.quats,
            model.scales,
            model.opacities,
            model.colors,
            means2d,
            w2c,
            K,
            height,
            width,
            background,
        )
        info = {
            "means2d": means2d,
            "radii": radii[None],
            "width": width,
            "height": height,
            "n_cameras": 1,
            "gaussian_ids": None,
        }
        return RenderOutput(image, None, info)


class GsplatRenderer:
    name = "gsplat"

    def __init__(self, packed: bool = False, absgrad: bool = False) -> None:
        ensure_gsplat_available()
        self.packed = packed
        self.absgrad = absgrad

    def render(
        self,
        model: GaussianModel,
        w2c: torch.Tensor,
        K: torch.Tensor,
        height: int,
        width: int,
        background: torch.Tensor,
    ) -> RenderOutput:
        from gsplat.rendering import rasterization

        image, alpha, info = rasterization(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.colors,
            viewmats=w2c[None],
            Ks=K[None],
            width=width,
            height=height,
            backgrounds=background[None],
            packed=self.packed,
            absgrad=self.absgrad,
        )
        return RenderOutput(image[0], alpha[0], info)


def ensure_gsplat_available() -> None:
    try:
        from gsplat.cuda._backend import _C

        if _C is None:
            raise RuntimeError("CUDA extension is not loaded")
    except Exception as error:
        raise RuntimeError(
            f"gsplat CUDA extension is unavailable. Run scripts/setup_gsplat_windows.ps1. Original error: {error}"
        ) from error


def create_renderer(name: str, packed: bool = False, absgrad: bool = False) -> WarpRenderer | GsplatRenderer:
    if name == "warp":
        return WarpRenderer()
    return GsplatRenderer(packed, absgrad)
