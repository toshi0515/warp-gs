from dataclasses import dataclass

import torch

from warpgs.model import GaussianModel


@dataclass
class RenderOutput:
    image: torch.Tensor
    alpha: torch.Tensor | None


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
        from warpgs.my_warp_rasterizer import WarpRasterizeFunction

        image = WarpRasterizeFunction.apply(
            model.means,
            model.quats,
            model.scales,
            model.opacities,
            model.colors,
            w2c,
            K,
            height,
            width,
            background,
        )
        return RenderOutput(image, None)


class GsplatRenderer:
    name = "gsplat"

    def __init__(self, packed: bool = False) -> None:
        ensure_gsplat_available()
        self.packed = packed

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

        image, alpha, _ = rasterization(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.colors,
            viewmats=w2c[None],
            Ks=K[None],
            width=width,
            height=height,
            packed=self.packed,
        )
        image = image[0] + (1.0 - alpha[0]) * background
        return RenderOutput(image, alpha[0])


def ensure_gsplat_available() -> None:
    try:
        from gsplat.cuda._backend import _C

        if _C is None:
            raise RuntimeError("CUDA extension is not loaded")
    except Exception as error:
        raise RuntimeError(
            f"gsplat CUDA extension is unavailable. Run scripts/setup_gsplat_windows.ps1. Original error: {error}"
        ) from error


def create_renderer(name: str, packed: bool = False) -> WarpRenderer | GsplatRenderer:
    if name == "warp":
        return WarpRenderer()
    return GsplatRenderer(packed)
