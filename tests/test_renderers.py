import pytest
import torch

from warpgs.model import GaussianModel
from warpgs.renderers import GsplatRenderer, WarpRenderer


def cuda_model() -> GaussianModel:
    means = torch.tensor(
        [[0.0, 0.0, 2.0], [0.2, 0.1, 2.5]],
        device="cuda",
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device="cuda",
    )
    scales = torch.full((2, 3), torch.log(torch.tensor(0.1)).item(), device="cuda")
    opacities = torch.full((2,), torch.logit(torch.tensor(0.8)).item(), device="cuda")
    colors = torch.tensor(
        [[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]],
        device="cuda",
    )
    return GaussianModel(means, quats, scales, opacities, colors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by both rasterizers")
def test_renderer_shapes_finite_and_gradients() -> None:
    try:
        gsplat_renderer = GsplatRenderer(packed=False)
    except RuntimeError as error:
        pytest.skip(str(error))

    w2c = torch.eye(4, device="cuda")
    K = torch.tensor(
        [[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )
    background = torch.ones(3, device="cuda")

    for renderer in (WarpRenderer(), gsplat_renderer, GsplatRenderer(packed=True)):
        model = cuda_model()
        output = renderer.render(model, w2c, K, 32, 32, background)
        assert output.image.shape == (32, 32, 3)
        assert torch.isfinite(output.image).all()
        if output.alpha is not None:
            assert output.alpha.shape == (32, 32, 1)
            assert torch.all((0 <= output.alpha) & (output.alpha <= 1))
        output.image.sum().backward()
        for parameter in model.params.values():
            assert parameter.grad is not None
            assert parameter.grad.shape == parameter.shape
            assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(model.means.grad) > 0
