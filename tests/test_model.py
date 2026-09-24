from pathlib import Path

import pytest
import torch
from torch import nn

from warpgs.model import PARAM_KEYS, GaussianModel, create_optimizers


def make_model(count: int = 4) -> GaussianModel:
    means = torch.linspace(-0.2, 0.2, count * 3).reshape(count, 3)
    quats = torch.zeros(count, 4)
    quats[:, 0] = 1.0
    scales = torch.full((count, 3), -2.0)
    opacities = torch.zeros(count)
    colors = torch.zeros(count, 3)
    return GaussianModel(means, quats, scales, opacities, colors)


def test_model_properties_follow_parameter_replacement() -> None:
    model = make_model()
    assert set(model.params) == set(PARAM_KEYS)
    assert torch.isfinite(model.scales).all()
    assert torch.isfinite(model.opacities).all()
    assert torch.isfinite(model.colors).all()

    replacement = nn.Parameter(torch.ones_like(model.means))
    model.params["means"] = replacement
    assert model.means is replacement


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "model.pth"
    model.save(path)
    loaded = GaussianModel.load(path, device="cpu")
    for key in PARAM_KEYS:
        assert torch.equal(loaded.params[key], model.params[key])


def test_optimizer_layout_is_accepted_by_gsplat() -> None:
    strategy_module = pytest.importorskip("gsplat.strategy")
    model = make_model()
    optimizers = create_optimizers(model)

    assert set(optimizers) == set(model.params)
    for name, optimizer in optimizers.items():
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]["params"]) == 1
        assert optimizer.param_groups[0]["params"][0] is model.params[name]

    strategy_module.MCMCStrategy().check_sanity(model.params, optimizers)
