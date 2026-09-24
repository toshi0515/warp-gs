from types import SimpleNamespace

import pytest
import torch

from warpgs.train import evaluate_test_psnr, make_training_curves, render_test_view


class FakeLoader:
    N = 2
    H = 2
    W = 2

    def __init__(self):
        self.requested = []

    def get_by_index(self, idx):
        self.requested.append(idx)
        return torch.full((2, 2, 3), float(idx)), torch.eye(4), torch.eye(3)


class Model:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True


class Renderer:
    def __init__(self):
        self.calls = 0

    def render(self, model, w2c, K, height, width, bg_color):
        assert model.eval_called
        assert not torch.is_grad_enabled()
        self.calls += 1
        return SimpleNamespace(image=torch.zeros((height, width, 3)))


def test_evaluation_averages_all_test_views_without_gradients():
    loader = FakeLoader()
    model = Model()
    renderer = Renderer()

    psnr = evaluate_test_psnr(model, renderer, loader, torch.ones(3), "cpu")

    assert loader.requested == [0, 1]
    assert renderer.calls == 2
    assert psnr == pytest.approx(40.0)


def test_comparison_view_comes_from_test_loader():
    loader = FakeLoader()
    renderer = Renderer()
    model = Model()
    model.eval()

    with torch.no_grad():
        target, pred = render_test_view(model, renderer, loader, 1, torch.ones(3), "cpu")

    assert loader.requested == [1]
    assert torch.all(target == 1)
    assert torch.all(pred == 0)


def test_training_curves_use_actual_step_numbers():
    import matplotlib.pyplot as plt

    fig, axes = make_training_curves([0.5, 0.4], [20.0, 21.0], [10, 12])
    try:
        assert list(axes[0].lines[0].get_xdata()) == [1, 2]
        assert list(axes[1].lines[0].get_xdata()) == [100, 200]
        assert list(axes[2].lines[0].get_xdata()) == [1, 2]
        assert axes[1].get_title() == "Test PSNR (dB)"
    finally:
        plt.close(fig)
