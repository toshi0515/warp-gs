import math
import subprocess
import sys

import pytest
import torch
from math_validation import check_differences, finite_differences, loss, reference, scene, signature

from warpgs.reference_math import composite_reference, covariance_3d, project_legacy, project_pinhole


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_reference_import_does_not_initialize_gpu_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch; import warpgs.reference_math; "
            "assert 'warp' not in sys.modules and 'gsplat' not in sys.modules; "
            "assert not torch.cuda.is_initialized()",
        ],
        check=True,
    )


def test_covariance_identity_and_quarter_turn():
    scales = tensor([[1, 2, 3]])
    close(covariance_3d(scales, tensor([[1, 0, 0, 0]])), torch.diag_embed(scales.square()))
    q = tensor([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]])
    close(covariance_3d(scales, q), torch.diag_embed(tensor([[4, 1, 9]])))


def test_covariance_invariants():
    s, q = tensor([[0.2, 0.5, 0.8]]), tensor([[0.9, 0.2, 0.3, -0.1]])
    covariance = covariance_3d(s, q)
    close(covariance, covariance.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0)
    close(covariance, covariance_3d(s, -q))
    close(covariance, covariance_3d(s, 3 * q))
    close(torch.linalg.eigvalsh(covariance), s.square().sort().values)


def test_projection_on_axis_and_rigid_camera():
    means, cov = tensor([[0, 0, 2]]), torch.diag_embed(tensor([[0.04, 0.09, 0.16]]))
    K = tensor([[20, 0, 3], [0, 30, 5], [0, 0, 1]])
    view = torch.eye(4, dtype=torch.float64)
    p = project_pinhole(means, cov, view, K)
    close(p.mean, tensor([[3, 5]]))
    close(p.jacobian, tensor([[[10, 0, 0], [0, 15, 0]]]))
    close(p.covariance, torch.diag_embed(tensor([[4, 20.25]])))
    # The transformed camera point is (0,0,2); x/y variances exchange.
    view[:3, :3] = tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    view[:3, 3] = tensor([2, -1, -1])
    p = project_pinhole(tensor([[1, 2, 3]]), cov, view, K)
    close(p.mean, tensor([[3, 5]]))
    close(p.covariance, torch.diag_embed(tensor([[9, 9]])))


def test_legacy_projection_is_separate_from_ideal():
    means, cov = tensor([[4, 0, 2]]), torch.diag_embed(tensor([[0.04, 0.09, 0.16]]))
    view, K = torch.eye(4, dtype=torch.float64), tensor([[20, 0, 3], [0, 30, 5], [0, 0, 1]])
    ideal = project_pinhole(means, cov, view, K)
    legacy = project_legacy(means, cov, view, K, 12, 12)
    close(legacy.mean, ideal.mean)
    close(legacy.jacobian[0, 0], tensor([10, 0, -3.9]))
    close(legacy.covariance[0, 0, 0], tensor(4 + 3.9**2 * 0.16 + 0.3))
    close(legacy.inverse @ legacy.covariance, torch.eye(2, dtype=torch.float64)[None])
    assert not torch.allclose(legacy.covariance, ideal.covariance)


@pytest.mark.parametrize("count", [1, 2])
def test_compositing_hand_calculation(count):
    mean, conic = tensor([[0.5, 0.5]] * count), tensor([[1, 0, 1]] * count)
    opacity, colors, bg = tensor([0.2, 0.4])[:count], tensor([[1, 0, 0], [0, 1, 0]])[:count], tensor([0, 0, 1])
    output = composite_reference(mean, conic, opacity, colors, 1, 1, bg)
    expected = [0.2, 0, 0.8] if count == 1 else [0.2, 0.32, 0.48]
    close(output.image[0, 0], tensor(expected))
    close(output.transmittance[0, 0], tensor(expected[2]))


@pytest.mark.parametrize("count", [1, 2])
def test_reference_autograd_central_differences(count):
    params, camera = scene(count)
    differentiable = {k: v.clone().requires_grad_() for k, v in params.items()}
    loss(reference(differentiable, camera).image).backward()
    gradients = {k: v.grad for k, v in differentiable.items()}

    def evaluate(values):
        output = reference(values, camera)
        return loss(output.image).item(), signature(output)

    steps = [1e-4, 1e-5, 1e-6]
    differences = finite_differences(params, evaluate, steps)
    check_differences(f"cpu_fd/n={count}", differences, gradients, steps, 1e-7, 1e-5)


def test_invalid_pure_math_inputs():
    with pytest.raises(ValueError, match="nonzero"):
        covariance_3d(tensor([[1, 1, 1]]), tensor([[0, 0, 0, 0]]))
    with pytest.raises(ValueError, match="positive camera depth"):
        project_pinhole(
            tensor([[0, 0, 0]]),
            torch.eye(3, dtype=torch.float64)[None],
            torch.eye(4, dtype=torch.float64),
            torch.eye(3, dtype=torch.float64),
        )
