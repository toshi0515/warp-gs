import json
from pathlib import Path

import pytest
import torch
from math_validation import (
    KEYS,
    check_differences,
    finite_differences,
    loss,
    metric,
    physical,
    reference,
    scene,
    signature,
    weights_for,
)

from warpgs.model import GaussianModel
from warpgs.reference_math import composite_reference, conic_entries, covariance_3d, project_legacy
from warpgs.renderers import WarpRenderer

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def diagnostic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch unavailable")
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp CUDA device unavailable")
    import warp_diagnostics

    return warp_diagnostics


def on_cpu(values):
    return {key: value.detach().cpu().double() for key, value in values.items()}


def cpu_camera(camera):
    return tuple(value.detach().cpu().double() if isinstance(value, torch.Tensor) else value for value in camera)


@pytest.mark.parametrize("case", ["center", "off_axis", "camera", "clamp_inside", "clamp_outside"])
def test_projection_intermediates(diagnostic, case):
    params, camera = scene(dtype=torch.float32, device="cuda")
    view, K, h, w, bg = camera
    if case == "center":
        params["means"][:, :2] = 0
    elif case == "camera":
        view[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device="cuda")
        view[:3, 3] = torch.tensor([0.05, -0.08, 0.2], device="cuda")
        K[0, 2], K[1, 2] = 4.1, 7.3
    elif case.startswith("clamp"):
        factor = 0.99 if case == "clamp_inside" else 1.01
        params["means"][:, 0] = params["means"][:, 2] * (0.65 * w / K[0, 0]) * factor
    means, quats, scales, _, _ = physical(params)
    actual, _ = diagnostic.projection(means, scales, quats, camera)
    # Promote identical physical inputs for intermediate comparisons.
    means64, scales64, quats64 = [v.cpu().double() for v in (means, scales, quats)]
    covariance = covariance_3d(scales64, quats64)
    expected = project_legacy(means64, covariance, view.cpu().double(), K.cpu().double(), h, w)
    values = {
        "cov3d": covariance,
        "mean": expected.mean,
        "diagonal": expected.covariance.diagonal(dim1=-2, dim2=-1),
        "conic": conic_entries(expected.inverse),
        "depth": expected.depth,
    }
    for key in values:
        metric(f"projection/{case}/{key}", actual[key], values[key], 1e-6, 1e-5)
    repeats = [diagnostic.projection(means, scales, quats, camera)[0] for _ in range(4)]
    for key in values:
        variation = max((item[key] - actual[key]).abs().max().item() for item in repeats)
        print("MEASUREMENT " + json.dumps({"name": f"repeat/projection/{case}/{key}", "repeats": 5, "max_abs": variation}))
        for item in repeats:
            torch.testing.assert_close(item[key], actual[key], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("channel", ["mean", "conic"])
def test_preprocessing_backward_paths(diagnostic, channel):
    params, camera = scene(dtype=torch.float32, device="cuda")
    means, quats, scales, _, _ = physical(params)
    actual, gradients = diagnostic.projection(means, scales, quats, camera, channel)
    inputs = {
        key: value.cpu().double().requires_grad_() for key, value in (("means", means), ("scales", scales), ("quats", quats))
    }
    view, K, h, w, _ = cpu_camera(camera)
    p = project_legacy(inputs["means"], covariance_3d(inputs["scales"], inputs["quats"]), view, K, h, w)
    value = p.mean if channel == "mean" else conic_entries(p.inverse)
    seed = torch.linspace(0.1, 0.9, value.numel(), dtype=torch.float32).double().reshape_as(value)
    (value * seed).sum().backward()
    for key, item in inputs.items():
        expected = item.grad if item.grad is not None else torch.zeros_like(item)
        metric(f"preprocessing_backward/{channel}/{key}", gradients[key], expected, 1e-5, 5e-4)


@pytest.mark.parametrize("count", [1, 2])
def test_screen_compositing_backward(diagnostic, count):
    params, camera = scene(count)
    means, quats, scales, opacity, colors = physical(params)
    view, K, h, w, bg = camera
    p = project_legacy(means, covariance_3d(scales, quats), view, K, h, w)
    inputs = {"mean": p.mean, "conic": conic_entries(p.inverse), "opacities": opacity, "colors": colors}
    gpu = {key: value.float().cuda().contiguous() for key, value in inputs.items()}
    cpu = {key: value.cpu().double().requires_grad_() for key, value in gpu.items()}
    expected = composite_reference(cpu["mean"], cpu["conic"], cpu["opacities"], cpu["colors"], h, w, bg)
    loss(expected.image).backward()
    actual, gradients = diagnostic.composite(
        gpu["mean"],
        gpu["conic"],
        gpu["opacities"],
        gpu["colors"],
        h,
        w,
        bg.float().cuda(),
        weights_for(expected.image).float().cuda(),
    )
    for key in ("image", "transmittance"):
        metric(f"screen/n={count}/{key}", actual[key], getattr(expected, key), 1e-6, 1e-5)
    for key in inputs:
        metric(f"screen_backward/n={count}/{key}", gradients[key], cpu[key].grad, 1e-5, 5e-4)


SPEC = json.loads((Path(__file__).parent / "baselines/phase0/scenes.json").read_text())


@pytest.mark.parametrize("case", SPEC["scenes"], ids=lambda case: case["name"])
def test_public_forward_independent_reference(diagnostic, case):
    params = {key: torch.tensor(value, dtype=torch.float32, device="cuda") for key, value in case["params"].items()}
    camera = (
        torch.tensor(SPEC["w2c"], device="cuda"),
        torch.tensor(SPEC["K"], dtype=torch.float32, device="cuda"),
        case["height"],
        case["width"],
        torch.tensor(case["background"], device="cuda"),
    )
    camera = tuple(v.float() if isinstance(v, torch.Tensor) else v for v in camera)
    images = [WarpRenderer().render(GaussianModel(**params), *camera).image.detach().cpu().double() for _ in range(5)]
    image = images[0]
    expected = reference(on_cpu(params), cpu_camera(camera)).image
    metric(f"public_forward/{case['name']}", image, expected, 1e-6, 1e-5)
    variation = max((item - image).abs().max().item() for item in images)
    print("MEASUREMENT " + json.dumps({"name": f"repeat/public_forward/{case['name']}", "repeats": 5, "max_abs": variation}))
    for repeated in images[1:]:
        torch.testing.assert_close(repeated, image, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("count", [1, 2])
def test_public_backward_and_finite_differences(diagnostic, count, config=None):
    params, camera = scene(count, dtype=torch.float32, device="cuda")
    renderer = WarpRenderer(config=config)
    settings = {} if config is None else config.effective_values()
    screen_settings = {
        k: v for k, v in settings.items() if k in ("gaussian_extent", "alpha_cutoff", "alpha_cap", "transmittance_threshold")
    }
    label = "" if config is None else "configured/"
    cpu_params = {key: value.requires_grad_() for key, value in on_cpu(params).items()}
    loss(reference(cpu_params, cpu_camera(camera), **settings).image).backward()
    expected = {key: value.grad for key, value in cpu_params.items()}
    repeats = []
    for _ in range(5):
        model = GaussianModel(**{key: value.clone() for key, value in params.items()})
        loss(renderer.render(model, *camera).image).backward()
        repeats.append({key: value.grad.cpu().double() for key, value in model.params.items()})
    for key in KEYS:
        metric(f"{label}public_backward/n={count}/{key}", repeats[0][key], expected[key], 1e-5, 5e-4)
        variation = max((r[key] - repeats[0][key]).abs().max().item() for r in repeats)
        print("MEASUREMENT " + json.dumps({"name": f"{label}repeat/n={count}/{key}", "repeats": 5, "max_abs": variation}))
        for repeated in repeats[1:]:
            torch.testing.assert_close(repeated[key], repeats[0][key], atol=1e-5, rtol=5e-4)

    def evaluate(values):
        with torch.no_grad():
            image = renderer.render(GaussianModel(**values), *camera).image
            ref = reference(on_cpu(values), cpu_camera(camera), **settings)
            branches = signature(ref)
            means, quats, scales, opacity, colors = physical(values)
            state = diagnostic.preprocess(means, scales, quats, camera, config=config)
            assert torch.all(state["touched"] == 1), "FD fixture must stay in its single tile"
            branches.update(tile_rect=state["rect"], tiles_touched=state["touched"])
            # Also audit branches using actual Warp-projected screen quantities.
            screen = composite_reference(
                state["mean"],
                state["conic"],
                opacity.cpu().double(),
                colors.cpu().double(),
                camera[2],
                camera[3],
                camera[4].cpu().double(),
                **screen_settings,
            )
            branches.update({"warp_screen_" + k: v for k, v in signature(screen).items()})
            observed, _ = diagnostic.composite(
                state["mean"].float().cuda(),
                state["conic"].float().cuda(),
                opacity,
                colors,
                camera[2],
                camera[3],
                camera[4],
                config=config,
            )
            h, w = image.shape[:2]
            branches["warp_last"] = observed["last"][h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1]
            return loss(image.double()).item(), branches

    steps = [3e-3, 1e-3, 3e-4]
    differences = finite_differences(params, evaluate, steps)
    check_differences(f"{label}warp_fd/n={count}", differences, repeats[0], steps, 2e-4, 2e-3)


@pytest.mark.parametrize("distance,opacity", [(3.32, 1.0), (3.34, 1.0), (0, 0.0038), (0, 0.004), (0, 0.98), (0, 0.999)])
def test_support_cutoff_cap_behavior(diagnostic, distance, opacity):
    mean = torch.tensor([[0.5 - distance, 0.5]], device="cuda")
    conic = torch.tensor([[1.0, 0.0, 1.0]], device="cuda")
    alpha, color, bg = (
        torch.tensor([opacity], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        torch.zeros(3, device="cuda"),
    )
    actual, _ = diagnostic.composite(mean, conic, alpha, color, 1, 1, bg)
    expected = composite_reference(
        mean.cpu().double(), conic.cpu().double(), alpha.cpu().double(), color.cpu().double(), 1, 1, bg.cpu().double()
    )
    metric(f"boundary/distance={distance}/opacity={opacity}", actual["image"], expected.image, 1e-6, 1e-5)
    contributing = distance == 3.32 or (distance == 0 and opacity >= 0.004)
    assert bool(actual["image"][0, 0, 0] > 0) == contributing
    if opacity == 0.999:
        torch.testing.assert_close(actual["image"][0, 0, 0], torch.tensor(0.99, dtype=torch.float64), atol=1e-6, rtol=0)


@pytest.mark.parametrize("opacity,last", [(0.95, 3), (0.96, 2)])
def test_early_exit_behavior(diagnostic, opacity, last):
    mean = torch.full((4, 2), 0.5, device="cuda")
    conic = torch.tensor([[1.0, 0.0, 1.0]] * 4, device="cuda")
    alpha, color, bg = torch.full((4,), opacity, device="cuda"), torch.eye(4, 3, device="cuda"), torch.ones(3, device="cuda")
    actual, _ = diagnostic.composite(mean, conic, alpha, color, 1, 1, bg)
    expected = composite_reference(
        mean.cpu().double(), conic.cpu().double(), alpha.cpu().double(), color.cpu().double(), 1, 1, bg.cpu().double()
    )
    assert actual["last"].item() == last
    for key in ("image", "transmittance"):
        metric(f"early_exit/{opacity}/{key}", actual[key], getattr(expected, key), 1e-6, 1e-5)


@pytest.mark.parametrize("depth,visible", [(0.099, False), (0.101, True)])
def test_near_plane_behavior(diagnostic, depth, visible):
    params, camera = scene(1, dtype=torch.float32, device="cuda")
    params["means"][:] = torch.tensor([0.0, 0.0, depth], device="cuda")
    means, quats, scales, _, _ = physical(params)
    state = diagnostic.preprocess(means, scales, quats, camera)
    assert bool(state["touched"].item() > 0) == visible
