"""Opt-in settings: effects, independent reference, gradients and graph isolation."""

import json
from dataclasses import asdict, replace

import pytest
import torch
from math_validation import KEYS, metric, physical, reference, scene
from test_warp_math import cpu_camera, on_cpu
from test_warp_math import test_public_backward_and_finite_differences as check_gradient_sweep

from warpgs.model import GaussianModel
from warpgs.raster_config import RasterConfig
from warpgs.reference_math import conic_entries, covariance_3d, project_legacy
from warpgs.renderers import WarpRenderer

pytestmark = pytest.mark.cuda
MIXED = RasterConfig(
    gaussian_extent=2.8,
    covariance_lowpass=0.6,
    jacobian_clamp_factor=1.1,
    alpha_cutoff=0.01,
    alpha_cap=0.9,
    transmittance_threshold=0.001,
    near_plane=0.2,
)
CHANGES = dict(
    gaussian_extent=2.5,
    covariance_lowpass=0.6,
    jacobian_clamp_factor=1.0,
    alpha_cutoff=0.02,
    alpha_cap=0.8,
    transmittance_threshold=0.01,
    near_plane=0.2,
)


@pytest.fixture(scope="module")
def diagnostic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch unavailable")
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp CUDA unavailable")
    import warp_diagnostics

    return warp_diagnostics


def render_graph(params, camera, renderer):
    model = GaussianModel(**{key: value.clone() for key, value in params.items()})
    image = renderer.render(model, *camera).image
    return image, model


def backward(image, model):
    image.mean().backward()
    return {key: value.grad.detach().cpu().double().clone() for key, value in model.params.items()}


def compare_reference(name, params, camera, config):
    image, model = render_graph(params, camera, WarpRenderer(config))
    actual_grad = backward(image, model)
    cpu = {key: value.requires_grad_() for key, value in on_cpu(params).items()}
    expected = reference(cpu, cpu_camera(camera), **config.effective_values()).image
    metric(f"settings/{name}/image", image, expected, 1e-6, 1e-5)
    if expected.requires_grad:
        expected.mean().backward()
    for key in KEYS:
        grad = cpu[key].grad if cpu[key].grad is not None else torch.zeros_like(cpu[key])
        metric(f"settings/{name}/grad_{key}", actual_grad[key], grad, 1e-5, 5e-4)
    return image.detach().cpu().double(), actual_grad


def effect_scene(field):
    params, camera = scene(1, dtype=torch.float32, device="cuda")
    if field in ("alpha_cutoff", "alpha_cap", "near_plane", "transmittance_threshold"):
        n = 4 if field == "transmittance_threshold" else 1
        params = {key: value.repeat((n,) + (1,) * (value.ndim - 1)) for key, value in params.items()}
        params["means"][:] = torch.tensor([[0.0, 0.0, 2.0 + 0.2 * i] for i in range(n)], device="cuda")
        params["quats"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
        params["scales"][:] = torch.tensor(0.1, device="cuda").log()
        opacity = {"alpha_cutoff": 0.01, "alpha_cap": 0.9, "near_plane": 0.6, "transmittance_threshold": 0.95}[field]
        params["opacities"][:] = torch.tensor(opacity, device="cuda").logit()
        if field == "near_plane":
            params["means"][:, 2] = 0.15
        if n > 1:
            params["colors"] = torch.tensor(
                [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [0.3, -0.2, 0.7]], device="cuda"
            )
        cx = 0.4 if field == "alpha_cap" else 0.5
        camera = (
            camera[0],
            torch.tensor([[10.0, 0.0, cx], [0.0, 10.0, 0.5], [0.0, 0.0, 1.0]], device="cuda"),
            1,
            1,
            camera[4],
        )
    elif field == "gaussian_extent":
        params["means"][:] = torch.tensor([0.0, 0.0, 2.0], device="cuda")
        params["quats"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
        params["scales"][:] = torch.tensor([0.5, 0.1, 0.1], device="cuda").log()
        params["opacities"][:] = torch.tensor(0.95, device="cuda").logit()
        camera = (
            camera[0],
            torch.tensor([[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]], device="cuda"),
            32,
            64,
            camera[4],
        )
    elif field == "jacobian_clamp_factor":
        params["means"][:, 0] = params["means"][:, 2] * (0.55 * camera[3] / camera[1][0, 0])
    return params, camera


@pytest.mark.parametrize("field", CHANGES)
def test_each_setting_has_effect_and_matches_reference(diagnostic, field):
    params, camera = effect_scene(field)
    default_image, default_grads = compare_reference(field + "/default", params, camera, RasterConfig())
    config = replace(RasterConfig(), **{field: CHANGES[field]})
    image, gradients = compare_reference(field + "/changed", params, camera, config)
    change = (image - default_image).abs().max().item()
    assert change > 1e-6, f"{field} did not affect its dedicated input"
    print(
        "MEASUREMENT "
        + json.dumps(
            {
                "name": f"effect/{field}",
                "max_abs_change": change,
                "requested": asdict(config),
                "effective": config.effective_values(),
            }
        )
    )
    means, quats, scales, opacity, colors = physical(params)
    if field == "gaussian_extent":
        original = diagnostic.preprocess(means, scales, quats, camera)
        altered = diagnostic.preprocess(means, scales, quats, camera, config=config)
        assert original["rect"].tolist() == [[0, 0, 2, 1]]
        assert altered["rect"].tolist() == [[0, 0, 1, 1]]
        assert original["touched"].item() == 6 and altered["touched"].item() == 4
    if field in ("covariance_lowpass", "jacobian_clamp_factor"):
        original, _ = diagnostic.projection(means, scales, quats, camera)
        actual, _ = diagnostic.projection(means, scales, quats, camera, config=config)
        assert not torch.equal(original["diagonal"], actual["diagonal"])
        c = cpu_camera(camera)
        projected = project_legacy(
            means.cpu().double(),
            covariance_3d(scales.cpu().double(), quats.cpu().double()),
            c[0],
            c[1],
            c[2],
            c[3],
            covariance_lowpass=config.effective_values()["covariance_lowpass"],
            jacobian_clamp_factor=config.effective_values()["jacobian_clamp_factor"],
        )
        metric(f"settings/{field}/diagonal", actual["diagonal"], projected.covariance.diagonal(dim1=-2, dim2=-1), 1e-6, 1e-5)
        metric(f"settings/{field}/conic", actual["conic"], conic_entries(projected.inverse), 1e-6, 1e-5)
    if field == "alpha_cap":
        assert gradients["opacities"].abs().max() == 0
        assert gradients["means"].abs().max() == 0
        assert default_grads["opacities"].abs().max() > 0 and default_grads["means"].abs().max() > 0
    if field == "near_plane":
        assert all(torch.count_nonzero(value) == 0 for value in gradients.values())
    if field == "transmittance_threshold":
        state = diagnostic.preprocess(means, scales, quats, camera)
        old, _ = diagnostic.composite(
            state["mean"].float().cuda(), state["conic"].float().cuda(), opacity, colors, camera[2], camera[3], camera[4]
        )
        new, _ = diagnostic.composite(
            state["mean"].float().cuda(),
            state["conic"].float().cuda(),
            opacity,
            colors,
            camera[2],
            camera[3],
            camera[4],
            config=config,
        )
        assert old["last"].item() == 3 and new["last"].item() == 1
        expected_t = (1 - float(opacity[0])) ** 2
        metric("settings/early_exit/residual_T", new["transmittance"], torch.full((1, 1), expected_t), 1e-6, 1e-5)


def test_default_equivalence_and_interleaved_backward(diagnostic):
    params, camera = scene(dtype=torch.float32, device="cuda")
    separate = {}
    for name, renderer in (
        ("implicit", WarpRenderer()),
        ("default", WarpRenderer(RasterConfig())),
        ("mixed", WarpRenderer(MIXED)),
    ):
        image, model = render_graph(params, camera, renderer)
        separate[name] = (image.detach().cpu().double(), backward(image, model))
    for key in KEYS:
        metric(f"default_equivalence/grad_{key}", separate["implicit"][1][key], separate["default"][1][key], 1e-5, 5e-4)
    metric("default_equivalence/image", separate["implicit"][0], separate["default"][0], 1e-6, 1e-5)
    renderer = WarpRenderer()
    a, model_a = render_graph(params, camera, renderer)
    renderer.config = MIXED  # Existing graph must retain its original settings.
    b, model_b = render_graph(params, camera, renderer)
    grad_b, grad_a = backward(b, model_b), backward(a, model_a)
    for name, image, gradients in (("default", a, grad_a), ("mixed", b, grad_b)):
        metric(f"interleaved/{name}/image", image, separate[name][0], 1e-6, 1e-5)
        for key in KEYS:
            metric(f"interleaved/{name}/grad_{key}", gradients[key], separate[name][1][key], 1e-5, 5e-4)
    renderer.config = RasterConfig()
    after, model_after = render_graph(params, camera, renderer)
    after_grad = backward(after, model_after)
    metric("default_after_mixed/image", after, separate["default"][0], 1e-6, 1e-5)
    for key in KEYS:
        metric(f"default_after_mixed/grad_{key}", after_grad[key], separate["default"][1][key], 1e-5, 5e-4)


def test_nondefault_autograd_and_finite_differences(diagnostic):
    print(
        "MEASUREMENT "
        + json.dumps({"name": "mixed_config", "requested": asdict(MIXED), "effective": MIXED.effective_values()})
    )
    check_gradient_sweep(diagnostic, 2, config=MIXED)


def test_legacy_ten_argument_autograd_call(diagnostic):
    from warpgs.my_warp_rasterizer import WarpRasterizeFunction

    params, camera = scene(dtype=torch.float32, device="cuda")
    model = GaussianModel(**params)
    image = WarpRasterizeFunction.apply(model.means, model.quats, model.scales, model.opacities, model.colors, *camera)
    gradients = backward(image, model)
    expected_image, expected_model = render_graph(params, camera, WarpRenderer())
    expected_grad = backward(expected_image, expected_model)
    metric("legacy_apply/image", image, expected_image, 1e-6, 1e-5)
    for key in KEYS:
        metric(f"legacy_apply/grad_{key}", gradients[key], expected_grad[key], 1e-5, 5e-4)


def test_zero_settings_match_reference(diagnostic):
    params, camera = scene(dtype=torch.float32, device="cuda")
    config = RasterConfig(covariance_lowpass=0, alpha_cutoff=0, transmittance_threshold=0, near_plane=0)
    compare_reference("zero_controls", params, camera, config)
    # With four opaque layers, disabling early exit retains the fourth contributor.
    p, c = effect_scene("transmittance_threshold")
    p["opacities"][:] = torch.tensor(0.96, device="cuda").logit()
    means, quats, scales, opacity, colors = physical(p)
    state = diagnostic.preprocess(means, scales, quats, c)
    old, _ = diagnostic.composite(state["mean"].float().cuda(), state["conic"].float().cuda(), opacity, colors, 1, 1, c[4])
    new, _ = diagnostic.composite(
        state["mean"].float().cuda(), state["conic"].float().cuda(), opacity, colors, 1, 1, c[4], config=config
    )
    assert old["last"].item() == 2 and new["last"].item() == 3
