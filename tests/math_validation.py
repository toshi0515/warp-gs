"""Fixed cases and measurement helpers shared by Phase 1 CPU/CUDA tests."""

import json

import torch

from warpgs.reference_math import render_legacy_reference

KEYS = ("means", "quats", "scales", "opacities", "colors")


def scene(count=2, *, dtype=torch.float64, device="cpu", height=12, width=12):
    # Round source inputs once to float32, then promote the same values for CPU references.
    raw = {
        "means": [[0.04, -0.03, 2.3], [-0.11, 0.07, 3.1]],
        "quats": [[0.9, 0.2, 0.3, 0.1], [0.8, -0.2, 0.1, 0.4]],
        "scales": torch.tensor([[0.22, 0.32, 0.17], [0.31, 0.17, 0.24]]).log(),
        "opacities": [0.4, -0.2],
        "colors": [[0.7, -0.4, 0.2], [-0.3, 0.6, 0.1]],
    }
    params = {
        key: torch.as_tensor(value, dtype=torch.float32)[:count].to(dtype=dtype, device=device).clone()
        for key, value in raw.items()
    }
    camera = (
        torch.eye(4, dtype=dtype, device=device),
        torch.tensor([[14, 0, width / 2], [0, 16, height / 2], [0, 0, 1]], dtype=dtype, device=device),
        height,
        width,
        torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32).to(dtype=dtype, device=device),
    )
    return params, camera


def physical(params):
    return (
        params["means"],
        params["quats"],
        params["scales"].exp(),
        params["opacities"].sigmoid(),
        params["colors"].sigmoid(),
    )


def reference(params, camera, **settings):
    return render_legacy_reference(*physical(params), *camera, **settings)


def weights_for(image):
    weights = torch.zeros_like(image)
    h, w = image.shape[:2]
    # Identical rounded cotangents for CPU float64, CUDA backward and CUDA FD.
    values = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3)
    values = (values / values.sum()).to(dtype=image.dtype, device=image.device)
    weights[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = values
    return weights


def loss(image):
    return (image * weights_for(image)).sum()


def signature(result):
    h, w = result.image.shape[:2]
    return {
        key: (value[:, h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] if value.ndim == 3 else value).detach().cpu()
        for key, value in result.branches.items()
    }


def assert_same_branches(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), f"Finite difference crossed branch: {key}"


def metric(name, actual, expected, atol, rtol):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    error = (actual - expected).abs()
    nonzero = expected.abs() > atol
    data = {
        "name": name,
        "max_abs": error.max().item(),
        "max_relative_above_atol": (error[nonzero] / expected[nonzero].abs()).max().item() if nonzero.any() else None,
        "near_zero_components": int((~nonzero).sum()),
        "atol": atol,
        "rtol": rtol,
    }
    print("MEASUREMENT " + json.dumps(data))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def finite_differences(params, evaluate, steps):
    """All coordinates and all steps; use the actual representable +/- spacing."""
    _, initial_signature = evaluate(params)
    results = {key: [] for key in KEYS}
    for factor in steps:
        for key in KEYS:
            derivative = torch.zeros_like(params[key])
            for index in range(params[key].numel()):
                base = params[key].flatten()[index].item()
                h = factor * max(1.0, abs(base))
                samples, endpoints = [], []
                for sign in (-1, 1):
                    perturbed = {k: v.detach().clone() for k, v in params.items()}
                    perturbed[key].view(-1)[index] = base + sign * h
                    value, branches = evaluate(perturbed)
                    assert_same_branches(branches, initial_signature)
                    samples.append(value)
                    endpoints.append(perturbed[key].flatten()[index].item())
                derivative.view(-1)[index] = (samples[1] - samples[0]) / (endpoints[1] - endpoints[0])
            results[key].append(derivative.detach().cpu().double())
    return results


def check_differences(name, differences, gradients, steps, atol, rtol):
    for key in KEYS:
        values = differences[key]
        # Select by FD stability alone, before looking at the analytic gradient.
        stable = [i for i in range(len(values) - 1) if torch.allclose(values[i], values[i + 1], atol=atol, rtol=rtol)]
        print(
            "MEASUREMENT "
            + json.dumps(
                {
                    "name": f"{name}/{key}/step_sweep",
                    "steps": steps,
                    "finite_differences": [v.tolist() for v in values],
                    "backward": gradients[key].detach().cpu().double().tolist(),
                    "stable_pairs": stable,
                }
            )
        )
        assert stable, f"No adjacent stable FD steps: {name}/{key}"
        for index in (stable[0], stable[0] + 1):
            metric(f"{name}/{key}/h={steps[index]}", values[index], gradients[key], atol, rtol)
