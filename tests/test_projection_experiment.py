import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from warpgs import projection_experiment
from warpgs.projection_experiment import (
    K,
    SamplingIncomplete,
    conditioned_moments,
    make_cases,
    measure_case,
    predictions,
    project_samples,
    sample_conditioned,
    summarize_repeats,
)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("case_id", [0, 17, 27])
def test_second_order_correction_matches_independent_hessian(case_id):
    case = make_cases()[case_id]
    mean, covariance = torch.tensor(case.mean), torch.tensor(case.covariance)
    expected = []
    for index in range(2):

        def direct(point, index=index):
            return K[index, index] * point[index] / point[2] + K[index, 2]

        hessian = torch.autograd.functional.hessian(direct, mean)
        expected.append((direct(mean) + 0.5 * torch.trace(hessian @ covariance)).item())
    close(predictions(case.mean, case.covariance)["mean_second"], expected)


def test_half_normal_conditional_moments_with_correlated_coordinates():
    # Z=0.1+0.2*G; X=1+0.1*G+0.3*E1; Y=2-0.06*G+0.4*E2.
    # Conditioning on G>0: E[G]=sqrt(2/pi), Var[G]=1-2/pi.
    mean = np.array([1.0, 2.0, 0.1])
    column = np.array([0.1, -0.06, 0.2])
    residual = np.diag([0.09, 0.16, 0])
    covariance = residual + np.outer(column, column)
    actual_mean, actual_covariance, probability = conditioned_moments(mean, covariance, 0.1)
    close(probability, 0.5)
    close(actual_mean, mean + column * math.sqrt(2 / math.pi))
    close(actual_covariance, residual + np.outer(column, column) * (1 - 2 / math.pi))


def test_fixed_depth_projection_has_exact_linear_moments_and_keeps_offscreen_points():
    points = np.array([[-8.0, 0, 2], [8.0, 0, 2], [0, -6.0, 2], [0, 6.0, 2]])
    projected = project_samples(points)
    expected = np.array([[-1680, 240], [2320, 240], [320, -1410], [320, 1890]])
    close(projected, expected)
    prediction = predictions(points.mean(axis=0), np.cov(points, rowvar=False))
    close(prediction["mean_first"], projected.mean(axis=0))
    close(prediction["mean_second"], projected.mean(axis=0))
    close(prediction["covariance_first"], np.cov(projected, rowvar=False))


def test_reflection_and_correlation_sign():
    mean = np.array([0.0, 0.0, 2.0])
    covariance = np.diag([0.04, 0.09, 0.16])
    covariance[0, 2] = covariance[2, 0] = 0.02
    positive = predictions(mean, covariance)
    reflection = np.diag([-1.0, 1, 1])
    negative = predictions(reflection @ mean, reflection @ covariance @ reflection)
    assert positive["mean_second"][0] < K[0, 2] < negative["mean_second"][0]
    close(positive["mean_second"][0] + negative["mean_second"][0], 2 * K[0, 2])
    close(positive["mean_second"][1], K[1, 2])
    points = np.array([[0.3, -0.2, 1.5], [-0.1, 0.6, 2.4]])
    close(project_samples(points @ reflection)[:, 0], 2 * K[0, 2] - project_samples(points)[:, 0])


@pytest.mark.parametrize("factor", [0.01, 100.0])
def test_world_scale_and_near_invariance(factor):
    case = make_cases()[40]
    cm, cc, p = conditioned_moments(case.mean, case.covariance, case.near)
    sm, sc, sp = conditioned_moments(case.mean * factor, case.covariance * factor**2, case.near * factor)
    close(sm / factor, cm)
    close(sc / factor**2, cc)
    close(sp, p)
    for key, value in predictions(cm, cc).items():
        close(predictions(sm, sc)[key], value)
    samples, _ = sample_conditioned(case.mean, case.covariance, case.near, 4096, np.random.default_rng(90))
    scaled, _ = sample_conditioned(
        case.mean * factor, case.covariance * factor**2, case.near * factor, 4096, np.random.default_rng(90)
    )
    close(project_samples(scaled), project_samples(samples))


def test_case_manifest_is_exact_and_positive_definite():
    cases = make_cases()
    assert len(cases) == 43
    assert [c.id for c in cases] == list(range(43))
    for case in cases:
        assert np.linalg.eigvalsh(case.covariance).min() > 0
        _, _, probability = conditioned_moments(case.mean, case.covariance, case.near)
        assert probability > 0.5
    close([c.value for c in cases if c.family == "depth"], [0.25, 0.5, 1, 2, 4])
    for c in cases[:5]:
        close(np.diag(c.covariance), np.square([0.08, 0.12, 0.1]))
    for c in cases[10:15]:
        std = np.sqrt(np.diag(c.covariance))
        close(std[0] / std[1], c.value)
        close(std[0] * std[1], 0.01)
    for c in cases[15:25]:
        close(np.linalg.eigvalsh(c.covariance), np.sort(np.square([0.08, 0.12, 0.4])))


def test_samples_agree_with_conditional_analytic_moments_in_standard_errors():
    mean = np.array([0.2, -0.1, 0.12])
    factor = np.array([[0.03, 0, 0.02], [0, 0.04, -0.01], [0, 0, 0.05]])
    covariance = factor @ factor.T
    expected_mean, expected_covariance, probability = conditioned_moments(mean, covariance, 0.1)
    means, covariances = [], []
    proposed = accepted = 0
    for repeat in range(16):
        samples, counts = sample_conditioned(mean, covariance, 0.1, 8192, np.random.default_rng([504, repeat]))
        assert np.all(samples[:, 2] > 0.1)
        means.append(samples.mean(axis=0))
        covariances.append(np.cov(samples, rowvar=False, ddof=1))
        proposed += counts["proposed"]
        accepted += counts["accepted_including_surplus"]
    # Fixed before measurement: six estimated SE, not a floating-point tolerance.
    for values, expected in ((means, expected_mean), (covariances, expected_covariance)):
        stats = summarize_repeats(values)
        assert np.all(np.abs(stats["estimate"] - expected) <= 6 * stats["mcse"])
    assert abs(accepted / proposed - probability) <= 6 * math.sqrt(probability * (1 - probability) / proposed)


def test_sampling_exhaustion_is_explicit():
    case = make_cases()[40]
    with pytest.raises(SamplingIncomplete) as error:
        sample_conditioned(case.mean, case.covariance, case.near, 100, np.random.default_rng(1), max_draws=10)
    assert error.value.counts["proposed"] == 10
    assert error.value.counts["accepted_including_surplus"] < 100


def test_exhausted_experiment_records_incomplete_without_plotting(monkeypatch, tmp_path):
    def exhausted(*args, **kwargs):
        raise SamplingIncomplete({"proposed": 10, "accepted_including_surplus": 0, "requested": 4096, "max_draws": 10})

    monkeypatch.setattr(projection_experiment, "sample_conditioned", exhausted)
    report = projection_experiment.run("quick", 20261030, tmp_path)
    assert report["status"] == "incomplete"
    assert all(c["status"] == "incomplete" for c in report["cases"])
    saved = json.loads((tmp_path / "measurements.json").read_text())
    assert saved["cases"][0]["failure"]["proposed"] == 10
    assert not (tmp_path / "convergence.png").exists()


@pytest.mark.parametrize("near", [0, -1, math.nan, math.inf])
def test_invalid_depth_condition_is_rejected(near):
    case = make_cases()[0]
    with pytest.raises(ValueError):
        conditioned_moments(case.mean, case.covariance, near)


@pytest.mark.parametrize("covariance", [np.zeros((3, 3)), np.diag([1, 1, -1]), np.full((3, 3), math.nan)])
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        conditioned_moments(np.array([0.0, 0, 1]), covariance, 0.1)


def test_extreme_conditioning_is_not_silently_approximated():
    with pytest.raises(ValueError, match="outside the study domain"):
        conditioned_moments(np.array([0.0, 0, 1]), np.eye(3), 20)


def test_prefix_convergence_uses_paired_repeat_differences():
    result = measure_case(make_cases()[9], 20261030, (256, 1024))
    actual = result["summaries"][1]["paired_change_from_previous_size"]
    for quantity in ("mean", "covariance"):
        differences = np.array(
            [r["measurements"][1][quantity] - r["measurements"][0][quantity] for r in result["repetitions"]]
        )
        close(actual[quantity]["mcse"], differences.std(axis=0, ddof=1) / 4)


def test_cli_cpu_only_reproducible_and_explicit_overwrite(tmp_path):
    # Both subprocesses actually execute the quick experiment and create plots.
    # Disallow backend imports, and hide CUDA hardware without mocking the experiment.
    script = """
import sys, runpy, torch
class BlockBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('warp', 'gsplat'):
            raise AssertionError('GPU backend imported: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
from warpgs.projection_experiment import main
code = main()
assert not torch.cuda.is_initialized()
sys.exit(code)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    reports = []
    for name in ("first", "second"):
        output = tmp_path / name
        command = [sys.executable, "-c", script, "--preset", "quick", "--output", str(output)]
        subprocess.run(command, check=True, env=env, capture_output=True, text=True)
        reports.append(json.loads((output / "measurements.json").read_text()))
        assert (output / "summary.csv").is_file()
        assert (output / "convergence.svg").is_file()
    assert reports[0]["cases"] == reports[1]["cases"]
    assert reports[0]["status"] == reports[1]["status"] == "complete"
    before = (output / "measurements.json").read_bytes()
    refused = subprocess.run(command, env=env, capture_output=True, text=True)
    assert refused.returncode != 0 and "--overwrite" in refused.stderr
    assert (output / "measurements.json").read_bytes() == before
    subprocess.run([*command, "--overwrite"], check=True, env=env, capture_output=True, text=True)
    assert json.loads((output / "measurements.json").read_text())["cases"] == reports[0]["cases"]
