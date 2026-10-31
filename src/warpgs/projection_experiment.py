"""CPU float64 study of a depth-conditioned point distribution, not a renderer.

Camera coordinates have positive z; distances use world units and projections
use pixels. No image clipping, ray integration, alpha, clamp or low-pass filter.
Run explicitly with ``python -m warpgs.projection_experiment --output ...``.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from warpgs.reference_math import project_pinhole

K = np.array([[500.0, 0, 320], [0, 550.0, 240], [0, 0, 1]], dtype=np.float64)
REPEATS = 16
T_CRITICAL = 2.131  # NIST two-sided 95%, df=15; approximate intervals, not simultaneous.
SAMPLE_SIZES = (4096, 16384, 65536)


@dataclass
class Case:
    id: int
    family: str
    value: float
    mean: np.ndarray
    covariance: np.ndarray
    near: float = 0.1


def validate_distribution(mean, covariance, near):
    mean, covariance = np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    if mean.shape != (3,) or covariance.shape != (3, 3):
        raise ValueError("Expected mean (3,) and covariance (3,3)")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not math.isfinite(near):
        raise ValueError("Finite inputs required")
    if near <= 0 or mean[2] <= 0:
        raise ValueError("Strictly positive near and mean depth required")
    if not np.allclose(covariance, covariance.T, atol=1e-14, rtol=1e-14):
        raise ValueError("Symmetric covariance required")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Positive definite covariance required")
    return mean, covariance


def conditioned_moments(mean, covariance, near):
    """Exact first two moments of N(mean,covariance) conditional on Z>near.

    a=(near-mu_z)/sigma_z, lambda=phi(a)/P(Z>near).
    E[P|Z>near]=mu+Sigma[:,z]/sigma_z*lambda.
    Cov[P|Z>near]=Sigma+outer(Sigma[:,z],Sigma[z,:])/sigma_z^2
                        *(a*lambda-lambda^2).
    Extreme upper-tail conditioning is explicitly outside this small study.
    """
    mean, covariance = validate_distribution(mean, covariance, near)
    sigma = math.sqrt(covariance[2, 2])
    a = (near - mean[2]) / sigma
    probability = 0.5 * math.erfc(a / math.sqrt(2))
    if probability < 1e-12:
        raise ValueError("Conditioning probability below 1e-12 is outside the study domain")
    mills = math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / probability
    column = covariance[:, 2] / sigma
    conditional_mean = mean + column * mills
    conditional_covariance = covariance + np.outer(column, column) * (a * mills - mills**2)
    return conditional_mean, conditional_covariance, probability


def project_samples(points, intrinsics=K):
    """Direct nonlinear point projection, independent of the Jacobian prediction."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("Finite Nx3 points with positive depth required")
    return points[:, :2] / points[:, 2, None] * np.diag(intrinsics)[:2] + intrinsics[:2, 2]


def predictions(mean, covariance, intrinsics=K):
    """First-order moments and second-order mean only; no Gaussian closure claim."""
    mean, covariance = np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    projection = project_pinhole(
        torch.from_numpy(mean[None]),
        torch.from_numpy(covariance[None]),
        torch.eye(4, dtype=torch.float64),
        torch.as_tensor(intrinsics, dtype=torch.float64),
    )
    correction = np.diag(intrinsics)[:2] * (mean[:2] * covariance[2, 2] / mean[2] ** 3 - covariance[:2, 2] / mean[2] ** 2)
    first_mean = projection.mean[0].numpy()
    return {
        "mean_first": first_mean,
        "mean_second": first_mean + correction,
        "covariance_first": projection.covariance[0].numpy(),
    }


def make_cases():
    cases = []

    def add(family, value, *, depth=2.0, scales=None, covariance=None, slope_x=0.2):
        if covariance is None:
            if scales is None:
                scales = np.array([0.04, 0.06, 0.05]) * depth
            covariance = np.diag(np.square(scales))
        cases.append(Case(len(cases), family, value, np.array([slope_x * depth, -0.1 * depth, depth]), covariance))

    for d in (0.25, 0.5, 1, 2, 4):
        add("depth", d, depth=d, scales=[0.08, 0.12, 0.1])
    for ratio in (0.01, 0.03, 0.1, 0.2, 0.4):
        add("depth_spread", ratio, scales=[0.08, 0.12, 2 * ratio])
    for ratio in (0.25, 0.5, 1, 2, 4):
        add("anisotropy", ratio, scales=[0.1 * math.sqrt(ratio), 0.1 / math.sqrt(ratio), 0.1])
    for axis in ("x", "y"):
        for angle in (-60, -30, 0, 30, 60):
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            rotation = (
                np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            )
            add(f"rotation_{axis}", angle, covariance=rotation @ np.diag(np.square([0.08, 0.12, 0.4])) @ rotation.T)
    for axis, index in (("xz", 0), ("yz", 1)):
        for rho in (-0.6, -0.3, 0, 0.3, 0.6):
            scales = np.array([0.08, 0.12, 0.1])
            covariance = np.diag(scales**2)
            covariance[index, 2] = covariance[2, index] = rho * scales[index] * scales[2]
            add(f"correlation_{axis}", rho, covariance=covariance)
    for slope in (0, 0.25, 0.5, 1, 2):
        add("off_axis", slope, slope_x=slope)
    for d in (0.11, 0.15, 0.2):
        add("near", d, depth=d, scales=[0.02, 0.03, 0.05])
    return cases


class SamplingIncomplete(RuntimeError):
    def __init__(self, counts):
        self.counts = counts
        super().__init__(f"Rejection sampling budget exhausted: {counts}")


def sample_conditioned(mean, covariance, near, count, rng, *, max_draws=None):
    """Rejection sampling with a fixed proposal cap; never clip sampled depths."""
    mean, covariance = validate_distribution(mean, covariance, near)
    if count < 2:
        raise ValueError("At least two accepted samples required")
    # All preset cases have acceptance > 0.5. A fixed cap also bounds custom calls.
    max_draws = 8 * count if max_draws is None else max_draws
    factor = np.linalg.cholesky(covariance)
    chunks, proposed, accepted = [], 0, 0
    while accepted < count and proposed < max_draws:
        size = min(16384, max_draws - proposed)
        points = rng.standard_normal((size, 3)) @ factor.T + mean
        valid = points[points[:, 2] > near]
        proposed += size
        accepted += len(valid)
        chunks.append(valid)
    counts = {"proposed": proposed, "accepted_including_surplus": accepted, "requested": count, "max_draws": max_draws}
    if accepted < count:
        raise SamplingIncomplete(counts)
    samples = np.concatenate(chunks)[:count]
    counts.update(used=count, minimum_used_depth=float(samples[:, 2].min()))
    return samples, counts


def summarize_repeats(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != REPEATS:
        raise ValueError("The fixed t coefficient requires exactly 16 independent repetitions")
    estimate = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    se = sd / math.sqrt(REPEATS)
    return {
        "estimate": estimate,
        "repeat_sd": sd,
        "mcse": se,
        "ci95_low": estimate - T_CRITICAL * se,
        "ci95_high": estimate + T_CRITICAL * se,
    }


def errors(prediction, mean_stats, covariance_stats):
    result = {}
    for name in ("mean_first", "mean_second", "covariance_first"):
        is_mean = name.startswith("mean")
        stats = mean_stats if is_mean else covariance_stats
        difference = prediction[name] - stats["estimate"]
        low, high = prediction[name] - stats["ci95_high"], prediction[name] - stats["ci95_low"]
        result[name] = {
            "signed_error": difference,
            "norm_error": float(np.linalg.norm(difference)),
            "error_ci95_low": low,
            "error_ci95_high": high,
            "component_resolved_from_zero": (low > 0) | (high < 0),
        }
        if not is_mean:
            denominator = np.linalg.norm(stats["estimate"])
            result[name]["relative_frobenius_error"] = float(np.linalg.norm(difference) / denominator)
    return result


def measure_case(case, seed, sizes):
    conditional_mean, conditional_covariance, probability = conditioned_moments(case.mean, case.covariance, case.near)
    predicted = {
        "original": predictions(case.mean, case.covariance),
        "conditioned": predictions(conditional_mean, conditional_covariance),
    }
    scales = np.sqrt(np.diag(case.covariance))
    result = {
        "id": case.id,
        "family": case.family,
        "value": case.value,
        "mean": case.mean,
        "covariance": case.covariance,
        "eigenvalues": np.linalg.eigvalsh(case.covariance),
        "correlation": case.covariance / np.outer(scales, scales),
        "near": case.near,
        "acceptance_probability": probability,
        "conditional_mean": conditional_mean,
        "conditional_covariance": conditional_covariance,
        "predictions": predicted,
        "repetitions": [],
        "summaries": [],
        "status": "complete",
    }
    for repeat in range(REPEATS):
        entropy = [seed, case.id, repeat]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
        try:
            points, counts = sample_conditioned(case.mean, case.covariance, case.near, max(sizes), rng)
        except SamplingIncomplete as exc:
            result.update(status="incomplete", failure={"repeat": repeat, **exc.counts})
            return result
        projected = project_samples(points)
        measurements = [
            {"count": n, "mean": projected[:n].mean(axis=0), "covariance": np.cov(projected[:n], rowvar=False, ddof=1)}
            for n in sizes
        ]
        result["repetitions"].append({"entropy": entropy, "sampling": counts, "measurements": measurements})
    previous_means = previous_covariances = None
    for index, n in enumerate(sizes):
        means = np.array([r["measurements"][index]["mean"] for r in result["repetitions"]])
        covariances = np.array([r["measurements"][index]["covariance"] for r in result["repetitions"]])
        mean_stats, covariance_stats = summarize_repeats(means), summarize_repeats(covariances)
        summary = {
            "count_per_repeat": n,
            "mean": mean_stats,
            "covariance": covariance_stats,
            "errors": {name: errors(value, mean_stats, covariance_stats) for name, value in predicted.items()},
        }
        if previous_means is not None:
            # Prefixes are correlated: use paired differences across independent repetitions.
            summary["paired_change_from_previous_size"] = {
                "mean": summarize_repeats(means - previous_means),
                "covariance": summarize_repeats(covariances - previous_covariances),
            }
        result["summaries"].append(summary)
        previous_means, previous_covariances = means, covariances
    return result


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def plot_results(cases, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c for c in cases if c["status"] == "complete"]
    families = list(dict.fromkeys(c["family"] for c in cases))
    labels = {
        "depth": "mean depth [world units]",
        "depth_spread": "sigma_z / mean depth",
        "anisotropy": "sigma_x / sigma_y",
        "rotation_x": "rotation about x [degrees]",
        "rotation_y": "rotation about y [degrees]",
        "correlation_xz": "rho_xz",
        "correlation_yz": "rho_yz",
        "off_axis": "mean x / mean depth",
        "near": "mean depth [world units]; near=0.1",
    }
    for quantity in ("mean", "covariance"):
        fig, axes = plt.subplots(
            math.ceil(len(families) / 3), 3, figsize=(15, 4 * math.ceil(len(families) / 3)), squeeze=False
        )
        for axis, family in zip(axes.flat, families, strict=False):
            rows = [c for c in cases if c["family"] == family]
            for origin in ("original", "conditioned"):
                orders = ("mean_first", "mean_second") if quantity == "mean" else ("covariance_first",)
                for order in orders:
                    field = "norm_error" if quantity == "mean" else "relative_frobenius_error"
                    y = [c["summaries"][-1]["errors"][origin][order][field] for c in rows]
                    axis.plot(
                        [c["value"] for c in rows],
                        y,
                        marker="o",
                        linestyle="--" if origin == "conditioned" else "-",
                        label=f"{origin}/{order}",
                    )
            axis.set(
                title=family,
                xlabel=labels[family],
                ylabel="mean L2 error [pixel]" if quantity == "mean" else "relative Frobenius error",
            )
            axis.grid(alpha=0.3)
        for axis in list(axes.flat)[len(families) :]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("Point estimates; sampling uncertainty is reported in convergence plots and measurements.json")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for extension in ("png", "svg"):
            fig.savefig(output / f"{quantity}_errors.{extension}", dpi=150)
        plt.close(fig)
    representatives = [next(c for c in cases if c["id"] == i) for i in (3, 9, 40) if any(c["id"] == i for c in cases)]
    fig, axes = plt.subplots(len(representatives), 2, figsize=(11, 3.5 * len(representatives)), squeeze=False)
    for row, case in zip(axes, representatives, strict=True):
        summaries = case["summaries"]
        sizes = [s["count_per_repeat"] for s in summaries]
        mean = np.array([s["mean"]["estimate"] for s in summaries])
        se = np.array([s["mean"]["mcse"] for s in summaries])
        for component, label in enumerate(("u", "v")):
            row[0].errorbar(
                sizes,
                mean[:, component] - case["predictions"]["original"]["mean_first"][component],
                yerr=T_CRITICAL * se[:, component],
                marker="o",
                label=label,
            )
        row[0].set(ylabel="MC mean - original first order [pixel]", title=f"{case['family']}={case['value']}")
        row[0].legend()
        row[1].plot(sizes, [np.linalg.norm(s["covariance"]["mcse"]) for s in summaries], marker="o")
        row[1].set(
            ylabel="Frobenius norm of covariance MCSE [pixel^2]", title="Sampling uncertainty (not approximation error)"
        )
        for axis in row:
            axis.set_xscale("log", base=2)
            axis.set_xlabel("accepted samples per independent repetition")
            axis.grid(alpha=0.3)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"convergence.{extension}", dpi=150)
    plt.close(fig)


def run(preset, seed, output, *, overwrite=False):
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; explicitly pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    sizes = SAMPLE_SIZES if preset == "full" else (4096,)
    if preset == "quick":
        cases = [c for c in cases if c.id in (3, 9, 40)]
    elif preset != "full":
        raise ValueError("preset must be quick or full")
    start = time.perf_counter()
    source_paths = [Path(__file__), Path(__file__).with_name("reference_math.py")]
    report = {
        "schema_version": 1,
        "preset": preset,
        "seed": seed,
        "rng": "NumPy PCG64 / SeedSequence([seed,case_id,repeat])",
        "dtype": "CPU float64",
        "repeats": REPEATS,
        "sample_sizes": sizes,
        "intrinsics": K,
        "interval": {
            "nominal_level": 0.95,
            "t_coefficient": T_CRITICAL,
            "df": 15,
            "scope": "approximate componentwise, not simultaneous; repeated estimators approximately normal",
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "source_sha256_lf": {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in source_paths},
        "command": [
            sys.executable,
            "-m",
            "warpgs.projection_experiment",
            "--preset",
            preset,
            "--seed",
            str(seed),
            "--output",
            str(output),
        ],
        "cases": [],
    }
    with (output / "run.log").open("w", encoding="utf-8") as log:
        for case in cases:
            result = measure_case(case, seed, sizes)
            report["cases"].append(result)
            line = f"case {case.id}: {case.family}={case.value}: {result['status']}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        report["measurement_seconds"] = time.perf_counter() - start
        report["status"] = "complete" if all(c["status"] == "complete" for c in report["cases"]) else "incomplete"
        (output / "measurements.json").write_text(
            json.dumps(report, default=json_value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        log.write(f"{report['status']}; measurement_seconds={report['measurement_seconds']:.3f}\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case", "family", "value", "count_per_repeat", "origin", "prediction", "norm_error", "relative_frobenius_error"]
        )
        for case in report["cases"]:
            for summary in case["summaries"]:
                for origin, values in summary["errors"].items():
                    for prediction, value in values.items():
                        writer.writerow(
                            [
                                case["id"],
                                case["family"],
                                case["value"],
                                summary["count_per_repeat"],
                                origin,
                                prediction,
                                value["norm_error"],
                                value.get("relative_frobenius_error", ""),
                            ]
                        )
    if report["status"] == "complete":
        plot_results(report["cases"], output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.preset, args.seed, args.output, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
