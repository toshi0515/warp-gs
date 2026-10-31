"""Phase 0 golden regression, NOT an independent mathematical correctness test.

Recording is explicit: python tests/test_warp_baseline.py --record [--overwrite]
Normal pytest execution never writes baseline files. No gsplat initialization here.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from warpgs.model import PARAM_KEYS, GaussianModel
from warpgs.renderers import WarpRenderer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tests/baselines/phase0"
SCENES_PATH = BASE / "scenes.json"
SPEC = json.loads(SCENES_PATH.read_text(encoding="utf-8"))
SCENES = SPEC["scenes"]
FORMULA = "max(4 * observed_max_abs_diff, 8 * float32_eps * max(1, reference_max_abs)); rtol=0"


def sha256(path):
    data = path.read_bytes()
    if path.suffix in {".py", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def command_output(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def environment():
    import warp as wp

    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warp": wp.__version__,
        "numpy": np.__version__,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "memory_bytes": properties.total_memory,
        "driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }


@pytest.fixture(scope="module")
def cuda_environment():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: Warp baseline requires CUDA PyTorch")
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp CUDA device unavailable")
    return environment()


def make_inputs(scene):
    params = {key: torch.tensor(scene["params"][key], dtype=torch.float32, device="cuda") for key in PARAM_KEYS}
    model = GaussianModel(**params)
    # Requires-grad flags audit the existing public bridge contract.
    w2c = torch.tensor(SPEC["w2c"], dtype=torch.float32, device="cuda", requires_grad=True)
    K = torch.tensor(SPEC["K"], dtype=torch.float32, device="cuda", requires_grad=True)
    bg = torch.tensor(scene["background"], dtype=torch.float32, device="cuda", requires_grad=True)
    return model, w2c, K, bg


def capture(scene):
    torch.manual_seed(SPEC["seed"])
    model, w2c, K, bg = make_inputs(scene)
    height, width = scene["height"], scene["width"]
    output = WarpRenderer().render(model, w2c, K, height, width, bg)
    assert output.alpha is None
    assert output.image.shape == (height, width, 3)
    assert output.image.dtype == torch.float32
    weights = torch.linspace(0.1, 1.0, height * width * 3, device="cuda").reshape(height, width, 3)
    (output.image * weights).sum().backward()
    assert w2c.grad is None and K.grad is None and bg.grad is None
    arrays = {"image": output.image.detach().cpu().numpy().copy()}
    for key in PARAM_KEYS:
        grad = model.params[key].grad
        assert grad is not None and grad.shape == model.params[key].shape
        arrays[f"grad_{key}"] = grad.detach().cpu().numpy().copy()
    assert all(np.isfinite(value).all() for value in arrays.values())
    if scene["expect_background"]:
        np.testing.assert_array_equal(arrays["image"], np.broadcast_to(bg.detach().cpu().numpy(), (height, width, 3)))
        assert all(np.count_nonzero(arrays[f"grad_{key}"]) == 0 for key in PARAM_KEYS)
    else:
        assert np.any(arrays["image"] != np.asarray(scene["background"], dtype=np.float32))
        assert np.count_nonzero(arrays["grad_means"]) > 0
    if scene["name"] == "rotated_anisotropic":
        assert np.count_nonzero(arrays["grad_quats"]) > 0
    return {f"{scene['name']}/{key}": value for key, value in arrays.items()}


def test_baseline_files_and_tolerance_evidence():
    """CPU-only integrity checks; no Warp import or CUDA initialization required."""
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["purpose"] == "existing-behavior regression; not mathematical correctness"
    assert metadata["scenes_sha256"] == sha256(SCENES_PATH)
    assert metadata["expected_sha256"] == sha256(BASE / "expected.npz")
    assert metadata["processes"] == 3 and metadata["repeats_per_process"] == 10
    assert metadata["tolerance_formula"] == FORMULA
    expected_keys = {f"{s['name']}/{key}" for s in SCENES for key in ["image", *(f"grad_{p}" for p in PARAM_KEYS)]}
    with np.load(BASE / "expected.npz", allow_pickle=False) as expected:
        assert set(expected.files) == set(metadata["arrays"]) == expected_keys
        for scene in SCENES:
            count = len(scene["params"]["means"])
            assert set(scene["params"]) == set(PARAM_KEYS)
            for key in PARAM_KEYS:
                shape = (count,) if key == "opacities" else (count, 4 if key == "quats" else 3)
                assert np.asarray(scene["params"][key]).shape == shape
            for key in ["image", *(f"grad_{p}" for p in PARAM_KEYS)]:
                name = f"{scene['name']}/{key}"
                array = expected[name]
                shape = (
                    (scene["height"], scene["width"], 3)
                    if key == "image"
                    else np.asarray(scene["params"][key.removeprefix("grad_")]).shape
                )
                assert array.shape == shape and array.dtype == np.float32 and np.isfinite(array).all()
                evidence = metadata["arrays"][name]
                assert len(evidence["repeat_max_abs_diffs"]) == 30
                observed = max(evidence["repeat_max_abs_diffs"])
                magnitude = float(np.max(np.abs(array)))
                floor = 8 * float(np.finfo(np.float32).eps) * max(1.0, magnitude)
                assert evidence["reference_max_abs"] == magnitude
                assert evidence["observed_max_abs_diff"] == observed
                assert evidence["roundoff_floor"] == floor
                assert evidence["atol"] == max(4 * observed, floor)
                assert evidence["rtol"] == 0


@pytest.mark.cuda
@pytest.mark.parametrize("scene", SCENES, ids=lambda scene: scene["name"])
def test_warp_golden(scene, cuda_environment):
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    # Smoke/behavior checks still execute on a different GPU/software profile.
    actual = capture(scene)
    if cuda_environment != metadata["environment"]:
        pytest.skip(f"behavior checks executed; golden environment differs: {cuda_environment}")
    with np.load(BASE / "expected.npz", allow_pickle=False) as expected:
        for key, value in actual.items():
            tolerance = metadata["arrays"][key]
            np.testing.assert_allclose(value, expected[key], atol=tolerance["atol"], rtol=0, err_msg=key)


@pytest.mark.cuda
@pytest.mark.xfail(strict=True, raises=IndexError, reason="Known defect: N=0 indexes offsets[-1]; not a normal contract")
def test_empty_point_cloud_should_return_background(cuda_environment):
    model, w2c, K, bg = make_inputs(SCENES[0])
    for key in PARAM_KEYS:
        model.params[key] = torch.nn.Parameter(model.params[key][:0].detach())
    output = WarpRenderer().render(model, w2c, K, 32, 32, bg)
    torch.testing.assert_close(output.image, bg.expand(32, 32, 3), atol=0, rtol=0)


def record_worker(destination):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for recording")
    runs = []
    for _ in range(10):
        arrays = {}
        for scene in SCENES:
            arrays.update(capture(scene))
        runs.append(arrays)
    np.savez(destination, **{key: np.stack([run[key] for run in runs]) for key in runs[0]})
    destination.with_suffix(".json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")


def record(overwrite):
    outputs = [BASE / "expected.npz", BASE / "metadata.json"]
    if any(path.exists() for path in outputs) and not overwrite:
        raise SystemExit("Baseline exists; recording requires --record --overwrite")
    # Sequential independent processes avoid introducing competing GPU workloads.
    with tempfile.TemporaryDirectory(prefix="warpgs-phase0-") as temporary:
        workers = []
        environments = []
        for index in range(3):
            destination = Path(temporary) / f"worker-{index}.npz"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(destination)], check=True, cwd=ROOT)
            with np.load(destination, allow_pickle=False) as data:
                workers.append({key: data[key].copy() for key in data.files})
            environments.append(json.loads(destination.with_suffix(".json").read_text(encoding="utf-8")))
        assert all(env == environments[0] for env in environments)
        references = {key: value[0] for key, value in workers[0].items()}
        evidence = {}
        for key, reference in references.items():
            diffs = [float(np.max(np.abs(value - reference))) for worker in workers for value in worker[key]]
            magnitude = float(np.max(np.abs(reference)))
            floor = 8 * float(np.finfo(np.float32).eps) * max(1.0, magnitude)
            evidence[key] = {
                "reference_max_abs": magnitude,
                "repeat_max_abs_diffs": diffs,
                "observed_max_abs_diff": max(diffs),
                "roundoff_floor": floor,
                "atol": max(4 * max(diffs), floor),
                "rtol": 0,
            }
        np.savez_compressed(outputs[0], **references)
        metadata = {
            "purpose": "existing-behavior regression; not mathematical correctness",
            "created_utc": datetime.now(UTC).isoformat(),
            "seed": SPEC["seed"],
            "processes": 3,
            "repeats_per_process": 10,
            "reference_selection": "first repeat of first process, without averaging",
            "environment": environments[0],
            "gsplat_version_record_only": importlib.metadata.version("gsplat"),
            "nvcc": command_output(["nvcc", "--version"]),
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "source_sha256": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted((ROOT / "src/warpgs").glob("*.py"))
            },
            "harness_sha256": sha256(Path(__file__)),
            "text_hash_normalization": "CRLF -> LF; binary NPZ hashed without normalization",
            "scenes_sha256": sha256(SCENES_PATH),
            "expected_sha256": sha256(outputs[0]),
            "tolerance_formula": FORMULA,
            "arrays": evidence,
        }
        outputs[1].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded {len(references)} arrays, 3 processes x 10 repeats, to {BASE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        record_worker(arguments.worker)
    else:
        record(arguments.overwrite)
