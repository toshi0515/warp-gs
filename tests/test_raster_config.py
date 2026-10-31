import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from warpgs.raster_config import RasterConfig


def test_defaults_immutable_and_cpu_only():
    config = RasterConfig()
    assert asdict(config) == dict(
        gaussian_extent=3.33,
        covariance_lowpass=0.3,
        jacobian_clamp_factor=1.3,
        alpha_cutoff=1 / 255,
        alpha_cap=0.99,
        transmittance_threshold=1e-4,
        near_plane=0.1,
    )
    with pytest.raises(FrozenInstanceError):
        config.alpha_cap = 0.8
    assert replace(config, alpha_cap=0.8).alpha_cap == 0.8
    assert config.effective_values()["alpha_cap"] < 1
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from warpgs.raster_config import RasterConfig; RasterConfig(); "
            "assert not {'torch','warp','gsplat'}.intersection(sys.modules)",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("gaussian_extent", 0),
        ("gaussian_extent", -1),
        ("gaussian_extent", 1e-50),
        ("gaussian_extent", 1e40),
        ("covariance_lowpass", -0.1),
        ("jacobian_clamp_factor", 0),
        ("alpha_cutoff", -0.1),
        ("alpha_cutoff", 1),
        ("alpha_cap", 1),
        ("alpha_cap", 0),
        ("alpha_cap", 0.999999999),
        ("transmittance_threshold", -0.1),
        ("transmittance_threshold", 1.1),
        ("near_plane", -0.1),
        *[(field, value) for field in asdict(RasterConfig()) for value in (float("nan"), float("inf"))],
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        RasterConfig(**{field: value})


@pytest.mark.parametrize("value", [True, "0.3", None])
def test_config_rejects_non_numeric_scalars(value):
    with pytest.raises(TypeError):
        RasterConfig(covariance_lowpass=value)


def test_zero_controls_and_closed_threshold_endpoints():
    RasterConfig(covariance_lowpass=0, alpha_cutoff=0, transmittance_threshold=0, near_plane=0)
    RasterConfig(alpha_cutoff=0.8, alpha_cap=0.8, transmittance_threshold=1)
