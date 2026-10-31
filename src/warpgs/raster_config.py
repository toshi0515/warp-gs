"""Immutable, CPU-only experimental settings for the Warp rasterizer."""

import math
import struct
from dataclasses import asdict, dataclass
from numbers import Real


@dataclass(frozen=True)
class RasterConfig:
    gaussian_extent: float = 3.33
    covariance_lowpass: float = 0.3  # pixel^2
    jacobian_clamp_factor: float = 1.3
    alpha_cutoff: float = 1.0 / 255.0
    alpha_cap: float = 0.99
    transmittance_threshold: float = 1e-4
    near_plane: float = 0.1  # world units, camera z

    def __post_init__(self) -> None:
        requested = asdict(self)
        for name, value in requested.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
        self._validate(requested)
        self._validate(self.effective_values())

    @staticmethod
    def _validate(values: dict[str, float]) -> None:
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Raster settings must be finite, including after float32 conversion")
        if values["gaussian_extent"] <= 0 or values["jacobian_clamp_factor"] <= 0:
            raise ValueError("gaussian_extent and jacobian_clamp_factor must be positive")
        if values["covariance_lowpass"] < 0 or values["near_plane"] < 0:
            raise ValueError("covariance_lowpass and near_plane must be nonnegative")
        if not 0 <= values["alpha_cutoff"] <= values["alpha_cap"] < 1 or values["alpha_cap"] <= 0:
            raise ValueError("Require 0 <= alpha_cutoff <= alpha_cap < 1 and alpha_cap > 0")
        if not 0 <= values["transmittance_threshold"] <= 1:
            raise ValueError("transmittance_threshold must be in [0, 1]")

    def effective_values(self) -> dict[str, float]:
        """Return float32 values actually passed to Warp, for experiment logs."""
        try:
            return {name: struct.unpack("<f", struct.pack("<f", value))[0] for name, value in asdict(self).items()}
        except (OverflowError, struct.error) as error:
            raise ValueError("Raster settings must be representable as finite float32") from error
