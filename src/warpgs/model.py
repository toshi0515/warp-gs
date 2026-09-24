import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, Optimizer

PARAM_KEYS = ("means", "quats", "scales", "opacities", "colors")
OptimizerDict = dict[str, Optimizer]


class GaussianModel(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
    ) -> None:
        super().__init__()
        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "colors": nn.Parameter(colors),
            }
        )

    @property
    def means(self) -> nn.Parameter:
        return self.params["means"]

    @property
    def quats(self) -> nn.Parameter:
        return self.params["quats"]

    @property
    def scales(self) -> torch.Tensor:
        return self.params["scales"].exp()

    @property
    def opacities(self) -> torch.Tensor:
        return self.params["opacities"].sigmoid()

    @property
    def colors(self) -> torch.Tensor:
        return self.params["colors"].sigmoid()

    def num_gaussians(self) -> int:
        return len(self.params["means"])

    @torch.no_grad()
    def normalize_quats(self) -> None:
        self.params["quats"].copy_(F.normalize(self.params["quats"], dim=-1))

    @classmethod
    def from_random(
        cls,
        num_gaussians: int = 5_000,
        scene_radius: float = 1.5,
        init_scale: float = 0.05,
        device: str | torch.device = "cuda",
    ) -> "GaussianModel":
        means = (torch.rand(num_gaussians, 3, device=device) - 0.5) * 2 * scene_radius
        quats = F.normalize(torch.randn(num_gaussians, 4, device=device), dim=-1)
        scales = torch.full((num_gaussians, 3), math.log(init_scale), device=device)
        opacities = torch.full((num_gaussians,), -2.2, device=device)
        colors = torch.zeros(num_gaussians, 3, device=device)
        return cls(means, quats, scales, opacities, colors)

    def save(self, path: str | Path) -> None:
        torch.save(self.params.state_dict(), path)
        print(f"[model] saved {self.num_gaussians()} Gaussians -> {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cuda") -> "GaussianModel":
        params = torch.load(path, map_location=device, weights_only=True)
        model = cls(**params).to(device)
        print(f"[model] loaded {model.num_gaussians()} Gaussians <- {path}")
        return model


def create_optimizers(model: GaussianModel) -> OptimizerDict:
    learning_rates = {
        "means": 1.6e-4,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    return {
        name: Adam([{"params": [model.params[name]], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in learning_rates.items()
    }
