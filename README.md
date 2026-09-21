# warpgs

Differentiable 3D Gaussian Splatting with interchangeable NVIDIA Warp and
gsplat renderers. Both backends consume the same `GaussianModel`, optimizer
layout, and checkpoint format.

## Backends and densification

| Renderer | `default` | `mcmc` |
| --- | --- | --- |
| `warp` | gsplat `DefaultStrategy` (`absgrad=False`) | gsplat `MCMCStrategy` |
| `gsplat` | gsplat `DefaultStrategy` | gsplat `MCMCStrategy` |

All Gaussian addition, removal, relocation, and optimizer-state updates are
owned by gsplat. The project contains no separate densification implementation.

```powershell
uv sync

# Only needed when the gsplat extension has not yet been built on Windows.
.\scripts\setup_gsplat_windows.ps1

.\.venv\Scripts\python.exe train\warp_train.py --renderer warp --densification mcmc
.\.venv\Scripts\python.exe train\warp_train.py --renderer warp --densification default
.\.venv\Scripts\python.exe train\warp_train.py --renderer gsplat --densification default
.\.venv\Scripts\python.exe train\warp_train.py --renderer gsplat --densification mcmc
```

Use `--init_checkpoint PATH` to train or render from the same initialization
with either backend. New checkpoints use the flat gsplat-compatible keys
`means`, `quats`, `scales`, `opacities`, and `colors`.

The gsplat CUDA extension is checked when the densification strategy is
created. Failure is explicit and never falls back to another implementation.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check src train tests
.\.venv\Scripts\python.exe -m pyrefly check src train tests
.\.venv\Scripts\python.exe -m pytest -q
```
