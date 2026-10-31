# WarpGS research playground — repository instructions

## Purpose

This project is a custom differentiable 3D Gaussian Splatting renderer implemented with NVIDIA Warp and PyTorch. We are turning it into a **mathematical research playground**: independently verify rendering formulas, vary one mathematical assumption at a time, compare forward/backward results with references, and measure numerical and GPU behavior.

**Do not optimize for feature count, production polish, or current benchmark SOTA.** Keep the system small, inspectable, and experiment-friendly.

## Working agreement

1. Inspect the actual repository structure before making a plan. Core modules (`my_warp_rasterizer.py`, `renderers.py`, `model.py`, `train.py`) live in `src/warpgs/`; viewer variants live in `viewers/`. There is currently no `densification.py`: `train.py` directly integrates gsplat MCMC. Trust the real repository.
2. Read `REFACTOR_PLAN.md` before changing code. Work **one approved phase at a time**, and stop at that phase's boundary.
3. Make small, reviewable changes. No sweeping rewrite, premature plugin framework, viewer unification, or speculative primitive hierarchy.
4. Preserve existing renderer behavior by default. Any deliberate numerical change must be opt-in and documented with an old/new comparison. Preserve the public `WarpRenderer` and `RenderOutput` contract unless approval is given.
5. Establish baseline outputs/tests **before** extracting or editing mathematical kernels. Run regression tests after every meaningful change.
6. Separate three different questions: (a) does Warp agree with an independent implementation of the same formula? (b) does the formula approximate a mathematical reference? (c) what happens after model retraining? Do not conflate them.
7. GPU tests require compatible NVIDIA/CUDA/Warp/PyTorch. Add CPU-friendly analytic/reference tests where possible; mark GPU-dependent tests for conditional skipping and report if GPU execution is unavailable. Never claim unexecuted tests passed.
8. Gradient checks should avoid discontinuities from culling, sorting, thresholding, and alpha clipping. Test these boundaries separately as behavioral cases.
9. Do not change view scripts, dataset format, checkpoint/PLY compatibility, densification strategy, training hyperparameters, or add major dependencies unless explicitly part of an approved phase.
10. Record exact test commands and observed outcomes. Flag assumptions, failures, numerical tolerances, and environment limitations explicitly.
11. Do not publish/push, delete user data, or make commits without separate approval. Work on a dedicated Git branch and show a diff before requesting acceptance.

## Important implementation details to verify against code

- The rasterizer uses 16x16 tile binning, prefix scan, tile/depth sort, and alpha compositing, with a custom PyTorch autograd bridge and Warp kernels.
- Fixed decisions include Gaussian support extent, low-pass covariance regularization, alpha cutoff/cap, near-plane handling, and transmittance early exit. These must be distinguished from the ideal mathematical model.
- Screen mean and covariance projection both use Warp preprocessing. A manual compositing backward feeds screen-mean and inverse-covariance gradients into the preprocessing Tape. Camera matrices, intrinsics and background receive no gradients through the Warp bridge.
- `renderers.py` provides Warp and gsplat wrappers. gsplat is a comparison backend, not an exact mathematical ground truth.
- `src/warpgs/train.py` directly uses gsplat `MCMCStrategy`; there is no Default strategy adapter. Do not silently remove or rewrite this integration.
- A Monte Carlo distribution of projected 3D *samples* and a ray integral of 3D *density* are different mathematical objects. Do not use one as a ground truth for the other without explicitly defining the comparison.

## Definition of a good contribution

- A formula is documented with assumptions and units.
- A reference implementation is independent enough to detect mistakes in the optimized path.
- A minimal deterministic test establishes reproducibility.
- Numerical and performance differences have named measurement methods and explicit tolerances.
- A reader can reproduce an experiment from the command line and interpret its outputs.
