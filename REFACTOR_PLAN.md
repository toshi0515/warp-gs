# WarpGS Research Playground — phased refactoring plan

## Objective

Convert the existing differentiable NVIDIA Warp 3DGS codebase into a reliable playground for mathematical hypotheses. This is a controlled refactor, **not** a request to implement a new renderer, primitive, optimization method, or GPU acceleration scheme.

The implementation should support: (1) independent mathematical references; (2) forward/backward correctness checks; (3) explicit approximation settings; (4) reproducible small experiments; and eventually (5) swappable mathematical components without rewriting binning/sorting code.

## Current architecture (verify actual files before acting)

- `src/warpgs/my_warp_rasterizer.py`: 3D covariance, screen mean and Jacobian covariance projection in Warp, tile preprocessing, duplicate/sort, rasterization, manual compositing backward and preprocessing Tape, PyTorch `WarpRasterizeFunction`.
- `src/warpgs/renderers.py`: `WarpRenderer`, `GsplatRenderer`, `RenderOutput`.
- `src/warpgs/model.py`: differentiable Gaussian parameter container.
- `src/warpgs/train.py`: training loop, evaluation and direct gsplat `MCMCStrategy` integration. No `densification.py` or Default strategy adapter currently exists.
- `viewers/05_warp_tile_viewer.py` and other viewers: separate interactive rendering paths; not in initial scope.

Examples of implementation choices to expose and audit later (confirm actual values in source): 16px tile size, Gaussian extent around 3.33, low-pass covariance addition around 0.3, alpha cutoff around 1/255, cap around 0.99, transmittance threshold around 1e-4, near clipping, determinant clamp, Jacobian field-of-view clamp, depth encoding/sort behavior.

## Safety / scope

- First make tests, then refactor. Preserve defaults and observable behavior unless a separately approved experiment changes them.
- Do not reorganize every Python module in one pass. Existing viewer CUDA/OpenGL interop is out of scope.
- Do not add a generic `Primitive` inheritance hierarchy, new neural models, or full plugin architecture.
- Do not change model/checkpoint/PLY schema, train defaults, or gsplat integrations without explicit approval.
- Evaluate CUDA availability before running Warp; pure-math tests should not require a GPU.
- Use a dedicated Git branch; never auto-push or auto-commit. No environment installations or dependency upgrades unless approved.

## Phase 0 — Inventory and baseline (first implementation phase)

1. Read actual repository, identify runnable test commands and local GPU availability, and identify exact forward/backward paths.
2. Record current behavior of a few deterministic synthetic scenes: one centered Gaussian, one anisotropic rotated Gaussian, two overlapping Gaussians, empty/fully culled Gaussian(s), and if supported a small image with non-black background.
3. Save baseline metadata (seed, dimensions, camera intrinsics, model parameters, GPU/software versions) and reference outputs where practical. Choose deterministic comparison tolerances based on measurements, not assumptions.
4. Write a concise architecture note identifying which parts of the computation are mathematical choices versus implementation details. Document which inputs have gradients today.
5. Report test results, GPU limitations, and a small Phase 1 proposal. **Stop after Phase 0.**

Acceptance: rerunnable baseline checks exist; observed results are recorded; no intentional algorithm change has been made; no claim of GPU verification without actual GPU execution.

Phase 0 golden tests record existing behavior, not mathematical correctness. Runtime rendering/backward/training code stays unchanged. Store per-image and per-parameter gradient tolerances with measured repeatability evidence. Baselines update only through an explicit recording command; ordinary tests never update them. Track empty-input `IndexError` as a known defect, not a normal behavior contract. Warp CUDA tests must remain independent of gsplat initialization. See `docs/research_baseline.md` for commands and results.

## Phase 1 — Reference mathematics and numerical tests

1. Implement a clean CPU-friendly PyTorch/NumPy reference for 3D covariance and first-order Jacobian projection. Keep pure mathematical reference and actual rasterization settings separate.
2. Add unit tests that compare PyTorch formula with Warp projection under matched clamping/regularization settings. Where independent Warp output access requires a diagnostic wrapper, make the narrowest change.
3. Add a minimal forward renderer comparison (single and overlapping Gaussians) against a simple independent reference for the *same alpha-compositing formula and ordering*.
4. Validate analytic/Warp gradients for mean, scale, quaternion, opacity, and color using finite differences on small smooth test cases. Check quaternion convention (wxyz in model vs xyzw inside Warp) and avoid threshold/sorting transitions in these tests.
5. Declare test tolerances explicitly. Separate expected numerical discrepancies from bugs. Mark CUDA tests for conditional skipping rather than fabricating results.

Acceptance: reproducible tests, evidence of actual execution, explicit list of tested gradients and known coverage gaps.

## Phase 2 — Externalize experimental settings

1. Introduce a small immutable `RasterConfig` or equivalent. Start with Gaussian extent, covariance low-pass, Jacobian clamp, alpha cutoff, cap, transmittance threshold, and near-plane threshold. Only implement parameters that can be safely threaded through current Warp functions.
2. Preserve existing values as defaults. Be mindful of Warp compile-time specialization versus runtime parameters.
3. Make baseline tests compare original-equivalent settings after the refactor.
4. Do not yet create generalized projection/kernel registries or rewrite binning.

Acceptance: changing a parameter requires changing a configuration value rather than editing kernel source; default configuration reproduces recorded baselines within justified tolerances.

## Phase 3 — First mathematical experiment: perspective-projection error

Research question: When does first-order Gaussian perspective projection deviate materially from the nonlinear push-forward of a 3D Gaussian distribution?

1. State mathematical assumptions before coding: camera coordinate convention, perspective projection, near-plane truncation/conditioning, and whether a reported quantity is a projected point-distribution statistic or a ray-integrated density image.
2. Compare: (a) Jacobian first-order mean/covariance; (b) second-order local mean estimate where valid; (c) Monte Carlo nonlinear projection as a sample-based numerical reference with confidence intervals/uncertainty.
3. Sweep depth, depth variance, anisotropy, orientation, and xz/yz covariance. Visualize errors and save a table plus settings and seed.
4. Important: naive unconditioned ratios X/Z of a Gaussian crossing Z=0 can lack finite moments. Do not present naive Monte Carlo moments as mathematical ground truth without explicit positive-depth truncation.
5. Keep the experiment runnable without training a full scene.

Acceptance: a reproducible study comparing analytic prediction and sampled estimates; explicit handling of edge cases, numerical uncertainty, and limitations.

## Phase 4 — Optional mathematical component separation

Only after Phases 0–3 demonstrate the need: progressively separate projection, support/culling, kernel evaluation, and compositing from binning/sorting orchestration, preserving the existing PyTorch/Warp autograd contract. Do not automatically merge viewer code.

## Future research modules — not part of initial refactor

- Sampling/antialiasing: center sampling versus pixel-area integration; mip filter behavior.
- Primitive approximation: Gaussian, compact polynomial, Wendland, superellipse; fixed-N comparisons before densification.
- Volume rendering: screen-space alpha versus integrated 3D Gaussian density with matched calibration.
- Optimization: gradient conditioning and training convergence after correctness has been verified.

## Workflow for Codex

For each phase:
1. Identify exact source files and tests to touch; report assumptions.
2. Show an implementation plan with acceptance checks.
3. Change only the approved scope.
4. Run appropriate tests and report observed output verbatim or in a concise summary.
5. Show `git diff --stat`, relevant diff summary, remaining limitations, and next phase recommendation.
6. **Stop and wait for user approval before continuing**.
