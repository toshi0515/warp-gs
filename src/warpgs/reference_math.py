"""Small independent PyTorch references for research, not a training backend.

Use CPU float64 for validation. No Warp/gsplat imports or production constants.
Coordinates: world units, positive camera z, pixels with centers at (x+.5,y+.5).
Quaternions are wxyz; scales are standard deviations, not log-scales.
Legacy constants below document the current renderer, not an ideal density model.
"""

from dataclasses import dataclass

import torch


@dataclass
class Projection:
    mean: torch.Tensor  # N,2 pixels
    jacobian: torch.Tensor  # N,2,3 pixels/world-unit, camera coordinates
    covariance: torch.Tensor  # N,2,2 pixel^2
    depth: torch.Tensor  # N world units
    inverse: torch.Tensor | None = None  # N,2,2 pixel^-2


@dataclass
class ReferenceImage:
    image: torch.Tensor
    transmittance: torch.Tensor
    branches: dict[str, torch.Tensor]


def covariance_3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """R diag(s^2) R^T, independently constructed by rotating basis vectors.

    Requires positive scales and nonzero quaternions. Extremely small quaternions
    subject to the production norm floor are outside this pure-math function.
    """
    norm = torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    if not bool(torch.all(scales > 0)) or not bool(torch.all(norm > 0)):
        raise ValueError("Positive scales and nonzero wxyz quaternions required")
    q = quats / norm
    w, v = q[..., :1], q[..., 1:]
    basis = torch.eye(3, dtype=scales.dtype, device=scales.device).expand(*scales.shape[:-1], 3, 3)
    # Rotated basis vectors are rows here; transpose to obtain R's columns.
    vector = v.unsqueeze(-2).expand_as(basis)
    cross = torch.linalg.cross(vector, basis, dim=-1)
    rotated = basis + 2 * (w.unsqueeze(-1) * cross + torch.linalg.cross(vector, cross, dim=-1))
    rotation = rotated.transpose(-1, -2)
    return rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)


def _project(means, cov3d, w2c, K, limits=None):
    camera = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    if not bool(torch.all(z > 0)):
        raise ValueError("Projection requires strictly positive camera depth")
    normalized = camera[:, :2] / z[:, None]
    focal = torch.stack((K[0, 0], K[1, 1]))
    center = K[:2, 2]
    mean = normalized * focal + center
    slope = normalized if limits is None else torch.maximum(torch.minimum(normalized, limits), -limits)
    left = torch.diag_embed(focal.expand(len(means), 2) / z[:, None])
    jacobian = torch.cat((left, (-focal * slope / z[:, None]).unsqueeze(-1)), dim=-1)
    camera_covariance = w2c[:3, :3] @ cov3d @ w2c[:3, :3].T
    covariance = jacobian @ camera_covariance @ jacobian.transpose(-1, -2)
    return Projection(mean, jacobian, covariance, z)


def project_pinhole(means, cov3d, w2c, K) -> Projection:
    """First-order pinhole push-forward; no regularization or FOV clamp.

    K has zero skew, and w2c is a rigid world-to-camera transform.
    This does not claim to give exact moments of nonlinear Gaussian projection.
    """
    return _project(means, cov3d, w2c, K)


def project_legacy(
    means, cov3d, w2c, K, height: int, width: int, *, covariance_lowpass=0.3, jacobian_clamp_factor=1.3
) -> Projection:
    """Current FOV clamp (1.3), low-pass (0.3 pixel^2), determinant floor (1e-6).

    Positive depth is required; near-plane selection is done by the renderer.
    The floor branch is a scaled adjugate, not the mathematical matrix inverse.
    """
    limits = means.new_tensor([width, height]) * (jacobian_clamp_factor * 0.5) / torch.stack((K[0, 0], K[1, 1]))
    result = _project(means, cov3d, w2c, K, limits)
    cov = result.covariance + torch.eye(2, dtype=means.dtype, device=means.device) * covariance_lowpass
    det = torch.linalg.det(cov)
    inverse = torch.linalg.inv(cov)
    result.covariance = cov
    result.inverse = inverse * (det / det.clamp_min(1e-6))[:, None, None]
    return result


def conic_entries(inverse: torch.Tensor) -> torch.Tensor:
    return torch.stack((inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]), dim=-1)


def composite_reference(
    mean,
    conic,
    opacities,
    colors,
    height,
    width,
    background,
    *,
    gaussian_extent=3.33,
    alpha_cutoff=1 / 255,
    alpha_cap=0.99,
    transmittance_threshold=1e-4,
) -> ReferenceImage:
    """Direct, front-to-back screen-space compositing in the supplied order.

    Inputs have already been projected, culled and sorted. No tile infrastructure.
    This helper permits independent derivatives with respect to screen quantities.
    """
    y, x = torch.meshgrid(
        torch.arange(height, dtype=mean.dtype, device=mean.device) + 0.5,
        torch.arange(width, dtype=mean.dtype, device=mean.device) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y), dim=-1)
    image = mean.new_zeros(height, width, 3)
    transmittance = mean.new_ones(height, width)
    trace = {key: [] for key in ("support", "cutoff", "cap", "alive", "contributes")}
    for i in range(len(mean)):
        inverse = torch.stack((conic[i, [0, 1]], conic[i, [1, 2]]))
        delta = pixels - mean[i]
        power = torch.einsum("...i,ij,...j->...", delta, inverse, delta)
        raw_alpha = opacities[i] * torch.exp(-power / 2)
        alpha = raw_alpha.clamp_max(alpha_cap)
        support = (power >= 0) & (power <= gaussian_extent**2)
        cutoff = alpha >= alpha_cutoff
        alive = transmittance >= transmittance_threshold
        contributes = support & cutoff & alive
        effective = torch.where(contributes, alpha, 0)
        image = image + (transmittance * effective)[..., None] * colors[i]
        transmittance = transmittance * (1 - effective)
        for key, value in zip(trace, (support, cutoff, raw_alpha >= alpha_cap, alive, contributes), strict=True):
            trace[key].append(value.detach())
    branches = {
        key: torch.stack(values) if values else torch.empty((0, height, width), dtype=torch.bool, device=mean.device)
        for key, values in trace.items()
    }
    return ReferenceImage(image + transmittance[..., None] * background, transmittance, branches)


def render_legacy_reference(
    means,
    quats,
    scales,
    opacities,
    colors,
    w2c,
    K,
    height,
    width,
    background,
    *,
    gaussian_extent=3.33,
    covariance_lowpass=0.3,
    jacobian_clamp_factor=1.3,
    alpha_cutoff=1 / 255,
    alpha_cap=0.99,
    transmittance_threshold=1e-4,
    near_plane=0.1,
) -> ReferenceImage:
    """Small-N current-formula reference; inputs are physical (exp/sigmoid applied).

    Visible depth keys must be distinct and fit uint32. Equal-key ordering is not
    specified by this reference. The ideal projection function remains separate.
    """
    depth = (means @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    visible = depth > near_plane
    ids = torch.where(visible)[0]
    keys = (depth[ids].detach().float() * 10000).to(torch.int64)
    if bool(torch.any(keys >= 2**32)) or len(torch.unique(keys)) != len(keys):
        raise ValueError("Reference comparison requires distinct in-range depth keys")
    ids = ids[torch.argsort(keys)]
    covariance = covariance_3d(scales[ids], quats[ids])
    projected = project_legacy(
        means[ids],
        covariance,
        w2c,
        K,
        height,
        width,
        covariance_lowpass=covariance_lowpass,
        jacobian_clamp_factor=jacobian_clamp_factor,
    )
    result = composite_reference(
        projected.mean,
        conic_entries(projected.inverse),
        opacities[ids],
        colors[ids],
        height,
        width,
        background,
        gaussian_extent=gaussian_extent,
        alpha_cutoff=alpha_cutoff,
        alpha_cap=alpha_cap,
        transmittance_threshold=transmittance_threshold,
    )
    normalized = (means @ w2c[:3, :3].T + w2c[:3, 3])[ids, :2] / projected.depth[:, None]
    limits = means.new_tensor([width, height]) * (jacobian_clamp_factor * 0.5) / torch.stack((K[0, 0], K[1, 1]))
    result.branches.update(visible=visible.detach(), order=ids.detach(), clamp=(normalized.abs() > limits).detach())
    return result
