"""Test-only access to existing Warp operations; no copied rendering formulas."""

import math

import torch
import warp as wp

from warpgs.my_warp_rasterizer import (
    RasterConfigWP,
    backward_render_kernel,
    compute_cov3d_wp,
    compute_gaussian_screen_data_wp,
    make_warp_config,
    preprocess_kernel,
    render_kernel,
)


@wp.kernel
def observe_projection(
    means: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    view: wp.array[wp.mat44],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
    config: RasterConfigWP,
    covariance: wp.array[wp.mat33],
    mean: wp.array[wp.vec2],
    conic: wp.array[wp.vec3],
    diagonal: wp.array[wp.vec2],
    depth: wp.array[float],
):
    i = wp.tid()
    cov = compute_cov3d_wp(scales[i], quats[i])
    data = compute_gaussian_screen_data_wp(means[i], cov, view[0], fx, fy, cx, cy, width, height, config)
    covariance[i] = cov
    mean[i] = data.mu_pixel
    conic[i] = data.cov2d_inv
    diagonal[i] = data.cov2d_diag
    depth[i] = data.depth


def array(tensor, dtype, grad=False):
    return wp.from_torch(tensor, dtype=dtype, requires_grad=grad)


def cpu(value):
    return wp.to_torch(value).detach().cpu().double().clone()


def projection(means, scales, quats, camera, channel=None, *, config=None):
    view, K, height, width, _ = camera
    n = len(means)
    wp.init()
    stream = wp.stream_from_torch(torch.cuda.current_stream())
    with wp.ScopedStream(stream):
        inputs = [array(means, wp.vec3, True), array(scales, wp.vec3, True), array(quats, wp.quat, True)]
        outputs = {
            key: wp.zeros(n, dtype=dtype, device="cuda:0", requires_grad=True)
            for key, dtype in (
                ("cov3d", wp.mat33),
                ("mean", wp.vec2),
                ("conic", wp.vec3),
                ("diagonal", wp.vec2),
                ("depth", wp.float32),
            )
        }
        tape = wp.Tape()
        with tape:
            wp.launch(
                observe_projection,
                dim=n,
                inputs=[
                    *inputs,
                    array(view.reshape(1, 4, 4), wp.mat44),
                    float(K[0, 0]),
                    float(K[1, 1]),
                    float(K[0, 2]),
                    float(K[1, 2]),
                    float(width),
                    float(height),
                    make_warp_config(config),
                ],
                outputs=list(outputs.values()),
                device="cuda:0",
            )
        gradients = {}
        if channel:
            target = wp.to_torch(outputs[channel])
            seed = torch.linspace(0.1, 0.9, target.numel(), dtype=torch.float32).cuda().reshape_as(target)
            tape.backward(grads={outputs[channel]: array(seed, outputs[channel].dtype)})
            gradients = {key: cpu(value.grad) for key, value in zip(("means", "scales", "quats"), inputs, strict=True)}
        return {key: cpu(value) for key, value in outputs.items()}, gradients


def preprocess(means, scales, quats, camera, *, config=None):
    view, K, height, width, _ = camera
    n = len(means)
    stream = wp.stream_from_torch(torch.cuda.current_stream())
    with wp.ScopedStream(stream):
        outputs = {
            key: wp.zeros(n, dtype=dtype, device="cuda:0")
            for key, dtype in (
                ("conic", wp.vec3),
                ("mean", wp.vec2),
                ("depth", wp.float32),
                ("rect", wp.vec4i),
                ("touched", wp.int32),
            )
        }
        wp.launch(
            preprocess_kernel,
            dim=n,
            inputs=[
                array(means, wp.vec3),
                array(scales, wp.vec3),
                array(quats, wp.quat),
                array(view.reshape(1, 4, 4), wp.mat44),
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
                float(width),
                float(height),
                math.ceil(width / 16),
                math.ceil(height / 16),
                make_warp_config(config),
            ],
            outputs=list(outputs.values()),
            device="cuda:0",
        )
        return {key: cpu(value) for key, value in outputs.items()}


def composite(mean, conic, opacity, colors, height, width, background, grad_image=None, *, config=None):
    """Supply one known ordered list to all tiles; isolate compositor from binning."""
    wp.init()
    n, columns = len(mean), math.ceil(width / 16)
    ranges = torch.tensor([[0, n]] * (columns * math.ceil(height / 16)), dtype=torch.int32, device="cuda")
    ids = torch.arange(n, dtype=torch.int32, device="cuda")
    stream = wp.stream_from_torch(torch.cuda.current_stream())
    with wp.ScopedStream(stream):
        arrays = [
            array(value, dtype)
            for value, dtype in (
                (ranges, wp.vec2i),
                (ids, wp.int32),
                (conic, wp.vec3),
                (mean, wp.vec2),
                (colors, wp.vec3),
                (opacity, wp.float32),
            )
        ]
        image = wp.zeros((height, width), dtype=wp.vec3, device="cuda:0")
        final_t = wp.zeros((height, width), dtype=float, device="cuda:0")
        last = wp.zeros((height, width), dtype=int, device="cuda:0")
        bg = wp.vec3(*background.tolist())
        settings = make_warp_config(config)
        wp.launch(
            render_kernel,
            dim=(width, height),
            inputs=[*arrays, width, height, columns, bg, settings],
            outputs=[image, final_t, last],
            device="cuda:0",
        )
        gradients = {}
        if grad_image is not None:
            grad_arrays = [wp.zeros_like(arrays[i]) for i in (2, 3, 4, 5)]
            wp.launch(
                backward_render_kernel,
                dim=(width, height),
                inputs=[*arrays, final_t, last, array(grad_image, wp.vec3), width, height, columns, bg, settings],
                outputs=grad_arrays,
                device="cuda:0",
            )
            gradients = {
                key: cpu(value) for key, value in zip(("conic", "mean", "colors", "opacities"), grad_arrays, strict=True)
            }
        return {"image": cpu(image), "transmittance": cpu(final_t), "last": cpu(last)}, gradients
