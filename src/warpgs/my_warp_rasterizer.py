import math

import torch
import warp as wp

mat23f = wp.types.matrix(shape=(2, 3), dtype=wp.float32)

TILE_SIZE = 16
GAUSSIAN_EXTENT = 3.33
MAX_ALPHA = 0.99
ALPHA_THRESHOLD = 1.0 / 255.0
TRANSMITTANCE_THRESHOLD = 1e-4
NEAR_PLANE = 0.1


@wp.func
def compute_cov3d_wp(scale: wp.vec3, quat: wp.quat) -> wp.mat33:
    S = wp.mat33(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2])
    quat_norm_sq = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
    inv_quat_norm = 1.0 / wp.sqrt(wp.max(quat_norm_sq, 1e-12))

    # warp.quatは(x, y, z, w)の順なので並び替える
    quat_normalized = wp.quat(
        quat[1] * inv_quat_norm,
        quat[2] * inv_quat_norm,
        quat[3] * inv_quat_norm,
        quat[0] * inv_quat_norm,
    )
    R = wp.quat_to_matrix(quat_normalized)
    RS = R @ S
    cov3d = RS @ wp.transpose(RS)
    return cov3d


@wp.struct
class GaussianScreenData:
    cov2d_diag: wp.vec2  # AABB半径計算用の対角成分 (a, c)
    mu_pixel: wp.vec2
    depth: float
    cov2d_inv: wp.vec3


@wp.func
def compute_gaussian_screen_data_wp(
    mean: wp.vec3,
    cov3d: wp.mat33,
    view_matrix: wp.mat44,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    screen_w: float,
    screen_h: float,
) -> GaussianScreenData:
    mean_h = wp.vec4(mean[0], mean[1], mean[2], 1.0)
    t_cam_h = view_matrix @ mean_h

    tx = t_cam_h[0]
    ty = t_cam_h[1]
    tz = t_cam_h[2]

    # tzの値によってJが大きくなりすぎるのを防ぐ
    lim_x = 1.3 * 0.5 * screen_w / fx
    lim_y = 1.3 * 0.5 * screen_h / fy
    tx_cov = tz * wp.clamp(tx / tz, -lim_x, lim_x)
    ty_cov = tz * wp.clamp(ty / tz, -lim_y, lim_y)

    J = mat23f(
        fx / tz,
        0.0,
        -(fx * tx_cov) / (tz * tz),
        0.0,
        fy / tz,
        -(fy * ty_cov) / (tz * tz),
    )

    W_rot = wp.mat33(
        view_matrix[0, 0],
        view_matrix[0, 1],
        view_matrix[0, 2],
        view_matrix[1, 0],
        view_matrix[1, 1],
        view_matrix[1, 2],
        view_matrix[2, 0],
        view_matrix[2, 1],
        view_matrix[2, 2],
    )

    mu_pixel = wp.vec2(fx * tx / tz + cx, fy * ty / tz + cy)

    cov_cam = W_rot @ cov3d @ wp.transpose(W_rot)
    cov2d = J @ cov_cam @ wp.transpose(J)
    cov2d[0, 0] += 0.3
    cov2d[1, 1] += 0.3

    a = cov2d[0, 0]
    b = cov2d[0, 1]
    c = cov2d[1, 1]

    det = wp.max((a * c) - (b * b), 1e-6)
    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det

    result = GaussianScreenData()
    result.cov2d_diag = wp.vec2(a, c)
    result.mu_pixel = mu_pixel
    result.depth = tz
    result.cov2d_inv = wp.vec3(inv_a, inv_b, inv_c)
    return result


@wp.kernel
def preprocess_kernel(
    means: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    mean_pixel: wp.array[wp.vec2],
    view_matrix_arr: wp.array[wp.mat44],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    screen_w: float,
    screen_h: float,
    tiles_per_row: int,
    tiles_per_col: int,
    # 出力 (すべて元のガウシアンインデックス tid で書き込む. サイズ N)
    out_cov2d_inv: wp.array[wp.vec3],
    out_depth: wp.array[wp.float32],
    out_tile_rect: wp.array[wp.vec4i],  # (min_x, min_y, max_x, max_y)
    out_tiles_touched: wp.array[wp.int32],
    out_radii: wp.array[wp.vec2],
):
    tid = wp.tid()

    # 可視外のデフォルト値, 早期returnする全パスでこれが有効になる
    out_tiles_touched[tid] = 0
    out_radii[tid] = wp.vec2(0.0, 0.0)

    mean = means[tid]
    mean_h = wp.vec4(mean[0], mean[1], mean[2], 1.0)
    view_matrix = view_matrix_arr[0]
    t_cam_h = view_matrix @ mean_h
    tz = t_cam_h[2]

    if tz <= NEAR_PLANE:
        return

    scale = scales[tid]
    quat = quats[tid]
    cov3d = compute_cov3d_wp(scale, quat)
    data = compute_gaussian_screen_data_wp(mean, cov3d, view_matrix, fx, fy, cx, cy, screen_w, screen_h)

    #
    radius_x = GAUSSIAN_EXTENT * wp.sqrt(wp.max(data.cov2d_diag[0], 0.0))
    radius_y = GAUSSIAN_EXTENT * wp.sqrt(wp.max(data.cov2d_diag[1], 0.0))
    mu = mean_pixel[tid]

    # 画面外に中心を持つ大きなGaussianも画像と交差する限り残す
    if mu[0] + radius_x <= 0.0 or mu[0] - radius_x >= screen_w:
        return
    if mu[1] + radius_y <= 0.0 or mu[1] - radius_y >= screen_h:
        return

    # タイル範囲は[min, max)で計算し既存の保存形式だけinclusive maxにする
    min_x = wp.max(wp.int32(wp.floor((mu[0] - radius_x) / float(TILE_SIZE))), 0)
    min_y = wp.max(wp.int32(wp.floor((mu[1] - radius_y) / float(TILE_SIZE))), 0)
    max_x_exclusive = wp.min(wp.int32(wp.ceil((mu[0] + radius_x) / float(TILE_SIZE))), tiles_per_row)
    max_y_exclusive = wp.min(wp.int32(wp.ceil((mu[1] + radius_y) / float(TILE_SIZE))), tiles_per_col)

    if max_x_exclusive <= min_x or max_y_exclusive <= min_y:
        return

    out_cov2d_inv[tid] = data.cov2d_inv
    out_depth[tid] = tz
    out_tile_rect[tid] = wp.vec4i(min_x, min_y, max_x_exclusive - 1, max_y_exclusive - 1)
    out_tiles_touched[tid] = (max_x_exclusive - min_x) * (max_y_exclusive - min_y)
    out_radii[tid] = wp.vec2(radius_x, radius_y)


@wp.kernel
def duplicate_kernel(
    tile_rect: wp.array[wp.vec4i],
    tiles_touched: wp.array[wp.int32],
    depth: wp.array[wp.float32],
    offsets: wp.array[wp.int32],  # array_scanで求めた exclusive prefix sum
    tiles_per_row: int,
    pair_keys: wp.array[wp.uint64],
    pair_values: wp.array[wp.int32],
):
    # offsetsで確定した書き込み位置に書き込む
    tid = wp.tid()

    count = tiles_touched[tid]
    if count == 0:
        return

    rect = tile_rect[tid]
    min_x = rect[0]
    min_y = rect[1]
    max_x = rect[2]
    max_y = rect[3]

    # depthを下位32bitに使えるよう正の整数化
    depth_int = wp.uint32(depth[tid] * 10000.0)

    write_idx = offsets[tid]
    for tiley in range(min_y, max_y + 1):
        for tilex in range(min_x, max_x + 1):
            tile_id = tiley * tiles_per_row + tilex
            key = (wp.uint64(tile_id) << wp.uint64(32)) | wp.uint64(depth_int)
            pair_keys[write_idx] = key
            pair_values[write_idx] = tid
            write_idx = write_idx + 1


@wp.kernel
def compute_tile_range_kernel(
    pair_keys: wp.array[wp.uint64],
    total_pairs: int,
    tile_ranges: wp.array[wp.vec2i],
):
    tid = wp.tid()
    if tid >= total_pairs:
        return

    curr_key = pair_keys[tid]
    curr_tile = wp.int32(curr_key >> wp.uint64(32))

    # tile_idが変わるタイミングでtile_rangesに開始位置と終了位置を記録する
    if tid == 0:
        tile_ranges[curr_tile][0] = 0
    else:
        prev_key = pair_keys[tid - 1]
        prev_tile = wp.int32(prev_key >> wp.uint64(32))
        if curr_tile != prev_tile:
            tile_ranges[curr_tile][0] = tid
            tile_ranges[prev_tile][1] = tid

    if tid == total_pairs - 1:
        tile_ranges[curr_tile][1] = total_pairs


@wp.kernel
def render_kernel(
    tile_ranges: wp.array[wp.vec2i],
    pair_values: wp.array[wp.int32],
    cov2d_inv: wp.array[wp.vec3],
    mu_pixel: wp.array[wp.vec2],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    screen_w: int,
    screen_h: int,
    tiles_per_row: int,
    bg_color: wp.vec3,
    image: wp.array2d[wp.vec3],  # 最終的なピクセルの色
    final_T: wp.array2d[wp.float32],  # 最終的な透過率
    last_contributor: wp.array2d[wp.int32],  # 最後に寄与したガウシアンのインデックス
):

    px, py = wp.tid()

    if px >= screen_w or py >= screen_h:
        return

    tile_x = px // TILE_SIZE
    tile_y = py // TILE_SIZE
    tile_id = tile_y * tiles_per_row + tile_x

    tr = tile_ranges[tile_id]
    start = tr[0]
    end = tr[1]

    px_center = float(px) + 0.5
    py_center = float(py) + 0.5

    T = float(1.0)  # noqa: UP018
    accum = wp.vec3(0.0, 0.0, 0.0)
    last = wp.int32(-1)

    for i in range(start, end):
        gid = pair_values[i]

        mu = mu_pixel[gid]
        dx = px_center - mu[0]
        dy = py_center - mu[1]

        conic = cov2d_inv[gid]
        power = dx * dx * conic[0] + 2.0 * dx * dy * conic[1] + dy * dy * conic[2]

        if power < 0.0 or power > GAUSSIAN_EXTENT * GAUSSIAN_EXTENT:
            continue

        alpha = wp.min(MAX_ALPHA, opacities[gid] * wp.exp(-0.5 * power))
        if alpha < ALPHA_THRESHOLD:
            continue

        accum = accum + colors[gid] * alpha * T
        T = T * (1.0 - alpha)
        last = i

        if T < TRANSMITTANCE_THRESHOLD:
            break

    final_color = accum + T * bg_color

    image[py, px] = final_color

    final_T[py, px] = T

    last_contributor[py, px] = last


@wp.kernel
def backward_render_kernel(
    tile_ranges: wp.array[wp.vec2i],
    pair_values: wp.array[wp.int32],
    cov2d_inv: wp.array[wp.vec3],
    mean_pixel: wp.array[wp.vec2],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    final_T: wp.array2d[wp.float32],
    last_contributor: wp.array2d[wp.int32],
    grad_image: wp.array2d[wp.vec3],
    screen_w: int,
    screen_h: int,
    tiles_per_row: int,
    bg_color: wp.vec3,
    grad_cov2d_inv: wp.array[wp.vec3],
    grad_mean_pixel: wp.array[wp.vec2],
    grad_colors: wp.array[wp.vec3],
    grad_opacities: wp.array[wp.float32],
):

    px, py = wp.tid()
    if px >= screen_w or py >= screen_h:
        return

    tile_id = (py // TILE_SIZE) * tiles_per_row + px // TILE_SIZE
    tile_range = tile_ranges[tile_id]
    last = last_contributor[py, px]
    if last < tile_range[0]:
        return

    pixel_x = float(px) + 0.5
    pixel_y = float(py) + 0.5
    grad = grad_image[py, px]
    behind_color = bg_color
    suffix_T = float(1.0)  # noqa: UP018 - must be dynamic in Warp loops
    final_t = final_T[py, px]

    for index in range(last, tile_range[0] - 1, -1):
        gid = pair_values[index]
        dx = pixel_x - mean_pixel[gid][0]
        dy = pixel_y - mean_pixel[gid][1]
        conic = cov2d_inv[gid]
        power = dx * dx * conic[0] + 2.0 * dx * dy * conic[1] + dy * dy * conic[2]

        if power >= 0.0 and power <= GAUSSIAN_EXTENT * GAUSSIAN_EXTENT:
            gaussian = wp.exp(-0.5 * power)
            raw_alpha = opacities[gid] * gaussian
            alpha = wp.min(MAX_ALPHA, raw_alpha)
            if alpha >= ALPHA_THRESHOLD:
                one_minus_alpha = 1.0 - alpha
                transmittance_before = final_t / (one_minus_alpha * suffix_T)
                color = colors[gid]

                grad_colors[gid] += grad * (alpha * transmittance_before)
                grad_alpha = transmittance_before * wp.dot(grad, color - behind_color)

                if raw_alpha < MAX_ALPHA:
                    grad_opacities[gid] += grad_alpha * gaussian
                    grad_power = -0.5 * grad_alpha * raw_alpha
                    grad_conic = wp.vec3(dx * dx, 2.0 * dx * dy, dy * dy) * grad_power
                    grad_dx = (2.0 * dx * conic[0] + 2.0 * dy * conic[1]) * grad_power
                    grad_dy = (2.0 * dx * conic[1] + 2.0 * dy * conic[2]) * grad_power
                    grad_cov2d_inv[gid] += grad_conic
                    grad_mean_pixel[gid] += wp.vec2(-grad_dx, -grad_dy)

                behind_color = color * alpha + behind_color * one_minus_alpha
                suffix_T = suffix_T * one_minus_alpha


def project_means_to_screen(means: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """3D中心を画像座標へ投影し、DefaultStrategyが参照できるTensorとして返す"""
    ones = torch.ones((means.shape[0], 1), dtype=means.dtype, device=means.device)
    means_camera = torch.cat((means, ones), dim=-1) @ w2c.T
    z = means_camera[:, 2]
    safe_z = torch.where(z.abs() > 1e-8, z, torch.ones_like(z))
    x = K[0, 0] * means_camera[:, 0] / safe_z + K[0, 2]
    y = K[1, 1] * means_camera[:, 1] / safe_z + K[1, 2]
    return torch.stack((x, y), dim=-1).unsqueeze(0)


class WarpRasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        means2d: torch.Tensor,
        w2c: torch.Tensor,
        K: torch.Tensor,
        H: int,
        W: int,
        bg_color: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # torch.TensorをWarpのwp.arrayに変換する. contiguous()でメモリレイアウトを保証
        # GCで回収されないようにctxで保持するためownersにまとめる
        owners = [
            means.detach().contiguous(),
            quats.detach().contiguous(),
            scales.detach().contiguous(),
            opacities.detach().contiguous(),
            colors.detach().contiguous(),
            means2d[0].detach().contiguous(),
            w2c.detach().contiguous(),
        ]

        means_wp = wp.from_torch(owners[0], dtype=wp.vec3, requires_grad=True)
        quats_wp = wp.from_torch(owners[1], dtype=wp.quat, requires_grad=True)
        scales_wp = wp.from_torch(owners[2], dtype=wp.vec3, requires_grad=True)
        opacities_wp = wp.from_torch(owners[3], dtype=wp.float32, requires_grad=True)
        colors_wp = wp.from_torch(owners[4], dtype=wp.vec3, requires_grad=True)
        mean_pixel = wp.from_torch(owners[5], dtype=wp.vec2, requires_grad=False)

        # w2c (4, 4) を (1, 4, 4) に変形し, 長さ1のwp.mat44 の配列として Warp にマッピングする
        w2c_1_4_4 = owners[6].view(1, 4, 4)
        view_matrix_arr = wp.from_torch(w2c_1_4_4, dtype=wp.mat44, requires_grad=False)

        bg_wp = wp.vec3(float(bg_color[0].item()), float(bg_color[1].item()), float(bg_color[2].item()))

        # torch CUDAストリーム上でWarp処理を行う
        device = wp.device_from_torch(means.device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(means.device))

        # タイルの情報
        tiles_per_row = math.ceil(W / TILE_SIZE)
        tiles_per_col = math.ceil(H / TILE_SIZE)
        num_tiles = tiles_per_row * tiles_per_col

        # warp arrayの確保
        n = means.shape[0]
        cov2d_inv = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=True)
        depth = wp.zeros(n, dtype=wp.float32, device=device)
        radii = wp.zeros(n, dtype=wp.vec2, device=device)
        tile_rect = wp.zeros(n, dtype=wp.vec4i, device=device)
        tiles_touched = wp.zeros(n, dtype=wp.int32, device=device)
        offsets = wp.zeros(n, dtype=wp.int32, device=device)
        tile_ranges = wp.zeros(num_tiles, dtype=wp.vec2i, device=device)
        image_wp = wp.zeros((H, W), dtype=wp.vec3, device=device)
        final_T = wp.zeros((H, W), dtype=wp.float32, device=device)  # 最終的な透過率
        last_contributor = wp.zeros((H, W), dtype=wp.int32, device=device)  # 最後に寄与したガウシアンのインデックス

        # カメラ内部パラメータ
        fx = float(K[0, 0].item())
        fy = float(K[1, 1].item())
        cx = float(K[0, 2].item())
        cy = float(K[1, 2].item())

        tape = wp.Tape()

        with wp.ScopedStream(stream):
            with tape:
                wp.launch(
                    preprocess_kernel,
                    dim=n,
                    inputs=[
                        means_wp,
                        scales_wp,
                        quats_wp,
                        mean_pixel,
                        view_matrix_arr,
                        fx,
                        fy,
                        cx,
                        cy,
                        float(W),
                        float(H),
                        tiles_per_row,
                        tiles_per_col,
                    ],
                    outputs=[cov2d_inv, depth, tile_rect, tiles_touched, radii],
                    device=device,
                )

            # Prefix sum (exclusive) 「累積和」を計算して各ガウシアンの書き込み先を確定する
            # 例: input = [3, 1, 5, 2] → output = [0, 3, 4, 9] (最後の要素は総和)
            wp.utils.array_scan(tiles_touched, offsets, inclusive=False)
            offsets_torch = wp.to_torch(offsets)
            touched_torch = wp.to_torch(tiles_touched)
            total_pairs = int((offsets_torch[-1] + touched_torch[-1]).item())

            # radix sort用のペア配列を確保する. pair_keys = (tile_id, depth), pair_values = ガウシアンindex
            pair_keys = wp.zeros(2 * total_pairs, dtype=wp.uint64, device=device)
            pair_values = wp.zeros(2 * total_pairs, dtype=wp.int32, device=device)

            # Duplicate: (tile_id, depth)キー と ガウシアンindex のペア配列を構築
            wp.launch(
                duplicate_kernel,
                dim=n,
                inputs=[tile_rect, tiles_touched, depth, offsets, tiles_per_row],
                outputs=[pair_keys, pair_values],
                device=device,
            )

            # radix sort: タイルごとに深度ソート. tile_idの昇順、同一tile_id内はdepthの昇順になる
            wp.utils.radix_sort_pairs(keys=pair_keys, values=pair_values, count=total_pairs)

            # Tile Range: 各タイルの担当範囲を特定
            wp.launch(
                compute_tile_range_kernel,
                dim=total_pairs,
                inputs=[pair_keys, total_pairs],
                outputs=[tile_ranges],
                device=device,
            )

            # Render: タイルごとに深度順にガウシアンを合成して最終的なピクセル色を計算
            wp.launch(
                render_kernel,
                dim=(W, H),
                inputs=[
                    tile_ranges,
                    pair_values,
                    cov2d_inv,
                    mean_pixel,
                    colors_wp,
                    opacities_wp,
                    W,
                    H,
                    tiles_per_row,
                    bg_wp,
                ],
                outputs=[image_wp, final_T, last_contributor],
                device=device,
            )

        # backwardで再利用するものをctxに保存する
        ctx.tape = tape
        ctx.input_arrays = (means_wp, quats_wp, scales_wp, opacities_wp, colors_wp)
        ctx.render_data = (
            tile_ranges,
            pair_values,
            cov2d_inv,
            mean_pixel,
            final_T,
            last_contributor,
            W,
            H,
            tiles_per_row,
            bg_wp,
        )
        ctx.owners = owners
        ctx.stream = stream

        radii_torch = wp.to_torch(radii)
        ctx.mark_non_differentiable(radii_torch)
        return wp.to_torch(image_wp), radii_torch

    @staticmethod
    def backward(ctx, grad_image: torch.Tensor, _grad_radii: torch.Tensor | None):  # type: ignore[override]

        # torchから来た勾配をwp.arrayに変換する. contiguous()でメモリレイアウトを保証
        grad_image_wp = wp.from_torch(grad_image.contiguous(), dtype=wp.vec3)
        means_wp, quats_wp, scales_wp, opacities_wp, colors_wp = ctx.input_arrays
        (
            tile_ranges,
            pair_values,
            cov2d_inv,
            mean_pixel,
            final_transmittance,
            last_contributor,
            W,
            H,
            tiles_per_row,
            bg_wp,
        ) = ctx.render_data

        device = means_wp.device

        # 出力の勾配を格納するwp.arrayを確保する
        grad_cov2d_inv = wp.zeros_like(cov2d_inv)
        grad_mean_pixel = wp.zeros_like(mean_pixel)
        grad_colors = wp.zeros_like(colors_wp)
        grad_opacities = wp.zeros_like(opacities_wp)

        with wp.ScopedStream(ctx.stream):
            for array in (means_wp, quats_wp, scales_wp):
                array.grad.zero_()  # 勾配を初期化して累積を防止

            # backward_render_kernelを起動してgrad_imageから各ガウシアンの勾配を計算する
            wp.launch(
                backward_render_kernel,
                dim=(W, H),
                inputs=[
                    tile_ranges,
                    pair_values,
                    cov2d_inv,
                    mean_pixel,
                    colors_wp,
                    opacities_wp,
                    final_transmittance,
                    last_contributor,
                    grad_image_wp,
                    W,
                    H,
                    tiles_per_row,
                    bg_wp,
                ],
                outputs=[grad_cov2d_inv, grad_mean_pixel, grad_colors, grad_opacities],
                device=device,
            )

            # 共分散の投影だけWarpで逆伝播する。2D中心の勾配はPyTorch側へ返す
            ctx.tape.backward(grads={cov2d_inv: grad_cov2d_inv})

        result_arrays = (means_wp.grad, quats_wp.grad, scales_wp.grad, grad_opacities, grad_colors)

        # 勾配をtorchへ変換して返す (tape.zero()する前にクローンが必須)
        gradients: list[torch.Tensor | None] = []
        for index, array in enumerate(result_arrays):
            if ctx.needs_input_grad[index]:
                gradients.append(wp.to_torch(array).clone())
            else:
                gradients.append(None)

        grad_means2d = wp.to_torch(grad_mean_pixel).clone().unsqueeze(0) if ctx.needs_input_grad[5] else None
        ctx.tape.zero()

        # forwardの引数の順序・個数に合わせてタプルで返す
        return (*gradients, grad_means2d, None, None, None, None, None)
