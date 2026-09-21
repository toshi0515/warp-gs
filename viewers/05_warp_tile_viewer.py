"""
3D Gaussian Splatting - タイルベースラスタライザ (Warp実装)

参考記事: https://zenn.dev/abist_inno/articles/cb13d00751b2c9
アルゴリズム構成 (記事の4章に対応):
    1. preprocess_kernel      : カリング・射影・タイル重複数のカウント (記事2章 Preprocess Pass)
    2. wp.utils.array_scan    : Prefix Scan (記事4.1節, Warp組み込みで代替)
    3. duplicate_kernel       : (tile_id, gaussian_index) 対応配列の構築 (記事4.2節)
    4. wp.utils.radix_sort_pairs : 深度+タイルIDキーでソート (記事4.3節, Warp組み込みで代替)
    5. compute_tile_range_kernel : 各タイルの担当範囲を特定 (記事4.4節)
    6. render_kernel          : タイルごとにα合成 (記事4.5節)
"""

import math
from pathlib import Path

import warp as wp

from warpgs.ply import load_ply
from warpgs.viewer_base import GSViewerBase

wp.init()

TILE_SIZE = 16
GAUSSIAN_EXTENT = 3.33
MAX_ALPHA = 0.99
ALPHA_THRESHOLD = 1.0 / 255.0
TRANSMITTANCE_THRESHOLD = 1e-4


def glm_view_to_wp(view) -> wp.mat44:
    """PyGLMのビュー行列をWarpで使う座標系へ変換する

    Args:
        view: PyGLMのビュー行列

    Returns:
        wp.mat44: Warpのビュー行列
    """
    axis_sign = (1.0, -1.0, -1.0, 1.0)
    values = [axis_sign[row] * float(view[col][row]) for row in range(4) for col in range(4)]
    return wp.mat44(*values)


mat23f = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


@wp.func
def compute_cov3d_wp(scale: wp.vec3, quat: wp.quat) -> wp.mat33:
    """スケールと回転から3次元共分散行列を計算する

    Args:
        scale (wp.vec3): 各軸のスケール
        quat (wp.quat): 回転を表すクォータニオン

    Returns:
        wp.mat33: 3次元共分散行列
    """
    # スケール行列(対角行列)を構築
    S = wp.mat33(scale[0], 0, 0, 0, scale[1], 0, 0, 0, scale[2])

    quat_norm_sq = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
    inv_quat_norm = 1.0 / wp.sqrt(wp.max(quat_norm_sq, 1e-12))
    quat_normalized = wp.quat(
        quat[0] * inv_quat_norm,
        quat[1] * inv_quat_norm,
        quat[2] * inv_quat_norm,
        quat[3] * inv_quat_norm,
    )
    R = wp.quat_to_matrix(quat_normalized)
    # RS = R * S
    RS = R @ S
    # cov3d = RS * RS^T
    cov3d = RS @ wp.transpose(RS)
    return cov3d


@wp.struct
class GaussianScreenData:
    cov2d_diag: wp.vec2  # AABB半径計算用の対角成分 (a, c)
    mu_screen: wp.vec2
    t_cam_z: float
    cov2d_inv: wp.vec3


@wp.func
def compute_gaussian_screen_data_wp(
    mean: wp.vec3,
    cov3d: wp.mat33,
    view_matrix: wp.mat44,
    fx: float,
    fy: float,
    screen_w: float,
    screen_h: float,
) -> GaussianScreenData:
    """3次元ガウシアンを画面へ投影し、描画に必要な情報を計算する

    Args:
        mean (wp.vec3): ガウシアンの中心座標
        cov3d (wp.mat33): 3次元共分散行列
        view_matrix (wp.mat44): ワールド座標からカメラ座標への変換行列
        fx (float): 水平方向の焦点距離
        fy (float): 垂直方向の焦点距離
        screen_w (float): 画面の幅
        screen_h (float): 画面の高さ

    Returns:
        GaussianScreenData: 画面上の中心座標, 深度, 2次元共分散行列の
            対角成分と逆行列
    """
    mean_h = wp.vec4(mean[0], mean[1], mean[2], 1.0)
    t_cam_h = view_matrix @ mean_h

    tx = t_cam_h[0]
    ty = t_cam_h[1]
    tz = t_cam_h[2]

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

    mu_screen = wp.vec2(fx * tx / tz, fy * ty / tz)

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
    result.mu_screen = mu_screen
    result.t_cam_z = tz
    result.cov2d_inv = wp.vec3(inv_a, inv_b, inv_c)
    return result


@wp.kernel
def preprocess_kernel(
    means: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    view_matrix: wp.mat44,
    fx: float,
    fy: float,
    screen_w: float,
    screen_h: float,
    tiles_per_row: int,
    tiles_per_col: int,
    # 出力 (すべて元のガウシアンインデックス tid で書き込む. サイズ N)
    out_cov2d_inv: wp.array[wp.vec3],
    out_mu_screen: wp.array[wp.vec2],
    out_depth: wp.array[wp.float32],
    out_tile_rect: wp.array[wp.vec4i],  # (min_x, min_y, max_x, max_y)
    out_tiles_touched: wp.array[wp.int32],
):
    """可視ガウシアンを抽出し, タイル描画に必要な情報を出力する

    Args:
        means (wp.array[wp.vec3]): ガウシアンの中心座標
        scales (wp.array[wp.vec3]): 各軸のスケール
        quats (wp.array[wp.quat]): 回転を表すクォータニオン
        view_matrix (wp.mat44): ワールド座標からカメラ座標への変換行列
        fx (float): 水平方向の焦点距離
        fy (float): 垂直方向の焦点距離
        screen_w (float): 画面の幅
        screen_h (float): 画面の高さ
        tiles_per_row (int): 横方向のタイル数
        tiles_per_col (int): 縦方向のタイル数
        out_cov2d_inv (wp.array[wp.vec3]): 2次元共分散行列の逆行列出力先
        out_mu_screen (wp.array[wp.vec2]): 画面上の中心座標出力先
        out_depth (wp.array[wp.float32]): カメラからの深度出力先
        out_tile_rect (wp.array[wp.vec4i]): 重なるタイル範囲の出力先
        out_tiles_touched (wp.array[wp.int32]): 重なるタイル数の出力先
    """
    tid = wp.tid()

    # 可視外のデフォルト値, 早期returnする全パスでこれが有効になる
    out_tiles_touched[tid] = 0

    mean = means[tid]
    mean_h = wp.vec4(mean[0], mean[1], mean[2], 1.0)
    t_cam_h = view_matrix @ mean_h
    tz = t_cam_h[2]

    if tz <= 0.1:
        return

    scale = scales[tid]
    quat = quats[tid]
    cov3d = compute_cov3d_wp(scale, quat)
    data = compute_gaussian_screen_data_wp(mean, cov3d, view_matrix, fx, fy, screen_w, screen_h)

    # ガウシアンのAABB
    radius_x = GAUSSIAN_EXTENT * wp.sqrt(wp.max(data.cov2d_diag[0], 0.0))
    radius_y = GAUSSIAN_EXTENT * wp.sqrt(wp.max(data.cov2d_diag[1], 0.0))

    # mu_screenは画像中心原点かつ+y下向き, 左上原点へ平行移動
    pixel_x = data.mu_screen[0] + 0.5 * screen_w
    pixel_y = data.mu_screen[1] + 0.5 * screen_h

    # 画面内カリング
    if pixel_x + radius_x <= 0.0 or pixel_x - radius_x >= screen_w:
        return
    if pixel_y + radius_y <= 0.0 or pixel_y - radius_y >= screen_h:
        return

    # タイル範囲は[min, max)で計算し既存の保存形式だけinclusive maxにする
    min_x = wp.max(wp.int32(wp.floor((pixel_x - radius_x) / float(TILE_SIZE))), 0)
    min_y = wp.max(wp.int32(wp.floor((pixel_y - radius_y) / float(TILE_SIZE))), 0)
    max_x_exclusive = wp.min(wp.int32(wp.ceil((pixel_x + radius_x) / float(TILE_SIZE))), tiles_per_row)
    max_y_exclusive = wp.min(wp.int32(wp.ceil((pixel_y + radius_y) / float(TILE_SIZE))), tiles_per_col)

    if max_x_exclusive <= min_x or max_y_exclusive <= min_y:
        return

    out_cov2d_inv[tid] = data.cov2d_inv
    out_mu_screen[tid] = data.mu_screen
    out_depth[tid] = tz
    out_tile_rect[tid] = wp.vec4i(min_x, min_y, max_x_exclusive - 1, max_y_exclusive - 1)
    out_tiles_touched[tid] = (max_x_exclusive - min_x) * (max_y_exclusive - min_y)


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
    """ガウシアンを重なる各タイルへ複製し, ソート用ペアを作成する

    Args:
        tile_rect (wp.array[wp.vec4i]): 各ガウシアンが重なるタイル範囲
        tiles_touched (wp.array[wp.int32]): 各ガウシアンが重なるタイル数
        depth (wp.array[wp.float32]): カメラからの深度
        offsets (wp.array[wp.int32]): 各ガウシアンの書き込み開始位置
        tiles_per_row (int): 横方向のタイル数
        pair_keys (wp.array[wp.uint64]): タイルIDと深度をまとめたキーの出力先
        pair_values (wp.array[wp.int32]): ガウシアンインデックスの出力先
    """
    tid = wp.tid()

    count = tiles_touched[tid]
    if count == 0:
        return

    rect = tile_rect[tid]
    min_x = rect[0]
    min_y = rect[1]
    max_x = rect[2]
    max_y = rect[3]

    # depthをそのまま下位32bitに使えるよう正の整数化
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
    tile_ranges: wp.array[wp.vec2i],
    num_pairs: int,
):
    """ソート済みペアから各タイルの担当範囲を計算する

    Args:
        pair_keys (wp.array[wp.uint64]): タイルIDを含むソート済みキー
        tile_ranges (wp.array[wp.vec2i]): 各タイルの開始位置と終了位置の出力先
        num_pairs (int): 有効なペア数
    """
    tid = wp.tid()
    if tid >= num_pairs:
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

    if tid == num_pairs - 1:
        tile_ranges[curr_tile][1] = num_pairs


@wp.kernel
def render_kernel(
    tile_ranges: wp.array[wp.vec2i],
    pair_values: wp.array[wp.int32],
    cov2d_inv: wp.array[wp.vec3],
    mu_screen: wp.array[wp.vec2],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    screen_w: int,
    screen_h: int,
    tiles_per_row: int,
    bg_color: wp.vec3,
    framebuffer: wp.array3d[wp.float32],  # shape (H, W, 4)
):
    """ピクセルごとにガウシアンを合成し, フレームバッファへ描画する

    Args:
        tile_ranges (wp.array[wp.vec2i]): 各タイルが参照するペアの範囲
        pair_values (wp.array[wp.int32]): ソート済みガウシアンインデックス
        cov2d_inv (wp.array[wp.vec3]): 2次元共分散行列の逆行列
        mu_screen (wp.array[wp.vec2]): 画面上の中心座標
        colors (wp.array[wp.vec3]): ガウシアンの色
        opacities (wp.array[wp.float32]): ガウシアンの不透明度
        screen_w (int): 画面の幅
        screen_h (int): 画面の高さ
        tiles_per_row (int): 横方向のタイル数
        bg_color (wp.vec3): 背景色
        framebuffer (wp.array3d[wp.float32]): RGBA画像の書き込み先
    """
    px, py = wp.tid()

    if px >= screen_w or py >= screen_h:
        return

    tile_x = px // TILE_SIZE
    tile_y = py // TILE_SIZE
    tile_id = tile_y * tiles_per_row + tile_x

    tr = tile_ranges[tile_id]
    start = tr[0]
    end = tr[1]

    # mu_screenと同じ、画像中心原点かつ+y下向きのピクセル中心
    px_center = float(px) + 0.5 - 0.5 * float(screen_w)
    py_center = float(py) + 0.5 - 0.5 * float(screen_h)

    # Warp 1.16では裸のリテラル代入はコンパイル時定数になる
    # 動的ループ内で更新するため明示的に動的変数として宣言
    T = float(1.0)  # noqa: UP018
    accum = wp.vec3(0.0, 0.0, 0.0)

    for i in range(start, end):
        gid = pair_values[i]

        mu = mu_screen[gid]
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

        if T < TRANSMITTANCE_THRESHOLD:
            break

    final_color = accum + T * bg_color

    framebuffer[py, px, 0] = final_color[0]
    framebuffer[py, px, 1] = final_color[1]
    framebuffer[py, px, 2] = final_color[2]
    framebuffer[py, px, 3] = 1.0


class GSViewerTileRasterizer(GSViewerBase):
    title = "3DGS Tile-Based Rasterizer (Warp) viewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(self.ctx.info)

        # フルスクリーンQuad表示用シェーダー (頂点属性なし, gl_VertexIDから生成)
        self.display_prog = self.ctx.program(
            vertex_shader="""
            #version 430 core
            out vec2 vUv;
            void main() {
                // 3頂点で画面全体を覆う
                vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                vUv = pos;
                gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 430 core
            in vec2 vUv;
            uniform sampler2D u_tex;
            out vec4 FragColor;
            void main() {
                // OpenGL texture row 0 is the bottom row, while the CUDA
                // framebuffer uses image row 0 as the top row.
                FragColor = texture(u_tex, vec2(vUv.x, 1.0 - vUv.y));
            }
            """,
        )
        self.display_vao = self.ctx.vertex_array(self.display_prog, [])

        ply_path = Path(__file__).resolve().parents[1] / "assets" / "models" / "final.ply"
        vertics_dict = load_ply(path=ply_path, is_quats_x_start=True)

        means = vertics_dict["means"].cuda()
        quats = vertics_dict["quats"].cuda()
        scales = vertics_dict["scales"].cuda()
        opacities = vertics_dict["opacities"].cuda()
        colors = vertics_dict["colors"].cuda()

        self.wp_means = wp.from_torch(means.contiguous(), dtype=wp.vec3)
        self.wp_scales = wp.from_torch(scales.contiguous(), dtype=wp.vec3)
        self.wp_quats = wp.from_torch(quats.contiguous(), dtype=wp.quat)
        self.wp_colors = wp.from_torch(colors.contiguous(), dtype=wp.vec3)
        self.wp_opacities = wp.from_torch(opacities.contiguous(), dtype=wp.float32)

        self.N = len(vertics_dict["means"])

        # preprocess_kernel出力バッファ (サイズN固定)
        self.out_cov2d_inv = wp.zeros(self.N, dtype=wp.vec3)
        self.out_mu_screen = wp.zeros(self.N, dtype=wp.vec2)
        self.out_depth = wp.zeros(self.N, dtype=wp.float32)
        self.out_tile_rect = wp.zeros(self.N, dtype=wp.vec4i)
        self.out_tiles_touched = wp.zeros(self.N, dtype=wp.int32)
        self.offsets = wp.zeros(self.N, dtype=wp.int32)

        # タイルグリッド
        W, H = self.window_size
        self.tiles_per_row = math.ceil(W / TILE_SIZE)
        self.tiles_per_col = math.ceil(H / TILE_SIZE)
        num_tiles = self.tiles_per_row * self.tiles_per_col
        self.tile_ranges = wp.zeros(num_tiles, dtype=wp.vec2i)

        # ソート用バッファ (サイズは使いながら拡張する)
        self.pair_capacity = self.N  # 初期サイズ, 足りなければ_ensure_sort_capacityで拡張
        self.pair_keys = wp.zeros(self.pair_capacity * 2, dtype=wp.uint64)
        self.pair_values = wp.zeros(self.pair_capacity * 2, dtype=wp.int32)

        # 表示用テクスチャ + PBO
        self.tex = self.ctx.texture((W, H), components=4, dtype="f4")
        self.tex.repeat_x = False
        self.tex.repeat_y = False
        self.fb_pbo = self.ctx.buffer(reserve=W * H * 4 * 4)  # RGBA float32
        self.cuda_gl_fb = wp.RegisteredGLBuffer(self.fb_pbo.glo)
        self.cuda_gl_fb.flags = 2  # WRITE_DISCARD

    def _ensure_sort_capacity(self, min_size: int):
        """ソート用バッファを必要な要素数まで拡張する。

        Args:
            min_size (int): 必要な最小要素数。
        """
        if min_size <= self.pair_capacity:
            return
        new_capacity = max(min_size, int(self.pair_capacity * 1.5))
        self.pair_capacity = new_capacity
        # radix_sort_pairsはcountの2倍以上の容量を要求する
        self.pair_keys = wp.zeros(new_capacity * 2, dtype=wp.uint64)
        self.pair_values = wp.zeros(new_capacity * 2, dtype=wp.int32)

    def on_render(self, time, frame_time):
        W, H = self.window_size
        fov_rad = math.radians(self.fov)

        view, projection = super().move_camera(frame_time)

        fy = (H / 2) / math.tan(fov_rad / 2)
        fx = fy
        view_wp = glm_view_to_wp(view)

        # 1. Preprocess: カリング・射影・タイル重複数カウント
        wp.launch(
            kernel=preprocess_kernel,
            dim=self.N,
            inputs=[
                self.wp_means,
                self.wp_scales,
                self.wp_quats,
                view_wp,
                float(fx),
                float(fy),
                float(W),
                float(H),
                self.tiles_per_row,
                self.tiles_per_col,
                self.out_cov2d_inv,
                self.out_mu_screen,
                self.out_depth,
                self.out_tile_rect,
                self.out_tiles_touched,
            ],
        )

        # 2. Prefix Scan: 各ガウシアンの書き込み開始位置を求める
        wp.utils.array_scan(self.out_tiles_touched, self.offsets, inclusive=False)

        # total_pairs = 最後のガウシアンのoffset + そのtiles_touched
        last_offset = int(self.offsets.numpy()[-1])
        last_touched = int(self.out_tiles_touched.numpy()[-1])
        total_pairs = last_offset + last_touched

        if total_pairs == 0:
            self.ctx.clear(0.2, 0.3, 0.3, 1.0)
            return

        self._ensure_sort_capacity(total_pairs)

        # 3. Duplicate: (tile_id, depth)キー と ガウシアンindex のペア配列を構築
        wp.launch(
            kernel=duplicate_kernel,
            dim=self.N,
            inputs=[
                self.out_tile_rect,
                self.out_tiles_touched,
                self.out_depth,
                self.offsets,
                self.tiles_per_row,
                self.pair_keys,
                self.pair_values,
            ],
        )

        # 4. Radix Sort: タイルID -> 深度の順にソート
        wp.utils.radix_sort_pairs(keys=self.pair_keys, values=self.pair_values, count=total_pairs)

        # 5. Tile Range: 各タイルの担当範囲を特定 (毎フレーム空タイルとしてリセットしてから)
        self.tile_ranges.zero_()
        wp.launch(
            kernel=compute_tile_range_kernel,
            dim=total_pairs,
            inputs=[self.pair_keys, self.tile_ranges, total_pairs],
        )

        # 6. Tile Render: フレームバッファに直接書き込む
        fb_array = self.cuda_gl_fb.map(dtype=wp.float32, shape=(H, W, 4))
        bg_color = wp.vec3(0.2, 0.3, 0.3)
        wp.launch(
            kernel=render_kernel,
            dim=(W, H),
            inputs=[
                self.tile_ranges,
                self.pair_values,
                self.out_cov2d_inv,
                self.out_mu_screen,
                self.wp_colors,
                self.wp_opacities,
                W,
                H,
                self.tiles_per_row,
                bg_color,
                fb_array,
            ],
        )
        self.cuda_gl_fb.unmap()

        # PBO -> テクスチャへGPU内転送
        self.tex.write(self.fb_pbo)

        # フルスクリーンQuadでテクスチャを表示
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tex.use(location=0)
        self.display_prog["u_tex"] = 0
        self.display_vao.render(mode=self.ctx.TRIANGLES, vertices=3)
        self.ctx.finish()


if __name__ == "__main__":
    GSViewerTileRasterizer.run()
