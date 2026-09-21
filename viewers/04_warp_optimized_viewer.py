import math
from pathlib import Path

import moderngl
import warp as wp

from warpgs.ply import load_ply
from warpgs.viewer_base import GSViewerBase

wp.init()

GAUSSIAN_EXTENT = 3.33


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
class ComputeCov2dResult:
    cov2d: wp.mat22
    mu_screen: wp.vec2
    t_cam: wp.vec3
    cov2d_diag: wp.vec2


mat23f = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


@wp.func
def compute_cov2d_wp(
    mean: wp.vec3,
    cov3d: wp.mat33,
    view_matrix: wp.mat44,
    fx: float,
    fy: float,
    screen_w: float,
    screen_h: float,
) -> ComputeCov2dResult:
    """3次元ガウシアンを画面へ投影し, 2次元共分散行列を計算する

    Args:
        mean (wp.vec3): ガウシアンの中心座標
        cov3d (wp.mat33): 3次元共分散行列
        view_matrix (wp.mat44): ワールド座標からカメラ座標への変換行列
        fx (float): 水平方向の焦点距離
        fy (float): 垂直方向の焦点距離
        screen_w (float): 画面の幅
        screen_h (float): 画面の高さ

    Returns:
        ComputeCov2dResult: 2次元共分散行列, 画面上の中心座標,
            カメラ座標, 共分散行列の対角成分
    """
    # ガウシアンの中心を同時座標系に変換
    mean_h = wp.vec4(mean[0], mean[1], mean[2], 1.0)
    # ガウシアンの中心をカメラ座標に変換
    t_cam_h = view_matrix @ mean_h

    tx = t_cam_h[0]
    ty = t_cam_h[1]
    tz = t_cam_h[2]

    lim_x = 1.3 * 0.5 * screen_w / fx
    lim_y = 1.3 * 0.5 * screen_h / fy
    tx_cov = tz * wp.clamp(tx / tz, -lim_x, lim_x)
    ty_cov = tz * wp.clamp(ty / tz, -lim_y, lim_y)
    J = mat23f(fx / tz, 0.0, -(fx * tx_cov) / (tz * tz), 0.0, fy / tz, -(fy * ty_cov) / (tz * tz))

    # View行列から回転行列構築
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

    # スクリーン座標系のガウシアンの中心
    mu_screen = wp.vec2(fx * tx / tz, fy * ty / tz)

    # cov_cam = W * cov3d * W^T
    cov_cam = W_rot @ cov3d @ wp.transpose(W_rot)

    # cov2d = J * cov_cam * J^T
    cov2d = J @ cov_cam @ wp.transpose(J)
    cov2d[0, 0] += 0.3
    cov2d[1, 1] += 0.3

    result = ComputeCov2dResult()
    result.cov2d = cov2d
    result.mu_screen = mu_screen
    result.t_cam = wp.vec3(tx, ty, tz)
    result.cov2d_diag = wp.vec2(cov2d[0, 0], cov2d[1, 1])

    return result


@wp.struct
class ComputeQuadResult:
    cov2d_inv: wp.vec3
    u_axis: wp.vec2
    v_axis: wp.vec2


@wp.func
def compute_quad_wp(cov2d: wp.mat22) -> ComputeQuadResult:
    """2次元共分散行列から描画用Quadの情報を計算する

    Args:
        cov2d (wp.mat22): 画面上の2次元共分散行列

    Returns:
        ComputeQuadResult: 共分散行列の逆行列とQuadの2本の軸
    """
    # cov2dの逆行列を計算
    a = cov2d[0, 0]
    b = cov2d[0, 1]
    c = cov2d[1, 1]

    det = (a * c) - (b * b)
    det = wp.max(det, 1e-6)

    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det

    cov2d_inv = wp.vec3(inv_a, inv_b, inv_c)

    # cov2dの固有値分解
    mid = 0.5 * (a + c)
    diff = wp.sqrt(wp.max((0.5 * (a - c)) * (0.5 * (a - c)) + b * b, 1e-6))

    lambda1 = mid + diff
    lambda2 = wp.max(mid - diff, 1e-6)

    theta = 0.5 * wp.atan2(2.0 * b, a - c)
    cos_t = wp.cos(theta)
    sin_t = wp.sin(theta)

    r1 = GAUSSIAN_EXTENT * wp.sqrt(lambda1)
    r2 = GAUSSIAN_EXTENT * wp.sqrt(lambda2)

    u_axis = wp.vec2(cos_t * r1, sin_t * r1)
    v_axis = wp.vec2(-sin_t * r2, cos_t * r2)

    result = ComputeQuadResult()
    result.cov2d_inv = cov2d_inv
    result.u_axis = u_axis
    result.v_axis = v_axis

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
    out_indices: wp.array[wp.int32],
    out_depth: wp.array[wp.float32],
    visible_count: wp.array[wp.int32],
):
    """画面内のガウシアンを抽出し、深度とインデックスを出力する

    Args:
        means (wp.array[wp.vec3]): ガウシアンの中心座標
        scales (wp.array[wp.vec3]): 各軸のスケール
        quats (wp.array[wp.quat]): 回転を表すクォータニオン
        view_matrix (wp.mat44): ワールド座標からカメラ座標への変換行列
        fx (float): 水平方向の焦点距離
        fy (float): 垂直方向の焦点距離
        screen_w (float): 画面の幅
        screen_h (float): 画面の高さ
        out_indices (wp.array[wp.int32]): 可視ガウシアンのインデックス出力先
        out_depth (wp.array[wp.float32]): ソート用の深度出力先
        visible_count (wp.array[wp.int32]): 可視ガウシアン数を格納する配列
    """
    tid = wp.tid()

    # 座標の取得とカメラ空間への変換
    mean = means[tid]
    mean_h = wp.vec4(mean[0], mean[1], mean[2], 1.0)
    t_cam_h = view_matrix @ mean_h

    tz = t_cam_h[2]

    # glm_view_to_wp()で+Z forwardへ変換済み。
    if tz <= 0.1:
        return

    cov3d = compute_cov3d_wp(scales[tid], quats[tid])
    screen_data = compute_cov2d_wp(mean, cov3d, view_matrix, fx, fy, screen_w, screen_h)

    # 中心だけでなく、gsplat互換の3.33σ footprintでカリングする。
    radius_x = GAUSSIAN_EXTENT * wp.sqrt(wp.max(screen_data.cov2d_diag[0], 0.0))
    radius_y = GAUSSIAN_EXTENT * wp.sqrt(wp.max(screen_data.cov2d_diag[1], 0.0))
    pixel_x = screen_data.mu_screen[0] + 0.5 * screen_w
    pixel_y = screen_data.mu_screen[1] + 0.5 * screen_h
    if pixel_x + radius_x <= 0.0 or pixel_x - radius_x >= screen_w:
        return
    if pixel_y + radius_y <= 0.0 or pixel_y - radius_y >= screen_h:
        return

    # 可視ガウシアンの書き込み
    # wp.atomic_add は加算前の値を返す性質を利用し、書き込み先のインデックスを取得する
    idx = wp.atomic_add(visible_count, 0, 1)

    # 抽出用配列の先頭から隙間なく詰めて書き込む
    out_indices[idx] = tid
    out_depth[idx] = -tz


@wp.kernel
def render_kernel(
    sorted_indices: wp.array[wp.int32],
    means: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    colors: wp.array[wp.vec3],
    opacities: wp.array[float],
    view_matrix: wp.mat44,
    fx: float,
    fy: float,
    screen_w: float,
    screen_h: float,
    vbo_array: wp.array2d[wp.float32],
):
    """ソート済みガウシアンの描画データをVBOへ書き込む

    Args:
        sorted_indices (wp.array[wp.int32]): 深度順のガウシアンインデックス
        means (wp.array[wp.vec3]): ガウシアンの中心座標
        scales (wp.array[wp.vec3]): 各軸のスケール
        quats (wp.array[wp.quat]): 回転を表すクォータニオン
        colors (wp.array[wp.vec3]): ガウシアンの色
        opacities (wp.array[float]): ガウシアンの不透明度
        view_matrix (wp.mat44): ワールド座標からカメラ座標への変換行列
        fx (float): 水平方向の焦点距離
        fy (float): 垂直方向の焦点距離
        screen_w (float): 画面の幅
        screen_h (float): 画面の高さ
        vbo_array (wp.array2d[wp.float32]): 描画データの書き込み先
    """
    tid = wp.tid()

    # インデックス
    idx = sorted_indices[tid]

    mean = means[idx]
    scale = scales[idx]
    quat = quats[idx]
    color = colors[idx]
    opacity = opacities[idx]

    # 3D共分散行列を構築
    cov3d = compute_cov3d_wp(scale, quat)

    # 2D共分散行列
    cov2d_result = compute_cov2d_wp(mean, cov3d, view_matrix, fx, fy, screen_w, screen_h)

    # Quadを構築
    quad_result = compute_quad_wp(cov2d_result.cov2d)

    # VBOへの書き込み
    # ガウシアン中心
    vbo_array[tid, 0] = cov2d_result.mu_screen[0]
    vbo_array[tid, 1] = cov2d_result.mu_screen[1]
    # opacity
    vbo_array[tid, 2] = opacity
    # color
    vbo_array[tid, 3] = color[0]
    vbo_array[tid, 4] = color[1]
    vbo_array[tid, 5] = color[2]
    # cov2d_inv
    vbo_array[tid, 6] = quad_result.cov2d_inv[0]
    vbo_array[tid, 7] = quad_result.cov2d_inv[1]
    vbo_array[tid, 8] = quad_result.cov2d_inv[2]
    # Quad
    vbo_array[tid, 9] = quad_result.u_axis[0]
    vbo_array[tid, 10] = quad_result.u_axis[1]
    vbo_array[tid, 11] = quad_result.v_axis[0]
    vbo_array[tid, 12] = quad_result.v_axis[1]


class GSViewerWarpOptimized(GSViewerBase):
    title = "3DGS Warp optimized viewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        print(self.ctx.info)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 430 core
    
                // CPU(VBO)から受け取るデータ
                in vec2 aPixelCenter;
                in float aOpacity;
                in vec3 aColor;
                in vec3 aCovInv;
                in vec4 aQuadAxes;
    
                uniform vec2 u_screenSize;
    
                // フラグメントシェーダーへ送るデータ
                out vec3 vColor;
                out vec2 vOffset;
                out float vOpacity;
                out vec3 vCovInv;
    
                void main() {
                    float x_sign = float(gl_VertexID & 1) * 2.0 - 1.0;
                    float y_sign = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
                    
                    // PyTorchから受け取った固有ベクトルを展開
                    vec2 U = aQuadAxes.xy; // 長軸ベクトル
                    vec2 V = aQuadAxes.zw; // 短軸ベクトル
                    
                    // 基底ベクトル(U, V)の線形結合で、斜めに傾いたQuadの頂点を計算
                    vec2 local_offset = x_sign * U + y_sign * V;
    
                    vec2 pixel_pos = aPixelCenter + local_offset;
                    // Warp projection uses image coordinates (+y down); OpenGL NDC uses +y up.
                    vec2 ndc_pos = vec2(
                        pixel_pos.x / (u_screenSize.x / 2.0),
                        -pixel_pos.y / (u_screenSize.y / 2.0)
                    );
    
                    gl_Position = vec4(ndc_pos, 0.0, 1.0); // Z=0.0, W=1.0
    
                    // フラグメントシェーダーへ
                    vColor = aColor;
                    vOffset = local_offset;
                    vOpacity = aOpacity;
                    vCovInv = aCovInv;
                }
            """,
            fragment_shader="""
                #version 430 core
    
                // 頂点シェーダーから受け取るデータ
                in vec3 vColor;
                in vec2 vOffset;
                in float vOpacity;
                in vec3 vCovInv;
    
                out vec4 FragColor;
    
                void main() {
                    // マハラノビス距離の2乗 (d2) を計算
                    float d2 = vOffset.x * vOffset.x * vCovInv.x
                            + 2.0 * vOffset.x * vOffset.y * vCovInv.y
                            + vOffset.y * vOffset.y * vCovInv.z;
    
                    // 3.33シグマ範囲外は計算せずに破棄
                    if (d2 > 11.0889) {
                        discard;
                    }
    
                    // ガウス分布に基づくアルファ値(不透明度)の計算
                    float alpha = min(0.99, vOpacity * exp(-0.5 * d2));
    
                    // 低alphaカットオフ
                    if (alpha < 1.0 / 255.0) {
                        discard;
                    }
    
                    // 最終的な色とアルファ値を出力
                    FragColor = vec4(vColor, alpha);
                }
            """,
        )

        # plyファイルを読み込む
        ply_path = Path(__file__).resolve().parents[1] / "assets" / "models" / "final.ply"
        self.vertics_dict = load_ply(path=ply_path, is_quats_x_start=True)

        means = self.vertics_dict["means"].cuda()
        quats = self.vertics_dict["quats"].cuda()
        scales = self.vertics_dict["scales"].cuda()
        opacities = self.vertics_dict["opacities"].cuda()
        colors = self.vertics_dict["colors"].cuda()

        self.wp_means = wp.from_torch(means.contiguous(), dtype=wp.vec3)
        self.wp_scales = wp.from_torch(scales.contiguous(), dtype=wp.vec3)
        self.wp_quats = wp.from_torch(quats.contiguous(), dtype=wp.quat)
        self.wp_colors = wp.from_torch(colors.contiguous(), dtype=wp.vec3)
        self.wp_opacities = wp.from_torch(opacities.contiguous(), dtype=wp.float32)

        self.N = len(self.vertics_dict["means"])

        # カーネル用バッファ
        self.out_indices = wp.empty(self.N * 2, dtype=wp.int32)
        self.out_depth = wp.empty(self.N * 2, dtype=wp.float32)
        # アトミックカウンタ
        self.visible_count = wp.zeros(1, dtype=wp.int32)

        buffer_size = self.N * 4 * 13
        self.vbo = self.ctx.buffer(reserve=buffer_size)

        # Warp OpenGL interop
        self.cuda_gl_buffer = wp.RegisteredGLBuffer(self.vbo.glo)
        self.cuda_gl_buffer.flags = 2  # Write Discard

        # VAOの生成時に index_buffer として紐付ける (/i をつけることでインスタンシングを有効化)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 1f 3f 3f 4f /i", "aPixelCenter", "aOpacity", "aColor", "aCovInv", "aQuadAxes")],
        )

    def on_render(self, time, frame_time):
        self.ctx.clear(0.2, 0.3, 0.3, 1.0, depth=1.0)
        self.visible_count.zero_()

        W, H = self.window_size  # (800, 600)
        self.prog["u_screenSize"] = (float(W), float(H))

        view, projection = super().move_camera(frame_time)

        # 焦点距離fx, fyの計算
        fov_rad = math.radians(self.fov)
        fy = (H / 2) / math.tan(fov_rad / 2)
        fx = fy

        view_wp = glm_view_to_wp(view)

        # カリング
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
                self.out_indices,
                self.out_depth,
                self.visible_count,
            ],
        )

        num_visible = int(self.visible_count.numpy()[0])

        if num_visible == 0:
            return  # 画面に何も映っていない場合はスキップ

        # Z深度(tz)をキーにして、抽出されたインデックスをソート
        wp.utils.radix_sort_pairs(keys=self.out_depth, values=self.out_indices, count=num_visible)

        # VBOのバッファをWarp arrayに変換
        vbo_array = self.cuda_gl_buffer.map(dtype=wp.float32, shape=(num_visible, 13))
        wp.launch(
            kernel=render_kernel,
            dim=num_visible,
            inputs=[
                self.out_indices,  # ソートされた元のインデックス配列
                self.wp_means,
                self.wp_scales,
                self.wp_quats,
                self.wp_colors,
                self.wp_opacities,
                view_wp,
                float(fx),
                float(fy),
                float(W),
                float(H),
                vbo_array,  # (num_visible, 13) として書き込まれる
            ],
        )
        self.cuda_gl_buffer.unmap()

        self.vao.render(mode=self.ctx.TRIANGLE_STRIP, vertices=4, instances=num_visible)
        self.ctx.finish()


if __name__ == "__main__":
    GSViewerWarpOptimized.run()
