import math
from pathlib import Path

import moderngl
import torch

from warpgs.ply import load_ply
from warpgs.viewer_base import GSViewerBase

GAUSSIAN_EXTENT = 3.33


def compute_cov3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """各ガウシアンの3次元共分散行列を計算する

    Args:
        scales (torch.Tensor): 各軸のスケール (N, 3)
        quats (torch.Tensor): 回転を表すクォータニオン (N, 4)

    Returns:
        torch.Tensor: 3次元共分散行列 (N, 3, 3)
    """
    N = scales.shape[0]

    quats = torch.nn.functional.normalize(quats, dim=1, eps=1e-8)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # 2. 回転行列 R (N, 3, 3) を構築
    R = torch.zeros((N, 3, 3), dtype=torch.float32, device=quats.device)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)
    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)
    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    # 3. スケール行列 S (N, 3, 3) を構築（対角行列）
    S = torch.zeros((N, 3, 3), dtype=torch.float32, device=scales.device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]

    # RS = R * S  (Mの形状は (N, 3, 3))
    RS = torch.bmm(R, S)

    # cov3d = RS * RS^T (M.transpose(1, 2) は各3x3行列の転置)
    cov3d = torch.bmm(RS, RS.transpose(1, 2))

    return cov3d


def compute_cov2d(
    means: torch.Tensor,
    cov3d: torch.Tensor,
    view_matrix: torch.Tensor,
    fx: float,
    fy: float,
    screen_w: float,
    screen_h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3次元ガウシアンを画面へ投影し, 2次元共分散行列を計算する

    Args:
        means (torch.Tensor): ガウシアンの中心座標``(N, 3)``
        cov3d (torch.Tensor): 3次元共分散行列``(N, 3, 3)``
        view_matrix (torch.Tensor): ワールド座標からカメラ座標への変換行列
        fx (float): 水平方向の焦点距離
        fy (float): 垂直方向の焦点距離
        screen_w (float): 画面の幅
        screen_h (float): 画面の高さ

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 画面上の中心座標,
            2次元共分散行列, カメラ座標を格納したタプル
    """
    N = means.shape[0]

    # view_matrix が numpy配列の場合はテンソルに変換
    if not isinstance(view_matrix, torch.Tensor):
        view_matrix = torch.tensor(view_matrix, dtype=torch.float32, device=means.device)

    # ガウシアン中心をカメラ座標系に変換
    # 同次座標系 (N, 4) を作成
    ones = torch.ones((N, 1), dtype=means.dtype, device=means.device)
    means_h = torch.cat([means, ones], dim=1)

    t_cam_h = means_h @ view_matrix.T
    t_cam = t_cam_h[:, :3]  # 同次座標系から(x,y,z)に戻す (N, 3)

    tx = t_cam[:, 0]
    ty = t_cam[:, 1]
    tz = t_cam[:, 2]

    # ヤコビアン J の構築 (N, 2, 3)
    # 共分散のJacobianだけをFOV外でクランプ
    # 投影中心には元のtx, tyを使うため、中心位置は変化しない
    lim_x = 1.3 * 0.5 * screen_w / fx
    lim_y = 1.3 * 0.5 * screen_h / fy
    tx_cov = tz * torch.clamp(tx / tz, min=-lim_x, max=lim_x)
    ty_cov = tz * torch.clamp(ty / tz, min=-lim_y, max=lim_y)

    J = torch.zeros((N, 2, 3), dtype=means.dtype, device=means.device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx_cov / (tz**2)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty_cov / (tz**2)

    # カメラ座標系の共分散 cov_cam
    W_rot = view_matrix[:3, :3]  # (3, 3)

    # torch.bmmを使うために、(3, 3)を (N, 3, 3) に拡張
    W_rot_batched = W_rot.unsqueeze(0).expand(N, -1, -1)

    mu_screen = torch.stack([fx * tx / tz, fy * ty / tz], dim=1)  # (N, 2)

    # cov_cam = W * cov3d * W^T
    # W_rot_batched.transpose(1, 2) で各バッチの3x3行列を転置する
    cov_cam = torch.bmm(W_rot_batched, torch.bmm(cov3d, W_rot_batched.transpose(1, 2)))

    # 2D共分散 cov2d (画面上の楕円の形)
    # cov2d = J * cov_cam * J^T
    cov2d = torch.bmm(J, torch.bmm(cov_cam, J.transpose(1, 2)))  # (N, 2, 2)
    cov2d = cov2d + 0.3 * torch.eye(2, dtype=means.dtype, device=means.device).unsqueeze(0)

    return mu_screen, cov2d, t_cam


class GSViewer(GSViewerBase):
    title = "3DGS PyToech viewer"

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
                // Pinhole pixels use +Y down while OpenGL NDC uses +Y up.
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

                // 3.33sigma範囲外は計算せずに破棄
                if (d2 > 11.0889) {
                    discard;
                }

                // ガウス分布に基づくアルファ値(不透明度)の計算
                float alpha = min(0.99, vOpacity * exp(-0.5 * d2));

                // Match gsplat's alpha cutoff and cap so transmittance never reaches zero.
                if (alpha < 1.0 / 255.0) {
                    discard;
                }

                FragColor = vec4(vColor, alpha);
            }
        """,
        )

        # plyファイルを読み込む
        ply_path = Path(__file__).resolve().parents[1] / "assets" / "models" / "final.ply"
        self.vertics_dict = load_ply(ply_path)

        self.means = self.vertics_dict["means"].cuda()
        self.quats = self.vertics_dict["quats"].cuda()
        self.scales = self.vertics_dict["scales"].cuda()
        self.opacities = self.vertics_dict["opacities"].cuda()
        self.colors = self.vertics_dict["colors"].cuda()

        # cov3dを求める
        self.cov3d = compute_cov3d(self.scales, self.quats)

        N = len(self.vertics_dict["means"])

        buffer_size = N * 4 * 13
        self.vbo = self.ctx.buffer(reserve=buffer_size)

        # VAOの生成時に index_buffer として紐付ける (/i をつけることでインスタンシングを有効化)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 1f 3f 3f 4f /i", "aPixelCenter", "aOpacity", "aColor", "aCovInv", "aQuadAxes")],
        )

    def on_render(self, time, frame_time):
        torch.cuda.synchronize()
        self.ctx.clear(0.2, 0.3, 0.3, 1.0, depth=1.0)

        W, H = self.window_size  # (800, 600)
        self.prog["u_screenSize"] = (float(W), float(H))

        view, projection = super().move_camera(frame_time)

        # 焦点距離fx, fyの計算
        fov_rad = math.radians(self.fov)
        fy = (H / 2) / math.tan(fov_rad / 2)
        fx = fy

        # PyGLMの行列は[列][行]のため，PyTorchの[行][列]へ変換
        gl_view = torch.tensor(
            [[view[column][row] for column in range(4)] for row in range(4)],
            dtype=torch.float32,
            device=self.means.device,
        )
        # OpenGL (右手系: +Y up, -Z forward) -> PinHole/CV (右手系: +Y down, +Z forward)
        opengl_to_pinhole = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=self.means.device))
        view_matrix = opengl_to_pinhole @ gl_view

        # カメラの前方部分のみ計算する
        ones = torch.ones((self.means.shape[0], 1), dtype=self.means.dtype, device=self.means.device)
        t_cam_all = torch.cat([self.means, ones], dim=1) @ view_matrix.T
        in_front = t_cam_all[:, 2] > 0.01
        front_indices = torch.nonzero(in_front, as_tuple=False).squeeze(1)
        if front_indices.numel() == 0:
            return

        # 2Dへ投影して，共分散行列から画面内にあるガウシアンのみにカリングする
        mu_screen, cov2d, t_cam = compute_cov2d(
            self.means[front_indices], self.cov3d[front_indices], view_matrix, fx, fy, float(W), float(H)
        )
        a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
        max_eigenvalue = 0.5 * (a + c) + torch.sqrt(torch.clamp((0.5 * (a - c)) ** 2 + b**2, min=0.0))
        radius = GAUSSIAN_EXTENT * torch.sqrt(torch.clamp(max_eigenvalue, min=0.0))
        visible = (
            (mu_screen[:, 0] + radius >= -W / 2)
            & (mu_screen[:, 0] - radius <= W / 2)
            & (mu_screen[:, 1] + radius >= -H / 2)
            & (mu_screen[:, 1] - radius <= H / 2)
        )  # boolのマスク
        visible_indices = front_indices[visible]  # 可視ガウシアン番号
        visible_count = visible_indices.numel()
        if visible_count == 0:
            return
        mu_screen, cov2d, t_cam = mu_screen[visible], cov2d[visible], t_cam[visible]
        torch.cuda.synchronize()  # compute_cov2dの完了を待つ

        # マハラノビス距離の計算のためのcov2dの逆行列を求める
        # cov2d は (N, 2, 2) の対称行列
        a = cov2d[:, 0, 0]
        b = cov2d[:, 0, 1]  # または [:, 1, 0]
        c = cov2d[:, 1, 1]

        # 行列式 (Determinant) を計算 (N,)
        det = (a * c) - (b * b)
        det = torch.clamp(det, min=1e-6)

        # 逆行列の要素を計算 (N,)
        inv_a = c / det
        inv_b = -b / det
        inv_c = a / det

        cov2d_inv = torch.stack([inv_a, inv_b, inv_c], dim=1)

        # cov2dの固有値を求める
        mid = 0.5 * (a + c)
        diff = torch.sqrt(torch.clamp((0.5 * (a - c)) ** 2 + b**2, min=1e-6))
        lambda1 = mid + diff
        lambda2 = torch.clamp(mid - diff, min=1e-6)

        # 固有ベクトル（楕円の傾き）の計算
        # 2x2対称行列の固有ベクトルは atan2(2b, a-c) の半分の角度
        theta = 0.5 * torch.atan2(2.0 * b, a - c)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # 各軸の長さ (3シグマ)
        r1 = GAUSSIAN_EXTENT * torch.sqrt(lambda1)
        r2 = GAUSSIAN_EXTENT * torch.sqrt(lambda2)

        # 長軸ベクトル(U) と 短軸ベクトル(V)
        U_x = cos_t * r1
        U_y = sin_t * r1
        V_x = -sin_t * r2
        V_y = cos_t * r2

        # シェーダーへ渡すために (N, 4) のテンソルに結合
        quad_axes = torch.stack([U_x, U_y, V_x, V_y], dim=1)

        # シェーダーへ渡す頂点行列を作成
        opacity = self.opacities[visible_indices].unsqueeze(1)

        # 頂点データの結合 mu_screen:2, opacity:1, colors:3, cov2d_inv:3, quad_axes:4 -> (N, 13)
        vertices = torch.cat([mu_screen, opacity, self.colors[visible_indices], cov2d_inv, quad_axes], dim=1)

        # 深度ソート
        tz = t_cam[:, 2]
        indices = torch.argsort(tz, descending=True)
        vertices = vertices[indices]
        torch.cuda.synchronize()  # ソートの完了を待つ

        vertices_np = vertices.cpu().numpy()
        self.vbo.write(vertices_np.tobytes())
        self.vao.render(mode=self.ctx.TRIANGLE_STRIP, vertices=4, instances=visible_count)
        self.ctx.finish()


if __name__ == "__main__":
    GSViewer.run()
