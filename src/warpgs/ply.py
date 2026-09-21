import os
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from warpgs.model import GaussianModel


def save_ply(model: GaussianModel, path: str) -> None:
    """
    学習済みガウシアンを PLY 形式で保存する
    SuperSplat や antimatter15/splat などのビューアで読み込める

    PLY の各頂点に以下の属性を持たせる:
        x, y, z        : 位置
        nx, ny, nz     : 法線 (3DGS では使わないが形式上必要)
        f_dc_0/1/2     : DC成分の SH係数 (= simple RGB を変換したもの)
        opacity        : logit_opacity
        scale_0/1/2    : log_scale
        rot_0/1/2/3    : クォータニオン (w,x,y,z)
    """
    import struct

    N = model.num_gaussians()

    with torch.no_grad():
        means = model.means.detach().cpu().numpy()
        colors = model.params["colors"].detach().cpu().numpy()
        opacities = model.params["opacities"].detach().cpu().numpy()
        scales = model.params["scales"].detach().cpu().numpy()
        quats = model.quats.detach().cpu().numpy()

    # SH DC 成分への変換: RGB = sigmoid(colors)
    # SH DC = (RGB - 0.5) / 0.2821  (逆変換)
    SH_C0 = 0.28209479177387814
    rgb = 1.0 / (1.0 + np.exp(-colors))  # sigmoid
    f_dc = (rgb - 0.5) / SH_C0  # SH DC 成分

    # PLY ヘッダー
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            # fmt: off
            data = struct.pack(
                "<17f",
                # 位置
                means[i, 0], means[i, 1], means[i, 2],
                # 法線 (ゼロで埋める)
                0.0, 0.0, 0.0,
                # 色 (SH DC)
                f_dc[i, 0], f_dc[i, 1], f_dc[i, 2],
                # 不透明度 (logit)
                opacities[i],
                # スケール (log)
                scales[i, 0], scales[i, 1], scales[i, 2],
                # クォータニオン (w,x,y,z)
                quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]
            )
            f.write(data)
            # fmt: on

    size_mb = os.path.getsize(path) / 1e6
    print(f"[PLY] saved {N} Gaussians ({size_mb:.1f} MB) → {path}")


def load_ply(path: str | Path, is_quats_x_start=False) -> dict:
    """3D Gaussian SplattingのPLYファイルを読み込む

    Args:
        path (str | Path): 読み込むPLYファイルのパス
        is_quats_x_start (bool, optional): ``True`` の場合, クォータニオンをWarp形式の ``(x, y, z, w)`` に並べ替える.
        ``False`` の場合はgsplat形式の ``(w, x, y, z)`` にする. デフォルトは ``False``.

    Returns:
        dict: 次の要素を持つテンソルの辞書。

            - ``means``: 中心座標``(N, 3)``
            - ``quats``: 回転を表すクォータニオン``(N, 4)``
            - ``scales``: 指数関数を適用した各軸のスケール``(N, 3)``
            - ``opacities``: シグモイド関数を適用した不透明度``(N,)``
            - ``colors``: 0次の球面調和関数から変換したRGB値``(N, 3)``
    """

    plydata = PlyData.read(path)
    vertics = plydata["vertex"]

    # 位置
    means_np = np.vstack([vertics["x"], vertics["y"], vertics["z"]]).T
    means = torch.from_numpy(means_np).float()  # (N, 3)

    # クォータニオン (w,x,y,z) gsplat規約 or Warp (x, y, z, w)
    if is_quats_x_start:
        quats_np = np.vstack([vertics["rot_1"], vertics["rot_2"], vertics["rot_3"], vertics["rot_0"]]).T
    else:
        quats_np = np.vstack([vertics["rot_0"], vertics["rot_1"], vertics["rot_2"], vertics["rot_3"]]).T
    quats = torch.from_numpy(quats_np).float()  # (N, 4)

    # スケール
    scales_np = np.vstack([vertics["scale_0"], vertics["scale_1"], vertics["scale_2"]]).T
    scales_np = np.exp(scales_np)  # log -> expで実数へ変換
    scales = torch.from_numpy(scales_np).float()  # (N, 3)

    # 不透明度
    opacities_np = np.array([vertics["opacity"]]).T
    opacities_np = opacities_np[:, 0]  # (N, )
    opacities = torch.from_numpy(opacities_np).float()
    opacities = torch.sigmoid(opacities)  # logit -> sigmoidで(0~1)へ変換

    # 色
    colors_np = np.vstack([vertics["f_dc_0"], vertics["f_dc_1"], vertics["f_dc_2"]]).T
    colors = torch.from_numpy(colors_np).float()
    SH_C0 = 0.28209479177387814
    colors = torch.clip(colors * SH_C0 + 0.5, min=0.0, max=1.0)  # SH_DC -> RGB

    vertics_dict = {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors}
    return vertics_dict
