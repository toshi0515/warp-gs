"""
dataset.py: NeRF Synthetic データセットの読み込み

NeRF Synthetic (lego) のディレクトリ構成:
    lego/
    ├── transforms_train.json   ← カメラパラメータ + 画像パスのリスト
    ├── transforms_val.json
    ├── transforms_test.json
    └── train/
        ├── r_0.png             ← RGBA画像 (800x800)
        ├── r_1.png
        └── ...

transforms_*.json の構造:
    {
        "camera_angle_x": 0.6911112070083618,  ← 水平方向のFOV (ラジアン)
        "frames": [
        {
            "file_path": "./train/r_0",
            "transform_matrix": [[...]]        ← C2W行列 (4x4)
        },
        ...
        ]
    }
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_nerf_synthetic(data_dir: str | Path, split: str = "train", downscale: int = 1) -> tuple:
    """NeRF Synthetic データセットを読み込む

    Args:
        data_dir (str | Path): lego/ ディレクトリのパス
        split (str, optional): "train" / "val" / "test". Defaults to "train".
        downscale (int, optional): 画像の縮小倍率 (1=原寸 800x800, 2=400x400, 4=200x200). Defaults to 1.

    Returns:
        tuple:
            images : (N, H, W, 3) float32 Tensor  [0, 1]  白背景でRGBA合成済み
            c2ws   : (N, 4, 4)    float32 Tensor  Camera-to-World 行列
            K      : (3, 3)       float32 Tensor  カメラ内部パラメータ

    """
    data_dir = Path(data_dir)
    json_path = data_dir / f"transforms_{split}.json"

    with open(json_path) as f:
        meta = json.load(f)

    images_list = []
    c2ws_list = []

    for frame in meta["frames"]:
        # 画像の読み込み (RGBA PNG)
        img_path = data_dir / (frame["file_path"].lstrip("./") + ".png")
        img = Image.open(img_path).convert("RGBA")

        # ダウンスケール
        if downscale > 1:
            W_orig, H_orig = img.size
            img = img.resize((W_orig // downscale, H_orig // downscale), Image.Resampling.LANCZOS)

        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4)

        # RGBA → RGB: 白背景(1.0)と合成
        # RGB_final = RGB * alpha + 1.0 * (1 - alpha)
        alpha = img_np[..., 3:4]
        rgb = img_np[..., :3]
        img_np = rgb * alpha + (1.0 - alpha)  # (H, W, 3)

        images_list.append(img_np)

        # JSONの行列を取得
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # OpenGL (NeRF) → OpenCV (gsplat) の座標系変換
        # カメラローカルの Y軸 と Z軸 の向きを反転させる (-1を掛ける)
        c2w[:, 1:3] *= -1
        c2ws_list.append(c2w)

    images = torch.from_numpy(np.stack(images_list))  # (N, H, W, 3)
    c2ws = torch.from_numpy(np.stack(c2ws_list))  # (N, 4, 4)

    # カメラ内部パラメータ K の計算
    H, W = images.shape[1], images.shape[2]
    # camera_angle_x = 水平方向のFOV (ラジアン)
    # fx = W / (2 * tan(FOV/2))
    focal = W / (2.0 * math.tan(meta["camera_angle_x"] / 2.0))
    K = torch.tensor([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    print(f"[dataset] split={split}, N={len(images)}, H={H}, W={W}, focal={focal:.2f}")

    return images, c2ws, K


class NeRFSyntheticDataloader:
    """
    NeRF Synthetic データローダー
    毎ステップランダムに1枚の画像とカメラパラメータを返す
    """

    def __init__(self, data_dir: str, split: str = "train", downscale: int = 1, device: str = "cuda"):
        images, c2ws, K = load_nerf_synthetic(data_dir, split, downscale)
        self.images = images.to(device)  # (N, H, W, 3)
        self.c2ws = c2ws.to(device)  # (N, 4, 4)
        self.K = K.to(device)  # (3, 3)
        self.N = len(images)
        self.H, self.W = images.shape[1], images.shape[2]

    def get_random(self) -> tuple:
        """ランダムに1枚サンプリングして (image, w2c, K) を返す"""
        idx = torch.randint(0, self.N, (1,)).item()
        image = self.images[idx]  # (H, W, 3)
        c2w = self.c2ws[idx]  # (4, 4)

        # gsplat は W2C (world-to-camera) を要求するので C2W の逆行列を計算
        # C2W = [R | t] → W2C = [R^T | -R^T t]
        w2c = torch.linalg.inv(c2w)  # (4, 4)
        return image, w2c, self.K

    def get_by_index(self, idx: int) -> tuple:
        """インデックス指定で1枚取得 (評価用)"""
        image = self.images[idx]
        w2c = torch.linalg.inv(self.c2ws[idx])
        return image, w2c, self.K
