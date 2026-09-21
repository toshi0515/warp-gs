"""
model.py: GaussianModel と Densification ロジック
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class GaussianModel(nn.Module):
    """3D Gaussian Splatting のパラメータを管理するクラス

    保持するパラメータ:
            means         : (N, 3)  位置 (world座標)
            quats         : (N, 4)  クォータニオン (w,x,y,z) gsplat規約
            log_scales    : (N, 3)  スケールのlog (exp で実スケール)
            logit_opacities: (N,)   不透明度のlogit (sigmoid で 0-1)
            logit_colors  : (N, 3)  色のlogit (sigmoid で 0-1, RGB, SHなし)
    """

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        logit_colors: torch.Tensor,
    ):
        super().__init__()
        self.means = nn.Parameter(means.clone())
        self.quats = nn.Parameter(quats.clone())
        self.log_scales = nn.Parameter(log_scales.clone())
        self.logit_opacities = nn.Parameter(logit_opacities.clone())
        self.logit_colors = nn.Parameter(logit_colors.clone())

        # Densification 用の勾配累積バッファ
        # nn.Parameter ではないのでオプティマイザに登録されない
        self.register_buffer("grad_accum", torch.zeros(means.shape[0]))
        self.register_buffer("grad_count", torch.zeros(means.shape[0]))

    # プロパティ: 制約付きパラメータへのアクセス
    @property
    def scales(self) -> torch.Tensor:
        """(N, 3) 実際のスケール (常に正)"""
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        """(N,) 不透明度 [0, 1]"""
        return torch.sigmoid(self.logit_opacities)

    @property
    def colors(self) -> torch.Tensor:
        """(N, 3) RGB色 [0, 1]"""
        return torch.sigmoid(self.logit_colors)

    def num_gaussians(self) -> int:
        return self.means.shape[0]

    # オプティマイザ用のパラメータ
    def get_optimizer_params(
        self,
        lr_means: float = 1.6e-4,
        lr_quats: float = 1e-3,
        lr_scales: float = 5e-3,
        lr_opacities: float = 5e-2,
        lr_colors: float = 2.5e-3,
    ) -> list:
        """
        パラメータごとに異なる学習率を設定して返す

        位置(means)は最も小さい学習率:
            大きすぎるとガウシアンが飛んでいく
        不透明度(opacities)は最も大きい学習率:
            素早く収束させて不要なガウシアンを削除できるようにする
        """
        return [
            {"params": [self.means], "lr": lr_means, "name": "means"},
            {"params": [self.quats], "lr": lr_quats, "name": "quats"},
            {"params": [self.log_scales], "lr": lr_scales, "name": "scales"},
            {"params": [self.logit_opacities], "lr": lr_opacities, "name": "opacities"},
            {"params": [self.logit_colors], "lr": lr_colors, "name": "colors"},
        ]

    # 正規化と勾配累積
    def normalize_quats(self):
        """
        クォータニオンを正規化して長さ=1を保つ
        Adam の更新で長さが変わるため、毎ステップ必須
        """
        with torch.no_grad():
            self.quats.data = F.normalize(self.quats.data, dim=-1)

    def accumulate_gradients(self):
        """
        位置の勾配のL2ノルムを累積する (Densification の判定に使用)
        どのGaussianをもっと詳細に表現すべきかを示すシグナル
        """
        if self.means.grad is not None:
            with torch.no_grad():
                grad_norm = self.means.grad.norm(dim=-1)  # (N,)
                self.grad_accum += grad_norm
                self.grad_count += 1.0

    def get_avg_gradient(self) -> torch.Tensor:
        """平均勾配を返す (累積 ÷ カウント)"""
        avg = torch.zeros_like(self.grad_accum)
        mask = self.grad_count > 0
        avg[mask] = self.grad_accum[mask] / self.grad_count[mask]
        return avg

    def reset_gradient_accum(self):
        self.grad_accum.zero_()
        self.grad_count.zero_()

    # 初期化
    @classmethod
    def from_random(
        cls, num_gaussians: int = 5000, scene_radius: float = 1.5, init_scale: float = 0.05, device: str = "cuda"
    ) -> "GaussianModel":
        """
        ランダムな初期ガウシアン群を生成する

        lego シーンの場合:
            - 物体は原点付近の [-1, 1]^3 に収まる
            - カメラは半径 ~4 の球面上に配置されている
            - scene_radius=1.5 が適切な初期化範囲

        num_gaussians: 初期ガウシアン数 (5000~10000 が目安)
        init_scale   : 初期スケール (0.05 = 5cm 程度の小さな楕円体)
        """
        # 位置: [-scene_radius, scene_radius]^3 の一様分布
        means = (torch.rand(num_gaussians, 3) - 0.5) * 2 * scene_radius

        # クォータニオン: ランダムな向き (w,x,y,z)
        quats = F.normalize(torch.randn(num_gaussians, 4), dim=-1)

        # スケール: log スペースで保持
        # exp(log(init_scale)) = init_scale
        log_scales = torch.full((num_gaussians, 3), math.log(init_scale))

        # 不透明度: 最初は低め (logit(0.1) = -2.2)
        # 透明なガウシアンから学習を始めてだんだん不透明にする
        logit_opacities = torch.full((num_gaussians,), -2.2)

        # 色: 中間値から開始 (logit(0.5) = 0.0 → sigmoid(0)=0.5)
        logit_colors = torch.zeros(num_gaussians, 3)

        return cls(means, quats, log_scales, logit_opacities, logit_colors).to(device)

    # 保存 / 読み込み
    def save(self, path: str):
        torch.save(
            {
                "means": self.means.data,
                "quats": self.quats.data,
                "log_scales": self.log_scales.data,
                "logit_opacities": self.logit_opacities.data,
                "logit_colors": self.logit_colors.data,
            },
            path,
        )
        print(f"[model] saved {self.num_gaussians()} Gaussians → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "GaussianModel":
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["means"], ckpt["quats"], ckpt["log_scales"], ckpt["logit_opacities"], ckpt["logit_colors"]).to(device)
        print(f"[model] loaded {model.num_gaussians()} Gaussians ← {path}")
        return model


# Densification
def densify_and_prune(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    grad_threshold: float = 0.0002,
    opacity_threshold: float = 0.005,
    scale_threshold: float = 0.1,
    max_gaussians: int = 300_000,
) -> tuple[GaussianModel, torch.optim.Optimizer]:
    """Clone / Split / Prune を実行して新しい GaussianModel を返す

    Args:
        model (GaussianModel): 元のGaussianModel
        optimizer (torch.optim.Optimizer): _description_
        grad_threshold (float, optional): 勾配の平均がこれ以上 → Clone or Split 対象. Defaults to 0.0002.
        opacity_threshold (float, optional): 不透明度がこれ以下 → Prune (削除) 対象. Defaults to 0.005.
        scale_threshold (float, optional): max(scale) がこれ以上 → Split (大きすぎる), これ未満 → Clone (小さすぎる).
        Defaults to 0.1.
        max_gaussians (int, optional): この数を超えたら Split を抑制 (VRAM保護). Defaults to 300_000.

    Returns:
        tuple[GaussianModel, torch.optim.Optimizer]: 新しいGaussianModel, オプティマイザー
    """
    avg_grad = model.get_avg_gradient()  # (N,)
    opac = model.opacities.detach()  # (N,)
    scale_max = model.scales.detach().max(dim=-1).values  # (N,)

    # 各マスクの計算
    # Prune: 不透明度が低い → 画像に寄与していない
    mask_keep = opac > opacity_threshold  # (N,) True=残す

    # Clone: 小さいのに勾配が大きい → もっと細かく表現が必要
    mask_clone = (avg_grad > grad_threshold) & (scale_max < scale_threshold) & mask_keep

    # Split: 大きいのに勾配が大きい → 1個で広すぎる
    # max_gaussians 超えたら Split を止める (メモリ保護)
    can_split = model.num_gaussians() < max_gaussians
    mask_split = (avg_grad > grad_threshold) & (scale_max >= scale_threshold) & mask_keep & can_split

    n_prune = int((~mask_keep).sum())
    n_clone = int(mask_clone.sum())
    n_split = int(mask_split.sum())
    print(
        f"  Prune:{n_prune:5d}  Clone:{n_clone:5d}  Split:{n_split:5d}  "
        f"Total:{model.num_gaussians():6d} → "
        f"{model.num_gaussians() - n_prune + n_clone + n_split:6d}"
    )

    # 新しいパラメータリストを構築
    new = {
        "means": [model.means.data[mask_keep]],
        "quats": [model.quats.data[mask_keep]],
        "log_scales": [model.log_scales.data[mask_keep]],
        "logit_opacities": [model.logit_opacities.data[mask_keep]],
        "logit_colors": [model.logit_colors.data[mask_keep]],
    }

    # Clone: そのままコピーを1個追加
    if n_clone > 0:
        for key in new:
            # attr = getattr(model, key.replace("logit_", "logit_").replace("log_scales", "log_scales"))
            # 直接 data 属性を参照
            src = getattr(model, key).data[mask_clone]
            new[key].append(src)

    # Split: スケールを 0.7 倍にして2個に分割
    if n_split > 0:
        s_orig = model.scales.data[mask_split]  # (M, 3) 実スケール
        ls_new = model.log_scales.data[mask_split] + math.log(0.7)
        # 元の向きに沿って少しずらす
        offset = torch.randn_like(s_orig) * s_orig * 0.5

        new["means"].append(model.means.data[mask_split] + offset)
        new["means"].append(model.means.data[mask_split] - offset)

        for key in ["quats", "log_scales", "logit_opacities", "logit_colors"]:
            src = getattr(model, key).data[mask_split]
            val = ls_new if key == "log_scales" else src
            new[key].append(val)
            new[key].append(val)

    # 新しいモデルを作成
    new_model = GaussianModel(
        means=torch.cat(new["means"]),
        quats=torch.cat(new["quats"]),
        log_scales=torch.cat(new["log_scales"]),
        logit_opacities=torch.cat(new["logit_opacities"]),
        logit_colors=torch.cat(new["logit_colors"]),
    ).to(model.means.device)

    # オプティマイザを再作成 (テンソルサイズが変わるため)
    new_optimizer = torch.optim.Adam(new_model.get_optimizer_params())

    return new_model, new_optimizer
