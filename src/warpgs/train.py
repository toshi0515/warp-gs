import argparse
import time
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from warpgs.dataset import NeRFSyntheticDataloader
from warpgs.densification import Densification
from warpgs.model import GaussianModel, create_optimizers
from warpgs.ply import save_ply
from warpgs.renderers import create_renderer


# 損失関数
def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """
    1 - SSIM 損失 (構造的類似度)

    pred, target: (H, W, 3) または (1, 3, H, W)
    """
    # (H, W, 3) → (1, 3, H, W)
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)

    C1, C2 = 0.01**2, 0.03**2
    kernel = torch.ones(3, 1, window_size, window_size, device=pred.device) / (window_size**2)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=3)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred**2, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target**2, kernel, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.2) -> torch.Tensor:
    """
    L1 + λ·SSIM 損失 (論文の設定: λ=0.2)

    L1  : ピクセルごとの絶対誤差 → 大まかな色を素早く合わせる
    SSIM: 局所的な構造の類似度   → エッジ・テクスチャを合わせる
    """
    l1 = F.l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    return (1.0 - lam) * l1 + lam * ssim


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR (dB) を計算する (高いほど良い30dB以上が目標)"""
    with torch.no_grad():
        mse = F.mse_loss(pred, target)
        psnr = -10.0 * torch.log10(mse + 1e-8)
    return psnr.item()


# メイン学習ループ
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)  # 白背景

    # データロード
    train_loader = NeRFSyntheticDataloader(args.data_dir, split="train", downscale=args.downscale, device=device)
    H, W = train_loader.H, train_loader.W
    print(f"[train] image size: {H}x{W}, device: {device}")

    # モデル初期化
    if args.init_checkpoint:
        model = GaussianModel.load(args.init_checkpoint, device=device)
    else:
        model = GaussianModel.from_random(
            num_gaussians=args.num_init_gaussians,
            scene_radius=1.5,
            init_scale=0.05,
            device=device,
        )
    optimizers = create_optimizers(model)
    controller = Densification(args.renderer, args.densification, args)
    renderer = create_renderer(
        args.renderer,
        packed=args.packed,
        absgrad=args.absgrad and args.renderer == "gsplat" and args.densification == "default",
    )
    controller.check_sanity(model, optimizers)

    print(f"[train] renderer={renderer.name}, densification={controller.name}, initial Gaussians={model.num_gaussians()}")
    # 出力ディレクトリ
    out_dir = Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    # 学習ループ
    loss_history = []
    psnr_history = []
    n_gauss_history = []
    start_time = time.time()

    for step in range(1, args.n_iters + 1):
        model.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        # ランダムに1枚サンプリング
        target_img, w2c, K = train_loader.get_random()

        render_output = renderer.render(model, w2c, K, H, W, bg_color)
        pred_img = render_output.image
        controller.before_backward(model, optimizers, step, render_output)

        loss = combined_loss(pred_img, target_img) + controller.regularization(model)

        # バックワードパス
        loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()

        model.normalize_quats()
        # gsplat 1.5.3 の公式 trainer と同じく optimizer 更新後に呼ぶ。
        controller.after_backward(model, optimizers, step, render_output)

        # ログ
        loss_val = loss.item()
        loss_history.append(loss_val)
        n_gauss_history.append(model.num_gaussians())

        if step % 100 == 0:
            psnr = compute_psnr(pred_img.detach(), target_img)
            psnr_history.append(psnr)
            elapsed = time.time() - start_time
            print(
                f"Step {step:5d}/{args.n_iters} | "
                f"Loss: {loss_val:.5f} | "
                f"PSNR: {psnr:.2f} dB | "
                f"N: {model.num_gaussians():6d} | "
                f"Time: {elapsed:.0f}s"
            )

        # 定期的な画像保存
        if step % args.save_img_interval == 0 or step == args.n_iters:
            model.eval()
            with torch.no_grad():
                # テスト用に固定カメラ (index 0) でレンダリング
                test_img, test_w2c, test_K = train_loader.get_by_index(0)
                pred_test = renderer.render(model, test_w2c, test_K, H, W, bg_color).image
                pred_np = pred_test.clamp(0, 1).cpu().numpy()
                target_np = test_img.cpu().numpy()

            # 並べて保存
            compare = np.concatenate([target_np, pred_np], axis=1)
            compare = (compare * 255).clip(0, 255).astype(np.uint8)
            from PIL import Image

            Image.fromarray(compare).save(out_dir / "images" / f"step_{step:05d}.png")

        # チェックポイント保存
        if step % args.save_ckpt_interval == 0 or step == args.n_iters:
            model.save(str(out_dir / "ckpts" / f"step_{step:05d}.pth"))

    # 学習完了: 最終 PLY を保存
    save_ply(model, str(out_dir / "final.ply"))
    print(f"\n学習完了! 総時間: {(time.time() - start_time) / 60:.1f}分")
    print(f"最終ガウシアン数: {model.num_gaussians()}")

    # 学習曲線の保存
    import matplotlib

    matplotlib.use("Agg")  # GUI なし
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#1a1a2e")

    axes[0].plot(loss_history, color="#00bfff", linewidth=0.8)
    axes[0].set_title("Loss曲線", color="white")
    axes[0].set_yscale("log")
    axes[0].set_facecolor("#0d0d1a")
    axes[0].tick_params(colors="gray")

    axes[1].plot([i * 100 for i in range(len(psnr_history))], psnr_history, color="#00ff88", linewidth=1.5)
    axes[1].set_title("PSNR (dB)", color="white")
    axes[1].set_facecolor("#0d0d1a")
    axes[1].tick_params(colors="gray")

    axes[2].plot(n_gauss_history, color="#ffaa00", linewidth=0.8)
    axes[2].set_title("ガウシアン数", color="white")
    axes[2].set_facecolor("#0d0d1a")
    axes[2].tick_params(colors="gray")

    for ax in axes:
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    plt.tight_layout()
    plt.savefig(str(out_dir / "training_curves.png"), dpi=150, facecolor="#1a1a2e")
    print(f"学習曲線 → {out_dir}/training_curves.png")


# 引数パーサー
def parse_args():
    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / "data" / "nerf_example_data" / "nerf_synthetic" / "lego"
    output_path = root_dir / "outputs" / "warp-lego"
    p = argparse.ArgumentParser(description="3DGS training on NeRF Synthetic")
    p.add_argument("--data_dir", default=data_path, help="lego/ ディレクトリのパス")
    p.add_argument("--output_dir", default=output_path, help="出力ディレクトリ")
    p.add_argument("--downscale", type=int, default=1, help="画像縮小率 (1=800x800, 2=400x400)")
    p.add_argument("--n_iters", type=int, default=10_000, help="学習ステップ数 (本格は30000)")
    p.add_argument("--num_init_gaussians", type=int, default=5_000, help="初期ガウシアン数")
    p.add_argument("--init_checkpoint", type=Path, help="共通形式の初期 checkpoint")
    p.add_argument("--renderer", choices=("warp", "gsplat"), default="warp")
    p.add_argument(
        "--packed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="gsplat packed rasterization (Warp では無視)",
    )
    p.add_argument("--absgrad", action="store_true", help="gsplat DefaultStrategy で absolute gradient を使う")
    # Densification
    p.add_argument(
        "--densification",
        choices=("default", "mcmc"),
        default="mcmc",
        help="gsplat DefaultStrategy または MCMCStrategy",
    )
    p.add_argument("--densify_from", type=int, default=500)
    p.add_argument("--densify_until", type=int, default=7_500)
    p.add_argument("--densify_interval", type=int, default=100)
    p.add_argument("--grad_threshold", type=float, default=0.0002)
    p.add_argument("--opacity_threshold", type=float, default=0.005)
    p.add_argument("--max_gaussians", type=int, default=150_000)
    # 3DGS MCMC
    p.add_argument("--mcmc_noise_lr", type=float, default=5e5, help="Langevin position noise の係数")
    p.add_argument("--mcmc_noise_until", type=int, default=10_000, help="position noise を加える最終 step")
    p.add_argument("--opacity_reg", type=float, default=0.01, help="MCMC opacity 正則化係数")
    p.add_argument("--scale_reg", type=float, default=0.01, help="MCMC scale 正則化係数")
    # 保存
    p.add_argument("--save_img_interval", type=int, default=1000)
    p.add_argument("--save_ckpt_interval", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
