# warpgs

個人的な趣味と勉強目的で作成したNeRF Synthetic 形式の画像データから、3D Gaussian Splatting を学習するプログラムと作成したplyファイルのビューアプログラムです。
学習は `src/warpgs/train.py` で行います。NVIDIA GPU と CUDA を利用できる環境が必要です。

---

## AIの利用

最初は3DGSビューアのみを作っていたので一部はAIで作成しました
- AIを利用した部分(plyファイルを生成するため)：train.py, model.py, dataset.py
- AIを利用せず完全に手書きの部分：viewers/内の各ビューア, viewer_base.py, my_warp_rasterizer.py

---

## 導入

Python 3.12 と [uv](https://docs.astral.sh/uv/) をインストールしてから、このリポジトリのルートで次を実行します。

```powershell
uv sync
```

--- 

## 1. ビューアについて

PLYファイル`assets/models/final.ply` はリポジトリに含まれています。そのため、ビューアを試すために学習を行う必要はありません。

すべてのビューアは、この PLY ファイルを読み込みます。リポジトリのルートで次のように実行してください。

| ファイル | 内容 | 実行コマンド |
| --- | --- | --- |
| `01_pointcloud_viewer.py` | 点群として表示 | `uv run python viewers/01_pointcloud_viewer.py` |
| `02_torch_viewer.py` | PyTorch で計算するガウシアン描画 | `uv run python viewers/02_torch_viewer.py` |
| `03_warp_interop_viewer.py` | Warp と OpenGL の相互運用を使う描画 | `uv run python viewers/03_warp_interop_viewer.py` |
| `04_warp_optimized_viewer.py` | Warp による最適化版の描画 | `uv run python viewers/04_warp_optimized_viewer.py` |
| `05_warp_tile_viewer.py` | Warp のタイルベース描画 | `uv run python viewers/05_warp_tile_viewer.py` |

ウィンドウを閉じるには `Esc` キーを押します。カメラは `W`、`A`、`S`、`D` キーとマウスで操作できます。

---

## 2. train.py の使い方

### データセットのダウンロード

3DGS学習には公式 NeRF Synthetic データセットが必要です。

[NeRF公式データセット（Google Drive）](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)

1. `nerf_example_data.zip` をダウンロードする
2. リポジトリルートの `data/nerf_example_data/` に展開する
3. 次の配置になっていることを確認する

```text
data/nerf_example_data/nerf_synthetic/lego/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/
├── val/
└── test/
```

---

### Known issue: Windows環境でのgsplatのCUDA Extensionが使えない

以下はCodexによって修正してもらいました。

以下のコマンドで修正したgsplat の CUDA 拡張をビルドします。

```powershell
.\scripts\setup_gsplat_windows.ps1
```

---

### train.pyの実行方法

`src/warpgs/train.py` は、NeRF Synthetic 形式の画像データから PLY ファイルを作成します。`lego` データセットを使う場合は、リポジトリのルートで次を実行してください。

```powershell
uv run train --data_dir data\nerf_example_data\nerf_synthetic\lego --output_dir outputs\warp-lego
```

学習には時間がかかります。まず動作だけを確認したい場合は、画像を小さくし、学習回数を減らします。

```powershell
uv run train --data_dir data\nerf_example_data\nerf_synthetic\lego --output_dir outputs\smoke-test --downscale 4 --n_iters 100
```

`--data_dir` には、`transforms_train.json` と `train` ディレクトリを含む NeRF Synthetic 形式のデータセットを指定します。

| オプション | 説明 | 既定値 |
| --- | --- | --- |
| `--data_dir PATH` | 学習用データセットの場所 | `data/nerf_example_data/nerf_synthetic/lego` |
| `--output_dir PATH` | 学習結果の保存先 | `outputs/warp-lego` |
| `--n_iters N` | 学習ステップ数 | `10000` |
| `--downscale N` | 画像の縮小率。大きいほど軽くなります | `1` |
| `--renderer {warp,gsplat}` | 使用するレンダラー | `warp` |
| `--num_init_gaussians N` | 学習開始時のガウシアン数 | `5000` |
| `--init_checkpoint PATH` | 既存のチェックポイントを初期値として使う | 指定なし |

gsplat レンダラーを使う場合は、次のように指定します。ガウシアンの増減には MCMC を使います。

```powershell
uv run train --data_dir data\nerf_example_data\nerf_synthetic\lego --output_dir outputs\gsplat-lego --renderer gsplat
```

指定した `--output_dir` には、次のファイルが保存されます。

- `images/step_*.png`: test split の固定視点の正解画像（左）と学習結果（右）の比較画像
- `ckpts/step_*.pth`: 学習途中のチェックポイント
- `final.ply`: 学習後のガウシアンを保存した PLY ファイル
- `training_curves.png`: 損失、test split 全画像の平均 PSNR（100ステップごと）、ガウシアン数の推移
