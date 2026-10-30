# warpgs

個人的な趣味と勉強目的で作成した、NeRF Synthetic 形式の画像データから、3D Gaussian Splatting (3DGS)を学習するプログラムおよび作成したplyファイルのビューアプログラムのリポジトリです。
3DGS学習は `src/warpgs/train.py` で行います。NVIDIA GPU と CUDA を利用できる環境が必要です。
点群ビューア `01_pointcloud_viewer.py` は NVIDIA GPU と CUDA 環境がなくても実行できます。それ以外のビューアと学習には NVIDIA GPU が必要です。

---

## AIの利用

最初は3DGSビューアのみを作っていたので一部はAIで作成しました
- AIを利用した部分(plyファイルを生成するため)：train.py, model.py, dataset.py
- AIを利用せず完全に手書きの部分：viewers/内の各ビューア, viewer_base.py, my_warp_rasterizer.py

---

## 実行環境と導入手順

以下は **Windows x64 で全ビューアと学習を動かす**ための手順です。各ビューアの違いは次の通りです。

| プログラム | NVIDIA GPU / CUDA | 画面表示 |
| --- | --- | --- |
| `01_pointcloud_viewer.py` | 不要。PLY の読み込みは CPU で行う | OpenGL 4.3 対応のグラフィックス環境が必要 |
| `02_torch_viewer.py` | 必要 | OpenGL 4.3 対応環境が必要 |
| `03_warp_interop_viewer.py`、`04_warp_optimized_viewer.py`、`05_warp_tile_viewer.py` | 必要 | CUDA と OpenGL の相互運用が可能な NVIDIA GPU が必要 |
| `train`（`warp` / `gsplat`） | 必要 | ウィンドウ表示は不要 |

全プログラムを実行するには、次を用意してください。

1. [Git](https://git-scm.com/downloads) と [uv](https://docs.astral.sh/uv/getting-started/installation/)。Python は 3.12 を使います。
2. NVIDIA GPU と対応するドライバー。ビューアを表示する GPU は OpenGL 4.3 に対応している必要があります。`03`～`05` では CUDA と OpenGL が同じ GPU を使える構成にしてください。
3. [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)（`nvcc.exe` を使用）。このリポジトリの `uv.lock` は CUDA 12.4 版の PyTorch を指定しています。
4. [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) の **Desktop development with C++** ワークロード（MSVC v143 x64 と Windows SDK を含む）。Windows で gsplat の CUDA 拡張をビルドするために必要です。

PowerShell で次を実行します。`uv sync --locked` は `uv.lock` のバージョンを使用します。初回はパッケージのダウンロードが必要です。

```powershell
git clone https://github.com/toshi0515/warp-gs.git
cd warp-gs
uv python install 3.12
uv sync --python 3.12 --locked
```

全プログラムを使う場合は、CUDA とコンパイラーが利用できることを確認し、**学習の実行前に** gsplat の拡張を準備します。`train` は `--renderer warp` の場合も gsplat の拡張を検査します。

```powershell
nvcc --version
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\scripts\setup_gsplat_windows.ps1
```

PyTorch の出力で `torch.cuda.is_available()` が `True` になることを確認してください。セットアップスクリプトは Visual Studio の C++ 環境を探し、gsplat 1.5.3 の Windows 用ビルド処理を修正して CUDA 拡張をビルドし、描画と逆伝播を検証します。`nvcc` が見つからない場合は CUDA Toolkit の `bin` を `PATH` に追加して新しい PowerShell を開いてください。PowerShell の実行ポリシーでスクリプトがブロックされる場合は、`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_gsplat_windows.ps1` を使用できます。

`01_pointcloud_viewer.py` だけを試す場合、NVIDIA GPU、CUDA Toolkit、C++ Build Tools、上記の gsplat セットアップは不要です。ただし OpenGL 4.3 対応環境と `uv sync` による Python 依存関係の導入は必要です。

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

### train.pyの実行方法

`src/warpgs/train.py` は、NeRF Synthetic 形式の画像データから PLY ファイルを作成します。`lego` データセットを使う場合は、リポジトリのルートで次を実行してください。

```powershell
uv run train --data_dir data\nerf_example_data\nerf_synthetic\lego --output_dir outputs\warp-lego
```

学習には時間がかかります。まず動作だけを確認したい場合は、画像を小さくし、学習回数を減らします。

```powershell
uv run train --data_dir data\nerf_example_data\nerf_synthetic\lego --output_dir outputs\smoke-test --downscale 4 --n_iters 100
```

`--data_dir` には、`transforms_train.json`、`transforms_test.json`、`train`、`test` を含む NeRF Synthetic 形式のデータセットを指定します。学習中の評価でも test split を読み込みます。

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
