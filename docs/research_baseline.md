# Phase 0 — 現行動作の記録と回帰検証

このGolden Testは**現行動作の回帰検証であり、数学的な正しさの検証ではない**。
現行レンダラーの出力を保存しているため、未知の数式・実装上の誤りも含みうる。
独立実装との同一数式の一致、近似と数学的参照の差、再学習後の差は別々に検証する。
Phase 0では `src/warpgs/` と `viewers/` を変更していない。

## 現在の構造と数式上の選択

- `src/warpgs/model.py`: means、wxyz quaternion、log-scale、opacity logit、RGB logitを保持。
  描画時はscaleをexp、opacityとRGBをsigmoidで変換する。色は視線に依存しないRGB。
- `src/warpgs/renderers.py`: `WarpRenderer` / `GsplatRenderer` が
  `RenderOutput(image, alpha)` を返す。imageはH×W×3。WarpのalphaはNone。
- `src/warpgs/my_warp_rasterizer.py`: PyTorchのcustom autograd bridgeからWarpへ接続。
  カメラ変換、カリング、投影、16×16タイル割り当て、exclusive scan、複製、radix sort、α合成。
- `src/warpgs/train.py`: NeRF SyntheticのRGBAを白背景で合成し、カメラを正深度の座標系へ変換。
  ランダム視点→描画→L1/SSIMとopacity/scale正則化→backward→パラメータ別Adam→
  quaternion正規化→gsplat MCMCの順。Warp学習でもgsplat拡張を検査する。
  100ステップごとにtest split全画像のPSNRを平均する。seed指定はない。
- `densification.py` は存在せず、MCMCはtrainから直接使用。Default戦略のadapterはない。
  checkpointはモデルパラメータのみで、optimizer・乱数状態を含む厳密な学習再開形式ではない。
- `viewers/01`〜`05` は独立した経路。特にviewer側のquaternionやカメラ表現を
  学習レンダラーと同一と仮定しない。統合・表示検証は今回の対象外。

基本式は `Σ3D = R diag(scale²) Rᵀ`、`μpixel = (fx x/z+cx, fy y/z+cy)`、
`Σ2D = J Rcamera Σ3D Rcameraᵀ Jᵀ`。
3Dの位置・scale・深度はシーン座標の単位、画面平均はpixel、画面共分散はpixel²。
α・透過率・RGBは無次元。Gaussianは画面上の未正規化opacity kernelであり、3D密度のray積分ではない。

| 数式・近似・境界の選択 | 現状 |
| --- | --- |
| quaternion | ノルム二乗を1e-12で下限化して正規化し、wxyz→xyzw |
| Jacobian | x/z、y/zを画像サイズと焦点距離による範囲の1.3倍でclamp |
| 低域通過正則化 | 画面共分散の対角に0.3 pixel²加算 |
| 逆共分散 | determinantの下限1e-6（画面共分散の単位ではpixel⁴） |
| 支持範囲 | Mahalanobis距離の二乗が0以上3.33²以下 |
| α | min(0.99, opacity×exp(-power/2))、1/255未満を除外 |
| 近接面 | Gaussian中心のカメラ深度z<=0.1を除外。分布の切断積分ではない |
| 合成 | ピクセル中心(px+0.5, py+0.5)で手前から合成し、残りのT×背景を加算 |
| 早期終了 | 寄与を加算しTを更新した後、T<1e-4なら終了 |

16×16タイル、scan・複製・range構築、Torch/Warpのメモリ・stream接続は実装上の構造。
一方、深度キーは `uint32(z*10000)` で量子化されるため、ソートは数値的な挙動にも関係する。
同一キーの順序や非常に大きな深度の整数変換は未検証。
CPU側へのitem()同期、呼び出しごとの配列確保もあるが、性能測定・最適化は行っていない。

## Backwardの対応表

手書きの逆向きα合成が画面平均・逆共分散・opacity・RGBへの勾配を計算する。
前処理だけを記録したWarp Tapeへ画面平均・逆共分散の勾配を渡し、3D入力へ戻す。
画面平均も共分散もWarp側で計算する。PyTorch側の投影経路ではない。

| 入力 | 現在の経路 |
| --- | --- |
| means | 画面平均と共分散の両方→Tape→custom backward |
| raw quaternion | 正規化と共分散→Tape→custom backward |
| log-scale | 共分散→Tape→scaleへの勾配→PyTorch exp |
| opacity logit | 手書き合成backward→PyTorch sigmoid |
| RGB logit | 手書き合成backward→PyTorch sigmoid |
| w2c、K、背景 | custom backwardはNoneを返す。requires_grad=Trueで実測確認 |
| カリング・タイル・ソート順 | 離散的な選択の変化を微分する経路なし |

高階微分、有限差分との一致は検証していない。
等方Gaussianでは回転勾配がゼロでも自然なので、回転した異方Gaussianで非ゼロ勾配も検査する。

## 固定入力・成果物・更新方法

`tests/baselines/phase0/scenes.json` に8シーンのモデル生パラメータを明記した。
centered、rotated_anisotropic、overlap、near_culled、offscreen_culled、
offscreen_intersection、nonblack、partial_tilesを含む。
float32、seed=0、単位w2c、fx=fy=30、cx=cy=16、基本32×32、端数画像19×23。
背景は黒または[0.2,0.3,0.4]。入力は乱数で生成しない。
損失は `sum(image * linspace(0.1,1,H*W*3).reshape(H,W,3))`。
これは学習性能の指標ではなく、固定した勾配の観測方法である。

- `expected.npz`: 画像と5種類の生パラメータ勾配、計48配列。最初のプロセスの最初の結果を保存。
- `metadata.json`: GPU/software、commit、ソース・検査コード・入力・出力のSHA256、
  各配列の参照最大値、30回分の差、個別のatol/rtolと算出根拠。
  Python/JSONのテキストハッシュはCRLFをLFへ正規化し、NPZはバイト列をそのままhashする。
  gsplatはバージョンを記録するのみで、Warp記録・検査で初期化しない。

ルートから既存環境を使用する。依存関係の追加・更新は不要。

```powershell
# 通常の回帰検証（保存ファイルを更新しない）
.\.venv\Scripts\python.exe -m pytest -q -ra
# CPUで実行できる検査。GPUテストは収集後に除外する
.\.venv\Scripts\python.exe -m pytest -q -ra -m "not cuda"
# 初回の明示的記録。既存ファイルがあれば拒否する
.\.venv\Scripts\python.exe tests/test_warp_baseline.py --record
# レビューする意図で明示的に再記録する場合のみ
.\.venv\Scripts\python.exe tests/test_warp_baseline.py --record --overwrite
# 静的チェック
.\.venv\Scripts\ruff.exe check tests/test_warp_baseline.py tests/test_renderers.py
.\.venv\Scripts\ruff.exe format --check tests/test_warp_baseline.py tests/test_renderers.py
```

更新時は旧baselineと新baselineの入力・ソース・環境・画像・勾配・許容差をレビューする。
テストが失敗したという理由だけで再記録したり、許容差だけを拡大したりしない。
まず再現条件、境界の変化、数値差、ソース変更を調査する。

## 許容差の根拠と実測結果（2026-10-30）

独立した3プロセスを順番に実行し、各プロセス10反復。競合するGPU作業を記録処理自身では起動しない。
配列ごとに最初の参照との最大絶対差D、参照の最大絶対値Mを測定した。
`atol=max(4D, 8*float32_epsilon*max(1,M))`、`rtol=0`。
4Dは実測した加算順序等の変動に対する余裕、8 epsilonはfloat32丸めに対する下限であり、
数式誤差の許容値ではない。GPUの厳密な決定性や別環境での誤差上限を保証しない。
以下は種類別の要約で、実際の検査は48配列それぞれのmetadata値を使う。

| 配列 | 観測最大絶対差（全シーンの最大） | 個別atolの範囲 |
| --- | ---: | ---: |
| image | 0 | 9.536743e-7 |
| grad_means | 1.251698e-5 | 9.536743e-7〜5.341737e-5 |
| grad_quats | 1.144409e-5 | 9.536743e-7〜4.577637e-5 |
| grad_scales（log-scale） | 7.629395e-6 | 9.536743e-7〜3.051758e-5 |
| grad_opacities（logit） | 1.668930e-6 | 9.536743e-7〜6.675720e-6 |
| grad_colors（logit） | 1.072884e-6 | 9.536743e-7〜4.291534e-6 |

近接面・画面外で全点が除外されるケースは、許容差比較に加えて背景との完全一致・全勾配ゼロも検査する。
支持範囲・カリング等のケースは挙動検査として扱う。有限差分を導入する際は、
摂動の前後でカリング・支持範囲・α閾値/cap・ソート・早期終了が変わらないことを別途確認する。

記録環境: Windows 11、Python 3.12.13、Torch 2.6.0+cu124、Warp 1.16.0、
NumPy 2.4.4、gsplat 1.5.3、pytest 9.1.1、RTX 4050 Laptop GPU（6141 MiB、sm_89）、driver 610.88。
nvccは12.4.99、Warp初期化表示のToolkitは12.9、Driverは13.3。表示は別々に扱う。

| 検証 | 結果 |
| --- | --- |
| 変更前の既存テスト（通常環境、調査時） | 7 passed、2 warnings |
| 最終版の全テスト | 18 passed、1 xfailed、skipなし、2 warnings、7.50秒 |
| CPU対象のみ | 7 passed、12 deselected、2 warnings、5.29秒 |
| Ruff check / format --check | 成功 |
| --recordのみで既存baselineへの記録 | 拒否。3成果物のhash不変 |
| 最終全テスト前後の3成果物のSHA256 | すべて不変 |

通常テストの非更新性は、上記の全テストを次のPowerShell処理で囲んで確認した。

```powershell
$baselinePaths = @('tests/baselines/phase0/scenes.json', 'tests/baselines/phase0/expected.npz', 'tests/baselines/phase0/metadata.json')
$baselineBefore = @($baselinePaths | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash })
.\.venv\Scripts\python.exe -m pytest -q -ra
$testExit = $LASTEXITCODE
$baselineAfter = @($baselinePaths | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash })
if (Compare-Object $baselineBefore $baselineAfter) { throw 'Baseline changed during normal tests' }
Write-Output 'PASS: all baseline hashes unchanged during normal tests'
exit $testExit
```

警告2件はjapanize_matplotlib経由のdistutils非推奨警告。
最初のsandbox内記録はNVRTCの一時ディレクトリアクセス制限により失敗した。
通常環境で明示的に再実行して記録・GPU検査を完了した。依存のインストール・更新はしていない。
最終記録前にはWindowsのcore.autocrlf=trueに対応するテキストhash正規化を追加し、
`--record --overwrite`で再測定した。許容差不足による失敗への対処ではない。

## 既知の制限とPhase 1への課題

- 空点群はoffsets[-1]参照でIndexErrorになる**既知の不具合**。維持すべき正常仕様ではない。
  望ましい背景画像の返却をテストに記述し、IndexError限定のstrict xfailとした。
  修正後はXPASSを失敗として検出し、別途承認した変更でxfailを外す。
- CUDAが利用できなければGPUテストは理由付きskip。Warpが欠けていても同様。
  カーネル実行・コンパイルの失敗はskipで隠さない。
  gsplat初期化を意図的に失敗させた状態でWarpの既存smoke testを実行するため、
  gsplat初期化だけを理由にWarpテストがskipされることはない。
- 記録環境が異なる場合も描画・勾配・挙動検査を実行した後、Golden比較を理由付きskipする。
  CPUのファイル整合性検査は継続する。別環境を同条件の回帰合格と表示しない。
- 異なるGPU・ソフトウェア、CUDAなしの実機、viewer/OpenGL、学習収束、MCMCの実行は今回未検証。
  既存CPUテストはoptimizer配置のMCMC check_sanityまでを検査する。
- Phase 1ではCPUの独立共分散・Jacobian参照、設定を一致させたWarp比較、
  同じ順序・α合成式の独立forward参照、滑らかな入力での5パラメータ群の有限差分検査を提案する。
  gsplatを厳密な数学的正解とせず、非線形投影されたサンプル分布とray積分も混同しない。

Phase 0で停止する。Phase 1以降は未実装。
