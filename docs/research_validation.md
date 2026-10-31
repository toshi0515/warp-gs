# Phase 1 — 独立参照数式とForward／Backward検証

2026-10-30実施。検査した条件では、事前に固定した許容差内で独立参照とWarpが一致した。
全テストは **57 passed、1 xfailed、skipなし**。Phase 1で追加した内訳はCPU 10件、CUDA 29件。
これは対象ケース・一次勾配の検証であり、全入力の数学的正しさや近似の精度を証明するものではない。
既存レンダリング・Backward・学習・viewer・Phase 0テストとbaselineには変更を加えていない。
不一致の修正、設定外部化、Phase 2以降は実施していない。

## 追加した構成と独立性

| ファイル | 役割 |
| --- | --- |
| `src/warpgs/reference_math.py` | CPU float64を基本とする参照数式。Torchのみをimport |
| `tests/warp_diagnostics.py` | 本番Warp関数・前処理・合成カーネルを直接呼ぶテスト専用adapter |
| `tests/math_validation.py` | 固定入力、同一cotangent、分岐監査、有限差分、測定出力 |
| `tests/test_reference_math.py` | 手計算・解析的性質・CPU中央差分 |
| `tests/test_warp_math.py` | 中間量、段階別Backward、公開API、境界挙動、CUDA中央差分 |
| `docs/phase1_measurements.json` | 174測定レコード、環境、検査コードhash、Phase 0成果物hash |

参照実装は本番の数式関数・定数・出力を利用しない。
回転はwxyz quaternionで基底ベクトルを回転して構成する。
通常の行列式・逆行列には`torch.linalg`、二次形式には行列とeinsumを使い、
本番の成分ごとの式をそのまま複製しない。
参照合成は全画素に対して順次計算し、tile/scan/radix sortを持たない。
診断adapterは本番関数を呼ぶだけで、投影や合成の式を再実装しない。

`covariance_3d`は物理scaleとraw wxyz quaternionから3D共分散を返す。
`project_pinhole`は純粋な一次投影、`project_legacy`は現行設定を適用した比較用投影。
両者は`Projection(mean, jacobian, covariance, depth, inverse)`を返し、純粋な投影のinverseはNone。
`render_legacy_reference`は物理入力から`ReferenceImage(image, transmittance, branches)`を返す。
`composite_reference`は投影済みの平均・conic・opacity・RGBと既定の順序を入力し、合成だけを検査できる。
これらは研究用の追加APIであり、既存の`WarpRenderer`／`RenderOutput`からは呼ばれない。

## 数式、単位、前提

3D位置・scaleはworld単位、scaleは標準偏差。quaternionはwxyzで非ゼロ、scaleは正。
カメラはrigidなworld-to-camera変換、Kはskewなし、投影対象のcamera zは正とする。

```text
qhat = q / ||q||
R = rotation(qhat)
Σworld = R diag(s²) Rᵀ
t = Rcamera μ + translation
μpixel = (fx tx/tz + cx, fy ty/tz + cy)
J = [[fx/tz, 0, -fx tx/tz²], [0, fy/tz, -fy ty/tz²]]
Σpixel = J Rcamera Σworld Rcameraᵀ Jᵀ
```

画面平均はpixel、Jはpixel/world、共分散はpixel²、逆共分散はpixel⁻²。
これは非線形投影されたGaussianの厳密なモーメントではなく、一次局所近似の式。
`project_pinhole`にはFOV clamp・低域通過正則化・determinant floorを入れない。

現行設定の比較経路では、Jに使うtx/tzとty/tzをそれぞれ±0.65W/fx、±0.65H/fyに制限し、
Σpixelの対角へ0.3 pixel²を加算する。平均の投影にはこのclampを使わない。
determinantが1e-6 pixel⁴未満の場合は、逆行列にdet/max(det,1e-6)を掛けた現行のscaled adjugateに合わせる。
テストした正定値・良条件の共分散ではdeterminant floorは非活性。

```text
power = (pixel_center - μpixel)ᵀ Σpixel⁻¹ (pixel_center - μpixel)
α = min(0.99, opacity exp(-power/2))
Cnext = C + T α color
Tnext = T (1-α)
image = Cfinal + Tfinal background
```

ピクセル中心は(x+0.5,y+0.5)。powerが[0,3.33²]内、α>=1/255の寄与を合成する。
合成後のTが1e-4未満になれば終了する。α・T・RGBは無次元。
これはscreen-space opacityの合成であって、3D密度のray積分ではない。
render参照はz>0.1を残し、float32で量子化した深度キーで並べる。
比較入力のキーは一意でuint32範囲内。同一キー・overflowは参照側で正常仕様を捏造せず拒否する。

## 検証したケース

CPUでは、単位回転と90度回転の対角共分散、対称性・正定値性・固有値、qと-q、qの倍率不変性を検査。
光軸上の手計算、非対称焦点距離・主点とカメラ回転・並進、clampしたJ、1個/2個のα合成も確認した。
別プロセスで参照moduleをimportし、Warp/gsplatがimportされず、Torch CUDAも初期化されないことを確認した。

CUDAでは中心・off-axis・rigid camera・FOV clamp両側について、3D共分散、画面平均、画面共分散対角、
逆共分散conic、深度を比較。前処理Backwardの画面平均経路と逆共分散経路を別々のseedで検査した。
合成だけのimage/Tと、画面平均・conic・opacity・RGB勾配も独立Autogradと比較した。
公開APIのForwardはPhase 0の8入力を再利用するが、期待値は保存画像ではなく独立参照から計算する。
黒/非黒背景、全点カリング、画面外支持の交差、複数タイル、端数画像を含む。

境界は微分テストと分けて、支持距離3.32/3.34、opacity 0.0038/0.004、0.98/0.999、
near深度0.099/0.101を比較した。早期終了は4層のopacity 0.95と0.96で、最後の寄与indexが
それぞれ3と2となること、背景項と残余Tが一致することを確認した。
閾値と完全に一致する点の導関数は検査しない。

## 有限差分と誤差の判定

勾配ケースは12×12、Gaussian 1個/2個、位置・異方scale・回転・RGBが非対称な固定入力。
全生パラメータ14成分/28成分を検査する。位置、quaternion、log-scale、opacity logit、RGB logitを含む。
入力をfloat32に丸めてからCPU float64へ昇格し、入力精度と演算精度を区別した。
中間量比較では物理scale等も同じfloat32値を昇格させる。
公開API比較では同じ生パラメータから各実装でexp/sigmoidを計算し、その連鎖も検証する。

損失は中央2×2×RGBに1〜12の非対称重みを付けた平均。
重みはCPU float32で1回定義してから各dtype/deviceへ渡し、参照・Backward・FDのcotangentを一致させた。
CUDA FDの差分を取る前の損失集計はfloat64で行い、float32画像の和による余分な桁落ちを避ける。
Forward自体は現行float32のまま。差分の分母には実際に表現されたplus/minus入力値の差を使う。

| 比較 | atol | rtol |
| --- | ---: | ---: |
| CPU手計算・解析的性質 | 1e-10 | 1e-10 |
| CPU Autograd / 中央差分 | 1e-7 | 1e-5 |
| Warp中間量・画像 / CPU参照 | 1e-6 | 1e-5 |
| Warp Backward / CPU Autograd | 1e-5 | 5e-4 |
| Warp Backward / float32中央差分 | 2e-4 | 2e-3 |

承認計画の判定値をそのまま使用し、実測に合わせた拡大はしていない。
要素ごとに`abs(actual-reference) <= atol + rtol*abs(reference)`で検査する。
相対誤差の集計は参照絶対値>atolの成分のみとし、ゼロ近傍の成分数も記録する。

刻み幅は`max(1,abs(raw_parameter))`に対し、CPU=[1e-4,1e-5,1e-6]、
Warp=[3e-3,1e-3,3e-4]。3段階すべての各成分の差分値とBackward値をJSONに保存した。
まずFD値同士だけで隣接ペアの安定性を判定し、最初の安定ペアの両方をBackwardと照合する。
全パラメータ群で両隣接ペアが安定だった。最小刻みも記録し、都合のよい結果だけを残さない。

各plus/minusで、損失対象画素の支持・cutoff・cap・alive・寄与mask、near選択、深度順、FOV clampを監査。
CUDA側では実際の前処理のtile rect/touched、Warp投影結果から算出したscreen分岐、
合成カーネルのlast contributorも監査する。いずれも基準入力から変化するとテストは失敗する。
損失領域外の画素はこの有限差分の分岐監査対象外であり、画像全体のForward比較で別途確認する。

## 実測結果

最終の174レコードを`docs/phase1_measurements.json`に保存した。
環境はPython 3.12.13、Torch 2.6.0+cu124、Warp 1.16.0、RTX 4050 Laptop GPU。
Phase 0と同じ通常CUDA環境を使用し、依存のインストール・更新は行っていない。

| 比較 | 観測した最大絶対差 |
| --- | ---: |
| 投影中間量（25比較） | 8.027865e-7 |
| 前処理Backward（2経路×3入力） | 2.518322e-7 |
| 合成単体の画像/T | 7.588376e-8 |
| 合成単体のBackward | 1.458304e-8 |
| 公開APIの画像（8シーン） | 2.545356e-7 |
| CPU Autograd / 判定対象の中央差分 | 5.116071e-9 |

| 生パラメータ群 | 公開Backward / CPU Autograd | 公開Backward / 判定対象CUDA FD |
| --- | ---: | ---: |
| means | 4.294052e-8 | 5.669892e-6 |
| quaternion | 7.961320e-9 | 1.257705e-5 |
| log-scale | 4.356515e-9 | 5.706446e-6 |
| opacity logit | 6.246023e-9 | 2.991408e-6 |
| RGB logit | 2.889155e-9 | 4.902482e-6 |

CUDA FDの最小刻みまで含めた全成分の最大絶対差は3.472331e-5。
小さい刻みで丸め誤差が目立つことも確認できる。許容差の計算方法や数式を変更してはいない。
画像8シーン・投影中間量5ケース・公開Backwardの2ケースは各5回反復し、
この測定での最大反復差はすべて0だった。小さな中央画素群の損失であり、
Phase 0の画像全体の和で観測された勾配変動を否定する結果ではない。

## 実行コマンドと成果物の保全

リポジトリルートの既存venvで実行する。

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_reference_math.py
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_warp_math.py
# 各比較と全刻み幅の値を標準出力に表示する最終検査
.\.venv\Scripts\python.exe -m pytest -q -s -ra
.\.venv\Scripts\ruff.exe check src/warpgs/reference_math.py tests/math_validation.py tests/warp_diagnostics.py tests/test_reference_math.py tests/test_warp_math.py
.\.venv\Scripts\ruff.exe format --check src/warpgs/reference_math.py tests/math_validation.py tests/warp_diagnostics.py tests/test_reference_math.py tests/test_warp_math.py
```

変更前の全体は18 passed/1 xfailed。CPU単独は10 passed、CUDA単独は29 passed。
最後の全体実行は57 passed/1 xfailed/2 warnings、13.91秒。警告は既存のdistutils非推奨警告。
Ruff checkとformatチェックも成功した。
最終実行はsubprocessで標準出力を取得し、`outputs/phase1/test_output.txt`へ保存した。
各行の`MEASUREMENT `以降のJSONを抽出し、環境・検査コードhashとともに測定JSONへ保存した。
通常のpytest実行はdocsの測定JSONやPhase 0baselineを更新しない。
CUDAテストはCUDAが使えない場合のみskipし、コンパイル・実行・数値比較の失敗は隠さない。

Phase 0成果物はPhase 1開始時と最終テスト前後のSHA256が一致した。

| ファイル | SHA256 |
| --- | --- |
| expected.npz | `540320947c56fbbd9d1170f3a593e4a8da534abef3b017f3fcc5f1755bee74e8` |
| metadata.json | `a765accfc981acfb5afb7d1059c64e63af973ab52482165a928c9e3fae669dc1` |
| scenes.json | `bcd5c2b3b14d30c7601c75485a545d86fe38c504bda9c18755ccf466b70014ae` |

## 制限と次の判断材料

- 今回の条件では許容差を超える不一致はなかった。全入力・全GPUでの保証ではない。
- 空点群のIndexErrorはPhase 0のstrict xfailを維持。望ましい正常動作としては扱わない。
- quaternionノルムfloorが活性になる極小ノルム、極端な悪条件共分散・determinant floor、
  同一深度キーやoverflow、閾値と完全一致する点、高階微分、別CUDA streamは未検証。
  純粋な参照quaternion正規化は非ゼロを仮定し、本番の極小ノルムclampを模倣しない。
- 相対誤差はゼロ近傍では評価しない。各群の全成分を照合したが、極小勾配の相対精度保証はしない。
- 有限差分は1タイルの小シーン。複数タイルと端数画像はForward比較で覆うが、
  大規模シーンの勾配集計や性能・メモリ挙動は対象外。
- 透視投影の一次近似誤差、Monte Carlo非線形push-forward、ray積分との関係、
  モデル再学習・MCMC・viewerは検証していない。

この結果は、既定の数値動作を守りながらPhase 2の設定外部化を検討するための基盤となる。
本作業はPhase 1で停止する。コミット・pushは行わない。
