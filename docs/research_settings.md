# Phase 2 — 数値設定の外部化と既定動作の維持

2026-10-30実施。全テストは **101 passed、1 xfailed、skipなし**。
Phase 0のGolden TestとPhase 1の参照・有限差分検査を、既存の許容差で通過した。
新規44件はCPU設定検査33件、CUDA検査11件。
既存baseline・許容差・Phase 1の測定成果物は更新していない。
Golden Testは現行動作の回帰検証であり、数学的な正しさの検証ではない。

## Python API

```python
from dataclasses import asdict

from warpgs.raster_config import RasterConfig
from warpgs.renderers import WarpRenderer

default_renderer = WarpRenderer()  # 既存コードはそのまま使える
config = RasterConfig(covariance_lowpass=0.6)
experimental_renderer = WarpRenderer(config=config)
print(asdict(config))              # 要求した値
print(config.effective_values())   # Warpへ渡すfloat32値
# experimental_renderer.render(model, w2c, K, height, width, background)
```

`RasterConfig`はfrozen dataclassであり、Torch・Warp・gsplatをimportしない。
設定を変更するときは別のconfigを作る。必要なら`dataclasses.replace(config, alpha_cap=0.8)`を使う。
`WarpRenderer`はconfigを保持し、各Forwardが新しいWarp structを作る。
設定はモデルパラメータではなく、設定に対する勾配は返さない。
`render()`の引数、`RenderOutput(image, alpha)`、Warpの`alpha=None`は従来どおり。
`create_renderer()`と学習CLIは変更せず、学習は既定設定を使用する。
gsplat側にこれらのWarp設定は適用しない。

| フィールド | 要求する既定値 | 単位・意味 |
| --- | ---: | --- |
| gaussian_extent | 3.33 | 支持範囲のMahalanobis距離 |
| covariance_lowpass | 0.3 | 画面共分散対角への加算、pixel² |
| jacobian_clamp_factor | 1.3 | Jacobianに使う画角制限の係数 |
| alpha_cutoff | 1/255 | このα未満を除外 |
| alpha_cap | 0.99 | α上限 |
| transmittance_threshold | 1e-4 | 寄与加算・T更新後の早期終了閾値 |
| near_plane | 0.1 | 中心のcamera zがこの値以下なら除外、world単位 |

要求値とfloat32への変換後の値を両方検証する。全項目が有限な実数であることを要求し、
bool・文字列・Tensor等は受け付けない。
extent/clamp係数は正、lowpass/nearは非負、`0<=cutoff<=cap<1`かつcap>0、T閾値は[0,1]。
例えばcap=0.999999999はfloat32で1になるため拒否し、極小extentが0に丸められる指定も拒否する。
lowpass=0、cutoff=0、near=0は許可する。T閾値=0は早期終了を無効にする。
これらの指定は極端な入力の数値安定性まで保証するものではない。

既定値の実際のfloat32値は以下。全実験の要求値・実効値を測定JSONにも記録した。

```text
gaussian_extent          3.3299999237060547
covariance_lowpass       0.30000001192092896
jacobian_clamp_factor    1.2999999523162842
alpha_cutoff             0.003921568859368563
alpha_cap                0.9900000095367432
transmittance_threshold  0.00009999999747378752
near_plane               0.10000000149011612
```

## 設定の伝播と互換性

`raster_config.py`のCPU型を、`my_warp_rasterizer.py`の`RasterConfigWP`へ変換する。
7個のfloat32フィールドを実行時引数として渡す。設定別のカーネル生成やグローバル値の書き換えは行わない。
コード変更による初回JIT再コンパイルは発生するが、値ごとのspecializationは追加しない。

- 画面投影: Jacobian clamp、covariance lowpass。
- 前処理: nearとextent。extentはGaussianのタイルAABBにも反映。
- 合成Forward: extent、cutoff、cap、透過率閾値。
- 手書きBackward: 同じextent、cutoff、capを使用。終了範囲はForwardのlast contributorと最終Tを再利用。
  透過率閾値を別に再適用して合成範囲を決め直すことはしない。
- Tape: Forwardの設定で実行した前処理を記録。
- Autograd ctx: そのForwardのWarp structを保持し、Backwardへ渡す。

`WarpRasterizeFunction.apply`の10引数形式を維持し、任意の11番目にconfigを指定できる。
Backwardの返却数は実際の入力数に合わせ、カメラ・K・寸法・背景・設定の勾配はNoneとする。
別設定のForwardや`renderer.config`の差し替えを挟んでも、既存グラフは元の設定を使用する。
privateなWarp関数・カーネルの引数は変更したため、テスト専用adapterも追従した。
viewer側の独立したカーネルは変更していない。

純粋な参照`covariance_3d()`と`project_pinhole()`は変更なし。
legacy参照にはkeyword-only設定引数を追加したが、本番のRasterConfig・定数はimportせず、
独立した既定値を持つ。比較テストが同じ指定値を両者へ渡す。
非既定設定の比較には実効float32値をCPU float64へ昇格して使い、入力丸めと演算誤差を区別する。

固定のまま残した項目は、tile 16×16、determinant下限1e-6、quaternionノルム二乗下限1e-12、
深度キーの`uint32(z*10000)`。ソート・scan・複製・カメラ/モデル形式は変更していない。
既定時の式の演算順序、extentの二乗、分岐の不等号、合成後の早期終了位置も維持した。

## 設定ごとの効果と独立参照の比較

各専用入力で既定値との差を確認し、変更前後それぞれの画像と全5パラメータ群の生勾配を独立参照と照合した。

| 変更 | 確認した効果 | 既定画像からの最大絶対変化 |
| --- | --- | ---: |
| extent=2.5 | 支持範囲が縮小。tile rectが[0,0,2,1]→[0,0,1,1]、6 tiles→4 tiles | 1.842074e-2 |
| lowpass=0.6 | 投影共分散対角・conic・画像が変化 | 1.494893e-2 |
| clamp係数=1.0 | 新旧クランプ範囲の間で共分散・画像が変化 | 4.241437e-3 |
| cutoff=0.02 | opacity=0.01の寄与が除外される | 4.681885e-3 |
| cap=0.8 | 飽和し、opacity・位置勾配がゼロになる。既定値では非ゼロ | 4.300553e-2 |
| T閾値=0.01 | 4層合成の最後の寄与indexが3→1。残余Tも照合 | 8.181036e-4 |
| near=0.2 | 深度0.15の中心を除外。背景画像・ゼロ勾配へ変化 | 2.809127e-1 |

各設定の画像と参照との差は最大6.871182e-8。
中間量や勾配を含む単独設定検査の95比較全体では最大9.253311e-7だった。
この値は各比較に適した既存のatol/rtolで判定した結果であり、全比較へ単一の許容差を適用したものではない。

複合設定はextent=2.8、lowpass=0.6、clamp係数=1.1、cutoff=0.01、cap=0.9、T閾値=0.001、near=0.2。
Phase 1の12×12・2 Gaussianの入力で、全28生パラメータ成分を中央差分でも検証した。

| パラメータ群 | Backward / CPU Autogradの最大絶対差 | 判定対象FDとの最大絶対差 |
| --- | ---: | ---: |
| means | 3.254212e-8 | 5.729496e-6 |
| quaternion | 2.789840e-9 | 5.586189e-6 |
| log-scale | 2.714384e-9 | 4.714821e-6 |
| opacity logit | 3.202664e-9 | 5.524606e-6 |
| RGB logit | 4.911054e-9 | 4.578382e-6 |

FD刻み幅はPhase 1と同じ[3e-3,1e-3,3e-4]×max(1,abs(raw))。
分岐監査と、FD同士の安定性だけによる隣接ペア選択も同じ。5群とも両隣接ペアが安定した。
全刻み幅の全成分を含む最大FD絶対差は4.037097e-5。各刻みの値はJSONに保存した。
画像はatol=1e-6/rtol=1e-5、Backwardは1e-5/5e-4、FDは2e-4/2e-3を使用し、拡大していない。

implicit defaultと明示的RasterConfig()、設定A/Bの逆順Backward、非既定設定後の既定値復帰、
10引数apply互換性を検査した。画像差は全て0。
勾配の最大差はそれぞれ7.450581e-9、3.725290e-9、1.862645e-9、1.396984e-9で、既存判定値内だった。
CPUのfrozen設定だけでなく、グラフごとの設定保持も確認している。
lowpass/cutoff/near/T閾値を0にした良条件の入力も参照と一致し、早期終了を無効にすると第4層が残ることを確認した。

## 実行結果・再現方法・成果物保全

既存venv、Python 3.12.13、Torch 2.6.0+cu124、Warp 1.16.0、RTX 4050 Laptop GPU、driver 610.88を使用。
依存関係の追加・更新なし。CUDA検査はgsplat初期化に依存せず、数値・コンパイル失敗をskipにしない。

```powershell
# 変更前、および設定伝播のみを追加した段階では、それぞれ57 passed / 1 xfailed
.\.venv\Scripts\python.exe -m pytest -q -ra
# 新規CPU/CUDA検査: 44 passed
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_raster_config.py tests/test_configured_rendering.py
# 最終全体: 101 passed / 1 xfailed / skipなし / 2 warnings、15.91秒
.\.venv\Scripts\python.exe -m pytest -q -s -ra
# CPU対象: 50 passed / 52 deselected / 2 warnings、9.50秒
.\.venv\Scripts\python.exe -m pytest -q -ra -m "not cuda"
```

警告は既存のjapanize_matplotlib経由のdistutils非推奨警告2件。
変更した9 Pythonファイルに対するRuff check、format --checkも成功。
最終全体実行は通常CUDA環境でsubprocessの標準出力を取得し、`outputs/phase2/test_output.txt`へ保存した。
`MEASUREMENT `に続くJSON 332レコードを、環境・要求/実効設定・ソースhashとともに
`docs/phase2_measurements.json`へ保存した。通常pytestはこの記録やbaselineを更新しない。

開始時と最終テスト前後で以下のSHA256が一致した。既存許容差も元のファイル内のまま維持した。

| 成果物 | SHA256 |
| --- | --- |
| Phase 0 expected.npz | `540320947c56fbbd9d1170f3a593e4a8da534abef3b017f3fcc5f1755bee74e8` |
| Phase 0 metadata.json | `a765accfc981acfb5afb7d1059c64e63af973ab52482165a928c9e3fae669dc1` |
| Phase 0 scenes.json | `bcd5c2b3b14d30c7601c75485a545d86fe38c504bda9c18755ccf466b70014ae` |
| Phase 1 measurements.json | `05b8b14c626ce02fe0eb05858e2459cfd112e29a3da59bf2989e147d3010d8a6` |
| Phase 1 research_validation.md | `f552d2cbb59b017ff7e5b7d85ecc0b39e783d68be73b4635337f86cbc700a750` |

## 制限と停止境界

空点群のIndexErrorは既存のstrict xfailを維持し、正常仕様として扱わない。
設定の有効範囲は入力検証上の条件であり、巨大な支持範囲のメモリ使用、極小nearでの投影、
lowpass=0での悪条件共分散、多数の高opacity層でのT underflow等に対する安全保証ではない。
alpha_cap<1は1-alphaの直接のゼロ除算を避けるが、長い積のunderflowまで防ぐものではない。
同一深度キー、深度uint32 overflow、quaternion極小ノルム、高階微分、別GPU・別CUDA streamは未検証。

今回の数値差は既存の許容差内であり、式・基準データ・許容差を変更する必要はなかった。
設定の変更は明示的なPython API指定によるopt-inで、学習デフォルト・戦略・ハイパーパラメータは維持した。
Phase 2で停止し、Phase 3の非線形投影実験や再学習、コミット・pushは行っていない。
