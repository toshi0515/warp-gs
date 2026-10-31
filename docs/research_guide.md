# WarpGS研究基盤の読み方・操作ガイド

2026-10-31のソースに対応する。行番号はこの時点の位置であり、後日の編集で移動した場合は併記した関数名で検索する。
この文書だけを追加した。新しい実験・描画方式・テストは追加していない。
後半のlowpass比較は数式からの予測であり、新たに測定した画像や誤差ではない。

## 1. 最初に押さえる全体像

現在の基盤には、目的の異なる3つの経路がある。

| 経路 | 答えたい問い | 主なファイル |
| --- | --- | --- |
| 実際の描画 | Gaussianをどの画像へ変換し、その勾配をどう返すか | `model.py` → `renderers.py` → `my_warp_rasterizer.py` |
| 同じ数式の独立検証 | Warpの実装が、別に書いた同じ数式と一致するか | `reference_math.py`とCPU/CUDAテスト |
| 近似そのものの研究 | 一次投影が非線形な透視投影をどこまで近似するか | `projection_experiment.py` |

「同じ近似式の実装が一致する」と「その近似が対象の数学的量を正確に表す」は別の結論である。
さらに、モデルを再学習した後の画質・収束も別の問いになる。

まず第2節で描画経路、第3節で設定、第4節で検証コードを読み、第6節から既存の確認コマンドを選ぶとよい。
GPUのタイル処理を変更しなくても、設定の効果や投影の近似誤差を調べられる。

## 2. GaussianModelから画像まで

### 2.1 モデルに保存される値

入口は[model.py:13 — GaussianModel](C:/Users/daich/github/warpgs/src/warpgs/model.py:13)。
N個のGaussianについて、次の生パラメータをPyTorchの`nn.ParameterDict`に保存する。
`nn.Parameter`は、勾配を計算してoptimizerで更新できるTensorである。

| `model.params`のキー | 形状 | 保存値 | 描画時の値 |
| --- | --- | --- | --- |
| `means` | N×3 | world座標の中心 | そのまま |
| `quats` | N×4 | wxyz順のquaternion | Warpで正規化・xyzw順へ変換 |
| `scales` | N×3 | **標準偏差の対数** | `exp(raw_scale)` |
| `opacities` | N | opacityのlogit | `sigmoid(raw_opacity)` |
| `colors` | N×3 | RGBのlogit | `sigmoid(raw_color)` |

quaternion（四元数）は3次元回転を4成分で表す方法。
logitは確率pを実数`log(p/(1-p))`に写した値で、sigmoidはその逆変換。
scaleは分散そのものではなく標準偏差であり、二乗すると分散になる。

特に、コンストラクターの引数名は`scales`でも、正の標準偏差を直接渡すと意図した大きさにならない。
[model.py:42 — scales](C:/Users/daich/github/warpgs/src/warpgs/model.py:42)が既に`exp`を適用するためである。
opacity・色も同様に、保存値と描画値を区別する。
学習検査でいう「log-scale・opacity logit・RGB logitの勾配」は保存値に対する微分を指す。

### 2.2 Python APIとWarpの接続

[renderers.py:18 — WarpRenderer.__init__](C:/Users/daich/github/warpgs/src/warpgs/renderers.py:18)が設定を保持し、
[同:26 — render](C:/Users/daich/github/warpgs/src/warpgs/renderers.py:26)が物理量に変換したモデル値、
world-to-camera行列`w2c`、内部パラメータ行列`K`、画像寸法、背景色を渡す。

`w2c`はworld座標をカメラ座標へ移す4×4行列。
`K`のfx・fyはpixel単位の焦点距離、cx・cyは主点、すなわち光軸が画像と交わる座標である。
この投影式はskew（画像軸の斜交成分）が0であることを前提にする。
可視側はカメラ深度zが正。画素(x,y)の評価位置は(x+0.5,y+0.5)である。

37行の`WarpRasterizeFunction.apply()`から
[my_warp_rasterizer.py:406 — forward](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:406)へ入る。
これはcustom autograd Function、つまりForwardとBackwardを自分で定義するPyTorchの接続点である。
内部でTensorをWarp配列へ接続し、GPU上の処理を呼ぶ。
途中に`detach()`があっても、ここで実装したBackwardが入力への勾配を返す。

返り値は[renderers.py:10 — RenderOutput](C:/Users/daich/github/warpgs/src/warpgs/renderers.py:10)。
`image`はH×W×3のRGB Tensor。Warp経路の`alpha`は現在`None`であり、合成中のalpha配列が公開されるわけではない。

### 2.3 3Dの広がりを2Dへ投影する

[my_warp_rasterizer.py:35 — compute_cov3d_wp](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:35)は
回転Rと標準偏差sから、3D共分散を作る。

\[
\Sigma=R\operatorname{diag}(s_x^2,s_y^2,s_z^2)R^\top.
\]

共分散行列は、各方向への広がりと軸間の相関を表す。対角が分散、非対角が軸間の共分散である。
入力quaternionはwxyz順だが、Warpのquaternion型はxyzw順なので40行以降で並べ替えている。
極小ノルムに対する下限もあるため、純粋な数学参照と全入力で同じ定義ではない。

[同:62 — compute_gaussian_screen_data_wp](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:62)では、
中心をカメラ座標(x,y,z)へ変換してから、次を計算する。

\[
m=(f_xx/z+c_x,\ f_yy/z+c_y),\quad
\Sigma_{cam}=R_{cam}\Sigma R_{cam}^\top,
\]
\[
J=\begin{pmatrix}f_x/z&0&-f_xx/z^2\\0&f_y/z&-f_yy/z^2\end{pmatrix},\qquad
C=J\Sigma_{cam}J^\top+\lambda I.
\]

Jacobian（ヤコビ行列）Jは、中心の近くで3D位置の微小な変化が画像座標をどれだけ動かすかを表す。
これで共分散を投影するのは一次の局所近似であり、非線形投影後の正確な分布を計算しているわけではない。
本番ではJのx/z・y/zをclampするため、上の無制限の式に第3節の補正を加えて読む。
中心m自体にはそのclampを適用しない。

112行でlowpass λIを加え、119行以降で逆共分散を求める。
逆共分散は共分散の逆行列で、Gaussian内の距離を評価するために使う。
ただしdeterminant（行列式）の下限1e-6が有効になると、通常の厳密な逆行列とは異なる。
返す`cov2d_inv`は対称2×2行列の独立な3成分。コード内の`conic`もこれを指す。

### 2.4 描画候補を整理する

| 処理 | 入口 | 理解しておくこと |
| --- | --- | --- |
| 前処理 | [preprocess_kernel:133](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:133) | nearによるカリングと投影を行い、Gaussianが触れるタイルを求める |
| 累積和 | [forward:501](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:501) | exclusive prefix scanで複製先の配列位置を決める |
| 複製 | [duplicate_kernel:200](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:200) | Gaussianを、触れる各タイルの候補へ登録する |
| ソート | [forward:520](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:520) | タイル番号・量子化した深度キーで順序付けする |
| 範囲の取得 | [compute_tile_range_kernel:236](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:236) | 各タイルが読む候補配列の範囲を求める |

カリングは描画不要な候補の除外。タイルは16×16画素の区画。
prefix scanは各要素より前の個数を累積する操作で、ここでは配列の書込位置を計算する。
この仕組みは「全Gaussianを全画素で調べる」処理を減らすためのもの。
新しい数式を考える最初の段階では、この整理機構まで書き換える必要はない。
ただし支持範囲を広げる式に変えた場合は、タイル候補が欠けないか確認する必要がある。

### 2.5 各画素の色を求める

[my_warp_rasterizer.py:263 — render_kernel](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:263)で、
画素中心pと投影中心mの差dから、Mahalanobis距離の二乗を計算する。

\[
q=d^\top C^{-1}d,\qquad g=\exp(-q/2),\qquad
\alpha=\min(\alpha_{cap},o\,g).
\]

Mahalanobis距離は、Gaussianの広がり・向きを基準とした無次元の距離。
ここでのqはquaternionとは別の記号である。
gは画素評価用のGaussian関数、oは中心のopacity、alphaはその画素での不透明度。
「GPU kernel」はGPU関数を、「Gaussian kernel」はこのgのような重み関数を意味し、用語を区別する。

支持範囲とalpha cutoffを通過した候補を、手前から順に合成する。

\[
C_{acc}\leftarrow C_{acc}+T\alpha c,\qquad
T\leftarrow T(1-\alpha),\qquad I=C_{acc}+T b.
\]

cはRGB、bは背景色。Tはtransmittance（透過率）、つまり背後の色がどれだけ届くかを表す。
初期値はT=1、累積色=0。これをalpha compositing（不透明度を使った合成）という。

### 2.6 Backwardはどこを通るか

loss（画像から作るスカラーの評価値）の`backward()`が呼ばれると、
[my_warp_rasterizer.py:574 — backward](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:574)へ入る。

```text
loss → grad_image
     → backward_render_kernel：画面平均・逆共分散・色・opacityへの勾配
     → Warp Tape：3D位置・quaternion・標準偏差への勾配
     → PyTorch：exp・sigmoidを通り、生パラメータへの勾配
```

[backward_render_kernel:334](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:334)は手書きの微分。
Forwardの最終Tと最後の寄与位置を使い、寄与したGaussianを逆順にたどる。
一方、投影部分は[629行のctx.tape.backward](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:629)で自動微分する。
TapeはForward演算を記録し、その計算を逆向きにたどって微分するWarpの仕組みである。

各Forwardが自身の設定をctxに保存するため、別設定の描画を挟んでも元のグラフは元の設定を使う。
カメラ、内部パラメータ、背景、寸法、RasterConfigにはこの接続から勾配を返さない。
カリング・ソート・閾値の境界を横切る微分も、滑らかな数式と同じ意味では扱えない。

学習では[train.py:182](C:/Users/daich/github/warpgs/src/warpgs/train.py:182)で描画し、191行で`loss.backward()`を呼ぶ。
Gaussianの増減は[train.py:147](C:/Users/daich/github/warpgs/src/warpgs/train.py:147)からのgsplat `MCMCStrategy`であり、
現在`densification.py`という独立ファイルはない。単純な数式検証の入口として学習を実行する必要はない。

## 3. RasterConfigの7設定を数式へ対応させる

[raster_config.py:10 — RasterConfig](C:/Users/daich/github/warpgs/src/warpgs/raster_config.py:10)はfrozen dataclass、
すなわち作成後のフィールドを書き換えられない設定型。学習パラメータではない。
[my_warp_rasterizer.py:24 — make_warp_config](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:24)が
各Forward用のfloat32 Warp structへ変換する。

| 設定 | 既定値・単位 | 数式・分岐への影響 | ソース位置 |
| --- | --- | --- | --- |
| `gaussian_extent` | 3.33、無次元 | q≤extent²を支持範囲とし、タイルAABBの各軸半径もextent√Ciiとする | [preprocess:174](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:174)、[render:310](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:310)、[Backward:379](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:379) |
| `covariance_lowpass` | 0.3、pixel² | C←C+λI。共分散・逆共分散・支持範囲・画像とその勾配に影響 | [投影:112](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:112) |
| `jacobian_clamp_factor` | 1.3、無次元 | Jで使うx/zを±factor×W/(2fx)、y/zを±factor×H/(2fy)へ制限 | [投影:82](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:82) |
| `alpha_cutoff` | 1/255、無次元 | cap適用後のalphaがcutoff未満なら寄与を飛ばす。等号は寄与側 | [render:314](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:314)、[Backward:383](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:383) |
| `alpha_cap` | 0.99、無次元 | alpha=min(cap,o exp(-q/2))。飽和中はopacity・形状経由の微分を止めるが、色の微分は残る | [render:313](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:313)、[Backward:391](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:391) |
| `transmittance_threshold` | 1e-4、無次元 | 合成・T更新後にT<thresholdなら終了。0は早期終了無効 | [render:321](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:321) |
| `near_plane` | 0.1、world単位 | 中心のカメラ深度z≤nearならGaussian全体を除外 | [preprocess:165](C:/Users/daich/github/warpgs/src/warpgs/my_warp_rasterizer.py:165) |

AABBはaxis-aligned bounding box、座標軸に平行な外接矩形である。
T閾値はBackwardで別に判定し直すのではなく、Forwardが保存した終了位置と最終Tを通じて効く。
支持範囲・cutoff・nearは寄与集合を変えるので、設定に対する変化が常に滑らかとは限らない。

Python APIでは`WarpRenderer(config=RasterConfig(covariance_lowpass=0.6))`と指定する。
既存モデルの描画は`renderer.render(model, w2c, K, height, width, background).image`で取得する。
引数Tensorは描画に合うdevice・dtypeで用意する。この呼出例はAPIの説明で、新しい実験プログラムではない。

`config.effective_values()`は実際にWarpへ渡すfloat32値を返す。
例えば0.3は0.30000001192092896、0.6は0.6000000238418579になる。
参照比較には同じ入力をfloat32へ丸め、その値をCPU float64へ昇格して使うと、入力丸めと計算差を区別できる。
参照側の既定値は独立しており、RasterConfigを自動importして一致させる仕組みではない。

設定検証は有限値、範囲、float32変換後の有効性を検査する。
extent・clamp係数は正、lowpass・nearは非負、0≤cutoff≤cap<1かつcap>0、T閾値は[0,1]。
lowpass=0を許すことは、特異に近い共分散でも安全という保証ではない。
タイルサイズ、determinant下限、quaternionノルム下限、深度キー量子化はこの7設定には含まれない。
学習の`create_renderer()`は既定設定を使い、CLIやviewerからこの7項目を自由に変更する機能はない。

## 4. 3つの数学コードの役割

### reference_math.py：同じ式を別実装で検証する

| 関数 | 入口 | 使いどころ |
| --- | --- | --- |
| `covariance_3d` | [30行](C:/Users/daich/github/warpgs/src/warpgs/reference_math.py:30) | 回転・標準偏差から共分散を独立に構成する |
| `project_pinhole` | [67行](C:/Users/daich/github/warpgs/src/warpgs/reference_math.py:67) | clamp・lowpassなしの純粋な一次投影 |
| `project_legacy` | [76行](C:/Users/daich/github/warpgs/src/warpgs/reference_math.py:76) | 現行clamp・lowpass・determinant処理を合わせてWarpと比較する |
| `composite_reference` | [98行](C:/Users/daich/github/warpgs/src/warpgs/reference_math.py:98) | 投影済みの平均・逆共分散・色・opacityから直接合成する |
| `render_legacy_reference` | [148行](C:/Users/daich/github/warpgs/src/warpgs/reference_math.py:148) | 小さなGaussian集合の投影から画像までを比較する |

Torch CPU float64を基本とし、Warp・gsplatをimportしない。
タイル・scan・radix sortは再実装せず、少数Gaussianを深度順に直接合成する。
比較には一意でuint32範囲内の深度キーが必要。同じキーの順番を数学仕様として勝手に決めない。
`legacy`は現行動作に合わせた比較式という意味であり、理想的な物理レンダリングの正解を意味しない。

### projection_experiment.py：一次近似の限界を測る

[predictions:85](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:85)は一次平均・一次共分散と二次平均補正を計算する。
[conditioned_moments:55](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:55)はZ>nearで条件付けした3D分布の平均・共分散を求める。
[sample_conditioned:145](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:145)が標本を生成し、
[project_samples:77](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:77)が各点を非線形に投影する。

Monte Carloは乱数標本から対象量を推定する方法。標本数が有限なので推定にも不確かさがある。
この実験は「投影された点分布」を扱い、rayに沿って3D密度を積分した画像とは異なる。
GaussianがZ=0付近まで分布するとX/Z等のモーメントが存在しない場合があるため、正の下限Z>0.1を使う。
これは中心だけを見る本番nearカリングとは違う条件付けである。

[make_cases:103](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:103)が43条件、
[measure_case:205](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:205)が反復測定を定義する。
clamp・lowpass・alpha画像の実験はここには含まれず、学習やCUDAも必要ない。

### my_warp_rasterizer.py：実際に描画し、微分する

第2節のGPU経路そのもの。独立参照と一致するかはテストで確認する。
Warpの実装に合わせて参照を単にコピーすると、同じ誤りを共有するため、参照の独立性が弱くなる。
gsplatは[renderers.py:53 — GsplatRenderer](C:/Users/daich/github/warpgs/src/warpgs/renderers.py:53)から使える別backendだが、
設定・近似・実装が違うので厳密な数学的正解とは扱わない。
viewers配下の描画経路も独立した実装で、WarpRendererの設定変更がそのまま反映されるとは限らない。

## 5. Phase 0〜3で分かったこと・残っていること

以下は保存済みの実行記録である。本ガイド作成時にはGPU検証や実験を再実行していない。
pass件数は各フェーズ完了時の累積であり、全入力・全GPUへの保証ではない。

| フェーズ | 保存済み結果 | 検証済みの内容 | 主な限界 |
| --- | --- | --- | --- |
| Phase 0 | 18 passed / 1 xfailed | 固定シーンの画像と5生パラメータ群の勾配の再現性、配列ごとの実測許容差 | Golden Testは現行動作の回帰検証であり、数学的正しさの証明ではない |
| Phase 1 | 57 passed / 1 xfailed | 共分散・投影の解析検査、独立参照との画像・中間量・勾配比較、有限差分 | 小さな滑らかなシーン中心。大規模な勾配集計や全境界を網羅しない |
| Phase 2 | 101 passed / 1 xfailed | 7設定の個別効果、複合設定のBackward・有限差分、設定A/Bのグラフ保持、既定回帰 | 任意の設定組合せや極端な入力での安定性は保証しない |
| Phase 3 | 123 passed / 1 xfailed | 43条件×16反復の条件付き非線形投影統計、近似誤差とMC不確かさ、別プロセス再現性 | 全ケースの収束を確定していない。再学習・ray積分・投影近似の勾配は未検証 |

元の報告は[research_baseline.md](C:/Users/daich/github/warpgs/docs/research_baseline.md)、
[research_validation.md](C:/Users/daich/github/warpgs/docs/research_validation.md)、
[research_settings.md](C:/Users/daich/github/warpgs/docs/research_settings.md)、
[research_projection.md](C:/Users/daich/github/warpgs/docs/research_projection.md)にある。
Phase 0〜2の途中報告にある「後続は未実装」は、その報告時点の停止記録として読む。

Phase 3ではσz/d=0.4のケースで一次平均誤差約31.49 pixel、共分散相対誤差約85.39%が観測された。
一方、小さな深度分散では差が標本推定の不確かさに埋もれた。
二次平均補正もnear付近では悪化することがあり、常に改善するとは言えない。
成分別近似95%区間は全ケース同時の保証ではなく、細かな差を断定しないための情報である。

既知の不具合・未検証項目は次のように読む。

- 空点群のIndexErrorは既知の不具合。`xfail(strict=True, raises=IndexError)`で管理し、維持すべき正常仕様にはしていない。
- 極小quaternionのノルム下限、悪条件共分散・determinant下限、同一深度キー、深度キーoverflow、高階微分は未検証。
- 巨大extentのメモリ消費、多層合成でのT underflow、別GPU・別CUDA stream、大規模シーンのBackwardは保証範囲外。
- viewer/OpenGL、学習収束、MCMCの実学習、モデル再学習後の比較は研究フェーズで検証していない。
- GPUが使えないテストは理由付きskipになる。Goldenは環境不一致でも数値比較をskipし得るため、件数だけでなく理由を見る。
  実行・コンパイル・数値不一致をskipにしてよいという意味ではない。

## 6. 既存の機能を操作する

### 6.1 何を確認したいかでコマンドを選ぶ

以下のPowerShellコマンドはリポジトリルート`C:\Users\daich\github\warpgs`から実行する。
既存venvを前提とし、データセット・学習済みモデルは不要。依存関係の導入が必要な場合はREADMEを参照する。

```powershell
# CPUの独立数式を理解する：GPU不要
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_reference_math.py

# Warpの投影・描画・Backwardを比較する：対応CUDA環境が必要
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_warp_math.py

# 7設定が効くことと参照一致を確認する：対応CUDA環境が必要
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_configured_rendering.py

# 現行動作のGolden回帰。baselineは更新しない
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_warp_baseline.py

# 全体。GPUがなければ理由付きskipの内容まで確認する
.\.venv\Scripts\python.exe -m pytest -q -ra
```

`-q`は簡潔表示、`-ra`はskip・xfailなどの理由の表示。
値の詳細を見たい検証には`-s`を追加すると、テストの`MEASUREMENT`出力を確認できる。
`-m "not cuda"`はCUDAマーカー付きテストを除外する。学習の動作を検証するコマンドではない。
Warpテストはgsplat初期化から独立しているが、`train.py`自体は146行でgsplatを検査するため、両者の実行条件は同一ではない。

### 6.2 既存の投影誤差実験を読む・再実行する

CLI入口は[projection_experiment.py:445 — main](C:/Users/daich/github/warpgs/src/warpgs/projection_experiment.py:445)。

```powershell
.\.venv\Scripts\python.exe -m warpgs.projection_experiment --help
# 実行すると指定先に測定JSON・CSV・図・ログを新規生成する
.\.venv\Scripts\python.exe -m warpgs.projection_experiment --preset quick --seed 20261030 --output outputs/projection_reading
```

これは既存Phase 3機能の操作説明である。quickは代表3条件、各16反復・4,096採用標本。
fullは43条件、標本数4,096・16,384・65,536。非空の出力先は`--overwrite`なしでは拒否する。
確認だけなら既存の`docs/phase3_measurements.json`と報告を読めばよく、再実行は必須ではない。
`predictions.original`は元の3Dモーメントによる局所予測、`conditioned`は深度条件付け後のモーメントによる局所予測。
どちらも同じ条件付きMC推定と比べている。これを無条件投影の正確な平均と読み替えない。

### 6.3 記録と許容差の扱い

通常pytestはbaselineを更新しない。Goldenの記録操作には通常検査と別の明示的な指定が必要である。
値が合わない場合は、入力・実効設定・環境・分岐・数式を調べ、baseline更新や許容差拡大だけで合格させない。
Phase 1以降の主な固定判定値は次の通り。atolは絶対許容差、rtolは参照値に比例する相対許容差。

| 比較 | atol | rtol |
| --- | ---: | ---: |
| CPU解析式・手計算 | 1e-10 | 1e-10 |
| Warpの投影・画像と独立参照 | 1e-6 | 1e-5 |
| Warp BackwardとCPU Autograd | 1e-5 | 5e-4 |
| Warp Backwardとfloat32有限差分 | 2e-4 | 2e-3 |

Goldenの配列別許容差やMonte Carloの不確かさは、この表とは別の根拠で定められている。
有限差分は、入力を±hだけ動かしたForwardの差から微分を近似する方法。
刻み幅1つだけを選ぶのではなく、複数のhでの安定性と、寄与集合・ソート等が変わらないことを確認する。

## 7. 新しい数式を試すときの変更地図

以下は今後変更する場合の案内であり、本ガイドで実装する変更ではない。
まず何の一致を確認したいかを決め、既存の意味を保ったまま別関数・明示設定で比較できる形を選ぶ。

| 試したい内容 | 主に変更する候補 | 併せて確認する場所 |
| --- | --- | --- |
| 同じ投影式でlowpass等だけ変える | Python側のRasterConfig指定 | `test_configured_rendering.py`。カーネルの編集は不要 |
| 新しい3D共分散 | `compute_cov3d_wp`と独立したCPU参照関数 | 共分散の解析的性質、投影中間量、quaternion・log-scale勾配 |
| 新しい投影平均・共分散 | `compute_gaussian_screen_data_wp`、独立したCPU参照、新方式の研究予測 | 画面平均・逆共分散・対角・深度、タイルAABB、Tape勾配 |
| Gaussian以外の画素重み関数 | `render_kernel`、`backward_render_kernel`、独立参照の合成 | 支持範囲を変えるなら`preprocess_kernel`、中心/形状/opacity/色の微分 |
| 合成方式や手書きBackward | 対応するForwardと`backward_render_kernel` | T・終了位置・cap/cutoff、画面量のAutogradと有限差分 |
| カメラ勾配・新しい中間出力 | 上記に加え`WarpRasterizeFunction.forward/backward`の接続 | 戻り値・ctx・Tapeの接続。現在の公開契約にはないため別途設計が必要 |

投影が同じ種類の画面量を返すなら、合成Backwardはその量の勾配を返し、投影変更の微分はTapeが担当できる。
ただし「自動微分だから正しい」とせず、参照Autogradと有限差分で確かめる。
新しい方式が追加パラメータや別の画面表現を要求する場合は、接続部分も変更対象になる。
現在は投影方式を選ぶレジストリ等はなく、低レベル数式の差替え口が完全に分離された構造でもない。

検証の入口は次の通り。

- [tests/warp_diagnostics.py:56 — projection](C:/Users/daich/github/warpgs/tests/warp_diagnostics.py:56)：既存Warp関数を呼んで中間量を観測するテスト用ヘルパー。
- [tests/test_warp_math.py:85 — test_preprocessing_backward_paths](C:/Users/daich/github/warpgs/tests/test_warp_math.py:85)：画面平均経由・共分散経由の位置勾配を個別に調べる。
- [同:103 — test_screen_compositing_backward](C:/Users/daich/github/warpgs/tests/test_warp_math.py:103)：合成部分だけの勾配を調べる。
- [同:154 — test_public_backward_and_finite_differences](C:/Users/daich/github/warpgs/tests/test_warp_math.py:154)：公開経路を通した生パラメータ勾配を調べる。
- [tests/math_validation.py:93 — finite_differences](C:/Users/daich/github/warpgs/tests/math_validation.py:93)：各生パラメータの全成分を複数刻み幅で検査する。
- [tests/test_configured_rendering.py:119](C:/Users/daich/github/warpgs/tests/test_configured_rendering.py:119)：各設定が画像・勾配へ影響し、同じ条件の独立参照と合うか調べる。

カリング・支持範囲・alpha cutoff/cap・早期終了・ソート順の境界は、まずForwardの挙動例として検査する。
境界をまたいだ有限差分を滑らかな微分検査の成功例には数えない。
比較が崩れたときは、共分散→投影→合成→公開モデルの順で調べると原因を局所化できる。

## 8. lowpass=0.3と0.6を実行前に予測する

ここは理解のための手計算。新しい単一Gaussian実験コード・画像・測定値は追加していない。

### 8.1 lowpassがしていること

投影共分散へのλIの加算は、各画面方向へ一定の分散を加える操作である。
数学的にはGaussian同士の畳み込みでも共分散は加算される。
ただし現在の描画式には積分量を保存するための振幅補正がなく、正規化した画像をGaussianぼかしする処理と同一ではない。

1個のGaussian、黒背景、cap非活性の場合は
\[
I(p)=c\,o\exp\{-\tfrac12(p-m)^\top(C_0+\lambda I)^{-1}(p-m)\}
\]
を支持範囲・cutoff内で評価する。
λを増やすと共分散の固有値（主軸方向の分散）が増え、逆共分散が表す距離は小さくなる。
中心以外の重みは増え、支持範囲も広がる。中心では指数が0なので、その画素が中心と一致する限り値は同じ。

### 8.2 手計算できる配置

仮に中心(0,0,2)、単位回転、標準偏差(0.02,0.04,0.03)、fx=fy=50とする。
65×65画像でcx=cy=32.5なら、中心は画素(32,32)の評価位置に一致する。
opacity=0.7、RGB=(0.8,0.4,0.2)、背景は黒とする。
中心は光軸上でclamp非活性、capも0.7<0.99なので非活性である。

\[
C_0=\operatorname{diag}((50\cdot0.02/2)^2,(50\cdot0.04/2)^2)
=\operatorname{diag}(0.25,1.00).
\]

| 量 | lowpass=0.3 | lowpass=0.6 |
| --- | ---: | ---: |
| 画面共分散の対角、pixel² | (0.55,1.30) | (0.85,1.60) |
| 画面標準偏差、pixel | 約(0.742,1.140) | 約(0.922,1.265) |
| 中心画素のRGB | (0.56,0.28,0.14) | (0.56,0.28,0.14) |
| 中心から横に1 pixelのRGB | 約(0.2256,0.1128,0.0564) | 約(0.3110,0.1555,0.0777) |

最後の行は0.7×exp(−1/(2Cxx))にRGBを掛けた予測である。両設定ともこの画素は支持範囲内・cutoff以上。
上表は指定した十進数を用いた数式上の予測で、float32の測定結果ではない。
画素中心がGaussian中心からずれていると、画像内の最大画素も変化し得るため、「ピークが常に同じ」とは一般化しない。

予測する見え方は、中心がほぼ同じで周辺が明るく広がる画像。
正規化係数1/(2π√det C)を掛けていないため、画像のRGB総和が保存されるとは予測しない。
外周ではextentとalpha cutoffによって寄与画素が切り替わる。
lowpassを増やした差を、実装の計算誤差や数学的正解からの誤差と呼ばないことも大切である。

将来画像を比較する場合は、設定間の意図的な差Δ=I0.6−I0.3と、
各設定のWarp対独立参照の差Eλ=Iwarp,λ−Iref,λを別々に読む。
最大絶対差はmax|D|、MAEは全画素・全RGB成分の|D|の平均、RMSEはD²の平均の平方根。
差分画像の表示倍率とカラーバーを明記し、PNGへ量子化する前のfloat画像で数値を計算する。
これは結果の読み方の説明であり、本ガイドでは実験や保存処理を追加していない。

## 9. 本ガイドの確認範囲

記載したファイル・関数・行番号は実際のソースと照合した。
保存済みテスト結果は各フェーズの報告から引用し、今回実行した結果とは区別した。
変更はこのMarkdownだけで、以前のフェーズの未コミット差分には触れていない。
新フェーズ、実験追加、baseline更新、コミット・pushは行っていない。
