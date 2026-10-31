# Phase 3 — 透視投影の近似誤差

2026-10-31実施。43ケース、16独立反復、3段階の標本数をCPU float64で測定した。
最終回帰テストは **123 passed・1 xfailed・skipなし**。新規CPUテストは22件。
既存の実装・テスト・Phase 0〜2成果物33ファイルは、開始時・終了時のSHA256が一致した。
本番投影・Backward・学習・RasterConfig・既存参照関数・既存許容差は変更していない。

## 何を測定したか

対象は3D Gaussianからの**点標本の非線形投影分布**である。
rayに沿った3D密度積分、Gaussian Splattingのα画像、学習後の品質ではない。
カメラ座標は正のzが奥行き、距離はworld単位、投影平均はpixel、共分散はpixel²。
カメラ座標の入力を直接使い、zero-skewの内部パラメータを
`fx=500, fy=550, cx=320, cy=240`とした。

\[
\pi(X,Y,Z)=(f_xX/Z+c_x,\ f_yY/Z+c_y),\qquad
P\sim\mathcal N(\mu,\Sigma)\mid Z>z_{\min},\quad z_{\min}=0.1.
\]

`Z>0`だけでは比のモーメントの存在を保証しないため、厳密に正の下限を要求する。
今回の条件では分母が0から離れ、対象モーメントは有限になる。
画面外の標本を含め、画像矩形・Gaussian支持範囲・alpha・clamp・lowpassは適用しない。
標本の深度はclampせず、`Z>0.1`を満たさない標本を棄却する。
これは本番の「中心がnear以下ならGaussian全体を除外する」操作とは異なる。

一次予測は既存の純粋な`project_pinhole()`から求める。
標本側はNumPyで各点の比を直接計算し、一次予測の中間結果を利用しない。

\[
m^{(1)}=\pi(\mu),\quad C^{(1)}=J(\mu)\Sigma J(\mu)^T,
\qquad m_i^{(2)}=\pi_i(\mu)+\tfrac12\operatorname{tr}(H_i(\mu)\Sigma).
\]

横方向の二次平均補正は
\[
\Delta m_x=f_x\left(\frac{\mu_x\Sigma_{zz}}{\mu_z^3}-\frac{\Sigma_{xz}}{\mu_z^2}\right)
\]
で、縦方向はxをyに置き換える。二次共分散は実装していない。
Hessianはテスト内の直接投影式のTorch Autogradと照合した。

### originalとconditionedの意味

`original`は入力3D Gaussianの平均・共分散を展開点とする局所予測である。
無条件の非線形投影に有限の平均・共分散があると主張する列ではない。
`conditioned`は条件付け後の解析的平均・共分散を使う局所予測である。
条件付け後の分布はGaussianとは限らない。二次のTaylor平均にはその最初の2モーメントを使うが、
高次モーメントや残差を捨てる誤差は残る。

条件付きモーメントは次式で求めた。\(\phi\)は標準正規密度、\(\Phi\)は分布関数。

\[
a=(z_{\min}-\mu_z)/\sigma_z,\quad p=1-\Phi(a),\quad\lambda=\phi(a)/p,\quad
v=\Sigma_{:,z}/\sigma_z,
\]
\[
\mu_c=\mu+v\lambda,\qquad
\Sigma_c=\Sigma+vv^T(a\lambda-\lambda^2).
\]

これは`E[P|Z]`がZの一次式であり、残差共分散がZに依存しないことから、
全期待値・全共分散の法則と片側切断されたZのモーメントを使って得られる。
実装は`erfc`でpを計算する。今回の全ケースではp>0.5。
実験用ヘルパーは極端な上側裾（p<1e-12）を対象外として明示的に拒否する。
正定値共分散、有限入力、正の平均深度・nearを要求し、正則化で不正入力を通さない。

## ケースと標本数

基準はd=2、平均(0.2d,-0.1d,d)、標準偏差(0.04d,0.06d,0.05d)、無相関。
系列ごとに次の条件を変更する。回転によって深度分散と相関が同時に変わることも、
構築後の共分散・固有値・相関行列を保存して明示する。

| ID | 系列 | 値・固定条件 |
| --- | --- | --- |
| 0–4 | 深度 | d=0.25,0.5,1,2,4。標準偏差(0.08,0.12,0.1)は固定 |
| 5–9 | 深度の広がり | σz/d=0.01,0.03,0.1,0.2,0.4 |
| 10–14 | 異方性 | σx/σy=0.25,0.5,1,2,4。幾何平均0.05d、σz=0.05d |
| 15–19 / 20–24 | x / y軸回転 | 標準偏差(0.04d,0.06d,0.2d)を−60,−30,0,30,60°回転 |
| 25–29 / 30–34 | xz / yz相関 | ρ=−0.6,−0.3,0,0.3,0.6。他の相関0 |
| 35–39 | 光軸からの距離 | μx/d=0,0.25,0.5,1,2。μy/d=−0.1 |
| 40–42 | near付近 | d=0.11,0.15,0.2。標準偏差(0.02,0.03,0.05) |

全43ケース、各16反復、各反復で4,096・16,384・65,536採用標本を評価した。
最大標本列は合計45,088,768点。小さい標本数は同じ反復のprefixなので、これらを追加の独立標本と数えない。
case IDはquick/full間で不変。基準seed=20261030、NumPy PCG64、
`SeedSequence([seed, case_id, repeat])`で反復間の乱数列を分ける。
proposalは最大16,384点単位で生成し、上限は要求採用数の8倍。
超過時は未完了として採用数・提案数を保存し、CLIは非ゼロ終了する。
全ケースで必要数を確保できた。最終chunkの余剰採用数と実際の使用数は別々に記録した。

## 不確かさと読み方

各反復の標本平均・不偏共分散を計算し、その16個の推定量の平均をMC推定値とする。
反復間標準偏差s、MC標準誤差s/√16を保存し、成分別に
`推定値 ± 2.131 × MCSE`を近似95%区間として表示する。
係数は[NISTのt分布表、自由度15](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm)による。
反復推定量の近似正規性が前提で、全ケース・全成分同時の区間ではない。
16反復だけで裾の形や区間の被覆率を保証するものではない。

- 成分別の符号付き誤差は「予測−MC推定」。その区間はMC区間を反転・平行移動して保存する。
- `component_resolved_from_zero`はこの成分別区間が0を含まないという記録で、正しさのpass判定ではない。
- 平均L2誤差、共分散Frobenius誤差、相対Frobenius誤差は点推定。
  非線形なノルムの区間や全成分同時検定を意味しない。共分散の相対誤差の分母はMC共分散のFrobeniusノルム。
- 標本数間の変化は、各反復内の差を取り、その16個の**対応付き差**からMCSE・区間を求める。
  prefixの相関を無視して独立な2推定量の誤差を足し合わせない。
- 既存Golden/Forward/Backwardの数値許容差は統計的不確かさに流用していない。

解析式テストはatol=rtol=1e-10。標本生成テストは独立した固定seedの16反復で
条件付きモーメントを比較し、事前に定めた6 MCSE以内を要求した。
提案標本の採用率は解析的確率と二項分布の6標準誤差以内で照合した。
このテスト用の広い統計的判定と、実験表示用の近似95%区間は別の目的を持つ。

## 観測した結果

以下は各反復65,536点での値。平均誤差はpixel、共分散誤差は相対Frobenius誤差の百分率。
`original`と`conditioned`は同じ条件付きMC投影と比較している。

| 条件 | original平均：一次→二次 | conditioned平均：一次→二次 | 共分散：original / conditioned |
| --- | ---: | ---: | ---: |
| 基準 d=2、σz/d=0.05 | 0.252769 → 0.033221 | 0.252769 → 0.033221 | 0.6491% / 0.6491% |
| σz/d=0.01 | 0.021707 → 0.024484 | 0.021707 → 0.024484 | 0.0661% / 0.0661% |
| σz/d=0.1 | 1.178448 → 0.048769 | 1.178448 → 0.048769 | 3.5680% / 3.5680% |
| σz/d=0.2 | 5.277112 → 0.712570 | 5.277227 → 0.712807 | 20.9870% / 20.9878% |
| σz/d=0.4 | 31.487182 → 13.226862 | 32.571593 → 15.847767 | 85.3907% / 86.4748% |
| 固定標準偏差、d=0.25 | 8.330602 → 9.935231 | 14.331979 → 2.344711 | 23.4999% / 31.6251% |
| near付近、d=0.11 | 22.813896 → 46.393489 | 3.988561 → 0.387871 | 57.5570% / 12.1325% |
| near付近、d=0.15 | 3.821673 → 16.498935 | 6.169629 → 0.212065 | 6.7642% / 15.2476% |
| near付近、d=0.2 | 5.170913 → 1.962411 | 6.725733 → 0.658383 | 14.1405% / 17.1557% |

基準ケースのMC平均は(420.218396,184.872740)、成分別近似95%区間半幅は
(0.049656,0.063899) pixelだった。一次平均(420,185)との差は両成分で区間外、
二次平均との差は両成分で0を含む。残る0.033221 pixelの誤差を正確に確定したとは言えない。

σz/d=0.4では区間半幅が(0.257615,0.233186) pixelであり、数十pixelの誤差とは大きさが異なる。
共分散MCSEのFrobeniusノルムはMC共分散ノルムの約1.19%。
共分散の85%以上の差と標本変動を区別できる一方、細かな値の追加桁まで確定したとは扱わない。
二次補正後にも13.23 pixel程度の差が残り、二次局所近似が十分とは結論しない。

near付近d=0.11では採用確率は0.579260。元の分布に二次補正を加えると悪化した。
条件付きモーメントで展開すると平均の差は小さくなるが、共分散の差は12.13%残る。
d=0.15では条件付き一次平均・共分散の方が元の一次予測より誤差が大きい。
条件付けの影響と局所近似誤差が打ち消し合う場合があり、条件付き展開の優越も一律には主張しない。

### 未解決・収束の限界

σz/d=0.01（ID 5）は、一次平均・二次平均・一次共分散とも全成分の誤差区間が0を含む。
0.021707→0.024484という点推定の差だけでは、二次補正が悪化したとは判定しない。
二次平均は19/43ケースで両成分の区間が0を含んだ。この件数は多重比較を補正した発見率ではない。

16,384→65,536点の対応付き変化では、少なくとも1成分の区間が0を含まないケースが、
平均でID 1,9,18,24,37,38、共分散でID 0,14,19,21,35にあった。
多重比較で偶然生じるものもあり得るが、全ケースで収束済みとは報告しない。
全反復・全標本数の値と区間を保存し、細かな差は未解決のまま残す。
有利な標本数・seedへの置き換え、標本数の自動増加、判定値の緩和はしていない。

## 再実行・テスト

既存venvを使用。Python 3.12.13、NumPy 2.4.4、Torch 2.6.0+cu124、Windows 11。
実験はTorchのCUDAビルドが入っていてもCPU float64のみを使う。
quick CLIの別プロセステストではCUDAを非表示にし、Warp・gsplatのimportを禁止して、
測定・図生成まで完了しCUDA未初期化であることを確認した。
既存CUDA回帰テストは実GPUで実行し、コンパイル・数値失敗をskipに置き換えていない。

```powershell
# 変更前: 101 passed / 1 xfailed / 2 warnings、22.28秒
.\.venv\Scripts\python.exe -m pytest -q -ra
# 初回の追加CPU検査: 21 passed、21.05秒
# その後、未完了実験の保存テストを1件追加し、最終全体で新規22件を検証
.\.venv\Scripts\python.exe -m pytest -q -ra tests/test_projection_experiment.py

# 最初の全ケース実験: 測定部分5.43秒
.\.venv\Scripts\python.exe -m warpgs.projection_experiment --preset full --seed 20261030 --output outputs/phase3
# 図の軸ラベルを明確化した最終コードで再実行: 測定部分5.61秒
.\.venv\Scripts\python.exe -m warpgs.projection_experiment --preset full --seed 20261030 --output outputs/phase3_final

# 最終全体: 123 passed / 1 xfailed / skipなし / 2 warnings、32.85秒
.\.venv\Scripts\python.exe -m pytest -q -ra
# CPU対象: 72 passed / 52 deselected / 2 warnings、27.62秒
.\.venv\Scripts\python.exe -m pytest -q -ra -m "not cuda"
.\.venv\Scripts\ruff.exe check src/warpgs/projection_experiment.py tests/test_projection_experiment.py
.\.venv\Scripts\ruff.exe format --check src/warpgs/projection_experiment.py tests/test_projection_experiment.py
```

2件の警告は既存japanize_matplotlib経由のdistutils非推奨警告。Ruff・format・git diff --checkは成功。
最終pytest出力は`outputs/phase3_audit/tests_final.txt`に保存した。
測定時間は図生成を除く実測であり、性能ベンチマークではない。

CLIは`--preset quick/full`、`--seed`、`--output`、`--overwrite`を受け付ける。
既定presetはquick、既定seedは20261030、outputは必須。
quickはID 3（基準）、9（大きな深度分散）、40（near付近）の各16反復・4,096点。
非空の出力先は明示的な`--overwrite`なしで拒否する。上書きしても無関係なファイルは削除しない。
通常pytestは一時ディレクトリだけでCLI検査し、保存済み測定JSONやbaselineを更新しない。

## 保存物・差分・停止境界

追加したソースは`src/warpgs/projection_experiment.py`と`tests/test_projection_experiment.py`の2つ。
その他の追加は本書と`docs/phase3_measurements.json`。
後者は最終測定に、既存33ファイルのhash、テスト結果、全ケース再実行一致の検証情報を付加したもの。
2回のfull実行は、環境・時間・ソースhashを除く**全casesデータがJSON値として完全一致**した。
実験・既存参照のソースhashはCRLFをLFに正規化し、保全監査hashはファイル生バイトで比較した。

`outputs/phase3_final`にはmeasurements.json、summary.csv、run.logと、
mean_errors・covariance_errors・convergenceのPNG/SVGを保存した。
図の配置を目視確認した。CSVはノルムの要約、成分別誤差・区間・反復値はJSONが正本。
outputsは既存.gitignoreに従い非追跡で、コマンドから再生成できる。

以前のフェーズの未コミット差分があるため、通常の`git diff --stat`には既存の5ファイル、
92 insertions / 45 deletionsも表示される。これらはPhase 3の変更ではない。
今回の4ファイルは新規であり、既存33ファイルが不変であることを別途hashで確認した。
Phase 0 Golden Testは引き続き現行動作の回帰検証であり、数学的な正しさの証明ではない。

空点群のIndexErrorは既存のstrict xfailのまま。今回の新しい失敗やskipはない。
一要因の有限な43ケースから、全カメラ・全分布・他の乱数ライブラリ版への一般化はしない。
極端なtail conditioning、歪んだ分布での区間被覆率、他環境でのbitwise再現性、
clamp/lowpass、ray積分、GPU性能、投影近似の勾配、再学習への効果は未検証。

Phase 3の実験と報告で停止した。本番式の変更・Phase 4・コミット・pushは行っていない。
