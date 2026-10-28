# MOSS Image Residual：ブロック倍率調整とAFM共存仕様

作成日：2026-10-29。調査時点のHEAD：`f10ab33`。
状態：会話で了承された実現方式を実装済み。native全79テストと実モデルGPU検証が成功。
詳細とUI確認の範囲は[検証記録](../../residual-validation.md)を参照。
ノード名、入力範囲、重複時の扱いなどの詳細は、以下に明示する設計判断である。

## 目的と利用経路

ネイティブComfyUIで読み込み、必要に応じてLoRAを重ねたMOSS MODELに対して、
ブロック9～25と34～41のResidual倍率を独立に設定し、既存AFMと併用できるようにする。

```text
UNETLoader → LoraLoaderModelOnly / LoraLoader（0個以上）
           → MOSS Image Residual → MOSS Image AFM → KSampler
```

Residual単独、LoRAなしも使える。LoraLoaderのCLIP出力は既存のテキストエンコード経路へ接続する。
新ノードはMODELのみを入出力し、モデル・LoRA・CLIPの読込機能は持たない。
両ノードはサンプリング中の同じモデル評価に作用する。生成を二段階に分割するものではない。

## 根拠

以下の行番号は調査時点のもの。参照フォルダーはGit管理外であり、配布物の実行依存にしない。

| ID | ソース | 確認事項 |
|---|---|---|
| E1 | `参考/moss block tune/moss_block_strength.py:241,264` | residualモードは対象blockの`proj.o`、`proj.ffn_down`の浮動小数点パラメータを乗算する |
| E2 | `参考/moss block tune/create_combined_int8_20261029.py:24,44,64,69` | 0始まり・両端を含む9～25×0.75、34～41×1.15。INT8版は50個の出力投影row scaleを変更 |
| E3 | `参考/moss block tune/moss_block_strength.py:409` | 参照UIの倍率範囲は0～2 |
| E4 | `afm/adapter.py:120,140,151,176,189` | native検出、既存DiTパッチ拒否、ブロック別AttentionOverride、cloneへの登録 |
| E5 | `afm/runtime.py:63,129` | runごとの状態とcleanup、CFG分岐・レイアウトの保持 |
| E6 | `nodes.py:7,17,20,78` | V3ノード、MODEL入出力、既存3プリセット、拡張の登録入口 |
| E7 | `tests/integration/test_native_hooks.py:29,44,80,116,159` | fused/unfused小型native fixture、no-op、2区間、prompt validationの既存検証 |
| E8 | `tools/smoke_contract.py:9`、`tools/run_gpu_smoke.py:53,87` | smokeは現在loader直結またはloader→AFMに限定。LoRA/Residual経路を実行しない |
| E9 | `tools/build_examples.py:20`、`tools/build_release.py:7` | 既存API例の生成とafm/tools/tests/docs/examplesの配布収録 |
| E10 | `pyproject.toml:5`、`README.md:94` | Python 3.12以上、native MOSS、AFM併用はEuler/ER-SDE |

手元のComfyUI `C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/ldm/moss_image/model.py:112`
では、`mod`の6要素をshift/scale/gateの2組へ分解し、Attention残差とMLP残差を順に計算している。
同ファイル`:185`のblock replacementへ`mod`が渡される。
`comfy/model_patcher.py:432`のcloneはLoRAのpatchesとmodel_optionsを保持する。
これらはインストール固有の根拠であり、上記パスを本番コードへ固定しない。

公開ソースでも同じ基本契約を確認した（2026-10-29取得）。実装時は使用中のComfyUIの契約を再確認する。

- [MOSS native blockとreplacement](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/moss_image/model.py)
- [ModelPatcher.cloneとパッチ登録](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_patcher.py)
- [ネイティブLoRAの適用](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd.py)

## ノードと設定

追加ノードのIDは`MossImageResidual`、表示名は`MOSS Image Residual`、
categoryは既存と同じ`model/patches/moss_image`とする。

| 入力 | 型 | 既定値 | 契約 |
|---|---|---:|---|
| `model` | MODEL | — | ネイティブMOSS。既存LoRAパッチを保持 |
| `mid_multiplier` | FLOAT | 1.0 | 9～25の17ブロック。0～2、UI刻み0.01 |
| `late_multiplier` | FLOAT | 1.0 | 34～41の8ブロック。0～2、UI刻み0.01 |

出力はMODELが1個。倍率の実行引数にも1.0を定義する。
コア関数・execute直接呼出しはboolを除く有限のint/floatを受理し、範囲外・NaN・Infinity・文字列をエラーにする。
ComfyUI APIでは本体がFLOATを先に変換するため、bool・数値文字列は本体によって数値化される。
ノードのvalidate_inputsは変換後の有限性・範囲を検証する。これは実装時に確認したnativeの契約に合わせた補正であり、
元のJSON型を拒否する目的でComfyUI本体を変更しない。リンク値はexecuteで検証する。
API経由の値を刻み0.01へ丸めたり、範囲へ黙ってクランプしたりしない。
範囲と刻みはE3を踏まえた設計判断。0.75/1.15は使用例であり、既定値や画質保証にはしない。

初期対応は通常の42 joint blocksを持つnative MOSSのテキスト画像生成とする。
モデル型・42層・非compileの確認は両倍率1.0でも行う。非対応モデルは明示エラーにする。
参照画像付きMOSS-Image-Edit、HFプラグイン、compile、skip-cache、multi-GPU、任意の他ブロック
パッチとの共存は今回の対応対象に含めない。Residual単独の新サンプラー認定は行わない。

## 数学的な動作

対象ブロックで、LoRA適用後のAttention/MLP出力をそれぞれ`A`、`F`、倍率を`s`とすると、
既存の変調・正規化を含む計算は次の形になる。

```text
h = x + (s × gate_a) × A(x)
y = h + (s × gate_f) × F(h)
```

`mod[2]`と`mod[5]`だけを乗算した新しいtupleを、そのblockの`original_block`へ渡す。
共有されるmodのtensorへin-place変更を行わず、shift/scale・他block・headは維持する。
MLPはAttention残差を反映した`h`で評価する。`x + s × (block(x) - x)`への置換は行わない。

- 対象ブロックでは画像・テキストの両トークン、CFGの両分岐、全サンプリング期間に同じ倍率を適用する。
  参照の重み調整と同じ適用範囲であり、AFMの`positive_only`や時間区間とは独立する。
- 出力投影ではLoRA差分を含む演算結果全体が倍率対象になる。`s × (W + ΔW)`に相当し、
  LoRA差分だけの倍率変更や`(s × W) + ΔW`とは異なる。
- モデルファイルの保存、LoRAの焼き込み、再量子化、パラメータの直接変更は行わない。
- 両倍率1.0は新しいResidualパッチも設定マーカーも追加せずcloneを返す。
  片方だけ1.0なら、もう片方の範囲だけを登録する。
- 有限な通常入力では、倍率0の対象blockは残差更新が0となる。演算を省略する最適化は初版に加えない。
- 保存済み調整モデルとの比較は丸め差を考慮する。gate乗算と重み/INT8 scale乗算のbitwise一致を約束しない。

## AFMとの合成と分岐保護

採用方式は独立Residualノード＋AFMによる既知パッチの合成とする。
AFM内へ入力を追加する方式より、Residual単独使用と既存AFM UIの維持に適する。
重みを変更する方式より、LoRA後の出力調整を明示しやすく、INT8ファイルの編集を不要にできる。

`ResidualBlockPatch`という自作の型と設定内容を照合して、AFMが受け入れてよいパッチを限定する。
汎用のパッチ合成フレームワークは導入しない。AFM設定だけ・文字列ラベルだけで未知のcallableを許可しない。

- AFM専用blockは従来経路、Residual専用blockは元のResidualパッチを保持する。
- 重複blockはAFMのAttentionOverrideを渡しつつResidualパッチ経由でnative blockを1回実行する。
  AFMの`24-26`と重なる24・25が必須検証対象。
- AFMのoff/strength=0/observe・区間外・非対象CFG行ではResidualが引き続き有効。
  fallback_originalも、その入力モデルのLoRA＋Residualを含む計算へ戻る。
- 既存AFMの2区間選択、優先順位、ログの既存フィールド、run cleanup、サンプラー制限を維持する。
- 効果を持つResidualの二重追加と、AFM→Residualの逆順追加は明示エラー。
  倍率1.0/1.0の追加はno-opとして既存パッチをそのまま保持する。
  AFM→AFMの重複拒否と未知のパッチ・wrapperの拒否も維持する。
- clone前の入力モデル、同じloaderから分岐する別経路、共有native moduleを変更しない。

## 受け入れ条件

| ID | 確認する結果 |
|---|---|
| A1 | V3に新ノードを追加し、旧AFMの入力順・既定値・IDと既存API例を保持 |
| A2 | 9～25、34～41だけを選択。8/26/33は非対象、9/25/34/41は対象。1.0側にはパッチなし |
| A3 | 両倍率1.0でLoRA付き／なしの元出力とbitwise一致、入力モデル・mod不変 |
| A4 | native小型モデルの浮動小数点出力投影を倍率変更した参照とgate方式が許容誤差内で一致。fused/unfused双方 |
| A5 | ネイティブで適用した1個／複数LoRAを保持し、LoRA込み出力に倍率が掛かる。synthetic LoRAと実ファイルの検証を区別 |
| A6 | AFMの全3プリセット、2区間・gap・重複24/25で各処理が1回ずつ働く。AFM off/zero/observeはResidual-onlyと一致 |
| A7 | AFMのCFG選択・fallbackとResidual両分岐適用が共存し、例外・反復生成後も他経路へ状態が漏れない |
| A8 | 無効倍率、非対応モデル、非対応mod契約、重複・逆順・未知パッチを明示エラー。無効な別方式へ自動退避しない |
| A9 | smokeがLoRA→Residual→AFMを実際に実行し、Residualを保持したbaselineで比較。入力にある未対応ノードを黙って省略しない |
| A10 | 実モデルINT8＋実LoRAの生成、UI保存・再読込、既存回帰の結果と未実行項目を区別して記録 |

画質改善や保存済み調整モデルとの同一画像は合否条件にしない。
追加依存関係、独自JavaScript、任意範囲UI、個別25スライダー、QKV倍率、Deltaモード、
Residualの時間区間設定、LoRAファイル生成・保存は今回の対象外。

実装タスク・検証コマンドは[実装計画](../plans/2026-10-29-residual-block-scaling.md)に記載する。
