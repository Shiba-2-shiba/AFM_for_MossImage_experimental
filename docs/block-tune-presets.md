# 画像別 Block Tune プリセット

`MOSS Image Block Tune Preset` を追加します。カテゴリは `model/patches/moss_image`。

```text
Load Diffusion Model → LoRA（任意）→ MOSS Image Block Tune Preset → MOSS Image AFM（任意）→ KSampler
```

全applicationでAFMを併用できます。AFMを使う場合はBlock Tuneの後、KSamplerの前へ接続します。
時間指定モードではBlock TuneとAFMが1つのサンプラーステップ進行を共有しますが、
Block Tuneの`application`と、AFMのblock・時間区間・CFG分岐は独立して作用します。
既存のResidual / Residual Test / K Row Balanceと重ねず、どれかを選んでください。
`off`は入力モデルのcloneを返し、このプリセットによる追加調整を行いません。

| preset | ブロック強度 | q/k/v |
|---|---|---|
| `image_1_individual_qkv` | 画像①の0～13・21～34。14～20・35～41は1.0 | 見えている28ブロックは各個別値。未確認14ブロックは1.70 / 1.00 / 0.80 |
| `image_2_global_qkv` | 画像②の全42ブロック | 全ブロック1.70 / 1.00 / 0.80 |
| `image_3_individual_qkv_partial` | 画像③で見える0～15・21～36。未確認の16～20・37～41は1.0 | 画像③で見える0～14・21～35は各個別値。未確認の15～20・36～41は1.00 / 1.00 / 1.00 |

3プリセットともHeadは1.0です。画像①・③のHeadは未確認のため1.0、画像②は表示値1.0です。
画像①・②・③は別設定として扱い、別画像の値で未確認箇所を埋めません。
q/k/vの未確認箇所・画像②の個別値は、ユーザーの指定によりグローバル欄の数値で補完しています。
これは画像から観測した個別値と区別し、JSONの`qkv_observed`へ記録しています。

画像③の元画像は`7WMYV4XJ4M584P309VBZ1Q3QH0.png`です。見えているQKVは
Block 0～14と21～35の30行、見えているBlock強度は0～15と21～36の32行です。
QKVが見えない15～20・36～41はグローバル欄の1.00 / 1.00 / 1.00、
強度が見えない16～20・37～41は1.0で補完します。例えばBlock 15の強度は0.85、
Block 36の強度は1.05ですが、両BlockのQKVは未確認のため1.00 / 1.00 / 1.00です。
Headは画面にないため`head_observed=false`として記録します。
グローバルQKV=1.00は未確認行へ最終値として設定し、個別値へ重ねて掛けません。

例: ①のBlock 0は強度1.00、q=1.45、k=1.54、v=1.39。
①のBlock 35は強度1.00、q=1.70、k=1.00、v=0.80。
②のBlock 35は強度1.10、q=1.70、k=1.00、v=0.80です。

## 画像②の質感比較プリセット

次の6プリセットは`image_2_global_qkv`から実行時に派生する実験用の比較条件です。
元の画像②JSONは変更しません。すべてQ=1.70、K=1.00を維持し、Headも1.0です。
`b34_41`は**0始まりのBlock 34～41（両端を含む）**だけを変更し、Block 0～33は画像②と同じです。

| preset | Vの変更 | Block強度の変更 |
|---|---|---|
| `image_2_global_v090` | 全42ブロックを0.90 | 画像②のまま |
| `image_2_global_v100` | 全42ブロックを1.00 | 画像②のまま |
| `image_2_b34_41_v090` | 34～41だけ0.90 | 画像②のまま |
| `image_2_b34_41_v100` | 34～41だけ1.00 | 画像②のまま |
| `image_2_b34_41_strength100` | 画像②のまま | 34～41だけ1.00 |
| `image_2_b34_41_v100_strength100` | 34～41だけ1.00 | 34～41だけ1.00 |

ここで「Block強度」は各Blockの**Attention出力`proj.o`とFFN出力`proj.ffn_down`の両方**へ
適用されます。Attentionだけ、またはFFNだけを戻す設定ではありません。
後半Blockの切り分けは、画像②を基準に次の2×2で比較できます。

| 34～41の条件 | V=0.80（画像②） | V=1.00 |
|---|---|---|
| Block強度=画像② | `image_2_global_qkv` | `image_2_b34_41_v100` |
| Block強度=1.00 | `image_2_b34_41_strength100` | `image_2_b34_41_v100_strength100` |

まず同じprompt・seed・sampler・VAEでこの4条件を比較し、その後にV=0.90を中間条件として
確認すると、後半BlockのVとBlock強度を分けて見やすくなります。
グローバルV=0.90/1.00は、後半だけを戻す比較と全ブロックを戻す比較を分けるための対照です。
これらは原因を切り分ける設定であり、肌や質感の改善を保証しません。
また、Block 34～41だけが質感を担う、または後半Blockが質感専用であることを示すものではありません。

STRINGレポートの`experiment`には、基準プリセット、変更対象Block、V上書き値、
Block強度上書き値を記録します。

## 適用区間（application）

`preset`とは独立して、次の`application`を選択できます。

| application | 動作 |
|---|---|
| `static`（既定） | 通常使用向け。時間比較用の進行管理を追加せず、選択プリセットを常時適用 |
| `full_run` | `middle_30_70`との比較用。同じpreset倍率を全ステップへ適用し、時間比較用の進行管理を使用 |
| `middle_30_70` | 生成進行率0.30～0.70（両端を含む）だけ、選択プリセット全体を適用 |
| `identity_control` | 時間指定モードと同じ対象・計算経路を使い、Block Tune倍率だけを全ステップで1.0にする対照 |

通常は`static`を使用します。`static`と`full_run`は同じpresetを全ステップへ適用し、
CPUの浮動小数点ネイティブテストでは数値が一致します。`full_run`は`static`より強い設定でも、
別のpresetでもありません。違いは`full_run`が中盤限定との比較に必要な進行管理を使う点です。

`identity_control`はBlock Tuneを`off`にする設定とは異なります。`off`はBlock Tuneの
weight wrapperを追加しませんが、`identity_control`は時間指定条件と同じwrapper対象・
逆量子化経路を保ったまま倍率を1.0にします。どちらの場合も、後段のAFMはAFM側の設定どおり動作します。

進行率はsigma値ではなく、実行ステップ順の
`step_index / max(total_steps - 1, 1)`です。25 stepsでは0始まりの8～16、
人が数えると9～17ステップ目の計9ステップが`middle_30_70`の対象です。

`middle_30_70`は対象区間だけ選択プリセット全体を有効にします。区間外ではQ/K/V、
`proj.o`、`proj.ffn_down`を含む**全倍率が1.0**です。区間外に静的な画像②設定を
残す動作ではありません。

時間による差を確認するときは、同じpresetで
`identity_control → full_run → middle_30_70`の順に比較してください。
AFMを併用する比較では、3条件すべてでAFMのmode・strength・block・時間区間・CFG分岐も固定してください。
3条件は同じweight wrapper対象と逆量子化経路を使用します。そのため、
`middle_30_70`の区間外は`identity_control`との比較ができます。一方、倍率1.0でも
wrapperを使わない元のINT8高速経路とのbitwise一致は保証しません。

時間指定の`full_run` / `middle_30_70` / `identity_control`はEulerとER-SDEの
テキストからの生成に限定します。AFMとは併用できますが、参照画像入力には対応しません。
動的ModelPatcherはComfyUIのネイティブ非動的delegateへ移してから実行するため、
初回にモデル再読み込みやメモリ使用量の変化が生じる場合があります。
このためGPUのINT8実行では`static`と`full_run`のbitwise一致を前提にせず、
時間比較は`identity_control` / `full_run` / `middle_30_70`の3条件内で評価してください。
従来の`static`は時間用wrapperを追加せず、既存どおりAFMの前へ接続できます。

## 数値と適用方法

設定本体は次のJSONです。

- [画像①](../afm/presets/image_1_individual_qkv.json)
- [画像②](../afm/presets/image_2_global_qkv.json)
- [画像③](../afm/presets/image_3_individual_qkv_partial.json)

`moss block tune/moss_block_strength.py`の`applyGlobal()`は各ブロックの値を
一括上書きし、保存時は個別値を使用します。グローバル値をさらに掛ける処理ではありません。
このノードもJSONの各ブロックのq/k/vを最終倍率として使います。

強度は各joint blockの`proj.o`と`proj.ffn_down`に適用します。
Q/K/Vはprojection重みへ作用し、Q/K正規化の前です。融合QKVは行方向に3等分して適用します。
Attention計算後のq/k/vへ倍率を掛けるものではありません。
元チェックポイントへ書き込まず、ネイティブのweight wrapperをMODEL cloneへ登録します。
通常のLoRA適用後の重みに倍率を掛け、画像・テキスト、両CFGを対象とします。
`static`と`full_run`は全ステップ、`middle_30_70`は上記の中盤だけが対象です。
Headと正規化重みは変更しません。STRING出力には適用した全値と補完方針を記録します。

画像の数値は小数点以下2桁までの転記です。元ツール内部の非表示桁までは復元できません。
INT8などでは、対象のprojectionを計算時に逆量子化して倍率を適用します。
再量子化やモデルファイル保存はしませんが、計算時間・一時メモリが増える場合があります。
元ツールで保存したBF16チェックポイントとのbitwise一致や画質改善は保証しません。

## 使用例と確認範囲

`examples/moss_block_tune_image_1_ui.json`と`examples/moss_block_tune_image_2_ui.json`
はComfyUIへ読み込めるUIワークフローです。同名の`_api.json`はAPI用です。
質感比較用のUI/API例は`examples/block_tune_texture/`にあり、ファイル名は
`{preset}__{application}_ui.json`と`{preset}__{application}_api.json`です。
画像②の基準、6派生プリセット、画像③について、`static` / `full_run` /
`middle_30_70` / `identity_control`の全組合せを、32組のUI/APIペアとして収録します。
比較workflowは、ユーザー提供のビーチバレーPNGに埋め込まれた実際のworkflowを元にし、
`flux2-vae.safetensors`、16:9、25 steps、CFG 2.5、固定seedを保持しています。
従来の`moss_block_tune_image_1_*` / `moss_block_tune_image_2_*`は古い独立例であり、
small VAEを使う元設定を保持しています。今回の32組の比較条件には使いません。
中盤限定Block TuneとAFMを正しい順序で接続した例は
`examples/moss_block_tune_middle_afm_ui.json`と
`examples/moss_block_tune_middle_afm_api.json`です。
既存例のモデル名・追加ノードはそのままなので、使用環境に合わせて変更してください。
既存のインストール先が別ディレクトリなら、今回の変更を含むパッケージをそこへ反映し、
ComfyUIを再起動してノードを読み込んでください。

テストは数値転記、融合・分離QKV、小型native MOSSでの重み倍率参照との比較、
元モデルへの切り替え、INT8 ConvRot projection、静的AFM接続、時間区間、
identity/full/middleの計算経路、動的patcher delegate、スキーマを対象とします。
実サイズのモデルでの画像生成・GPU動作はこの変更のテストには含みません。
