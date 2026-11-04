# MOSS Image K Row Balance (Experimental)

選択したblockのK重みの行ごとの大きさを、head内で調整する仮説を検証するノードです。
重み統計の偏りが画像の欠点の原因だとは確認されていません。
レポートの偏り指標が減ることは処理の確認であり、画質改善の証明ではありません。

## 操作

`UNETLoader → MOSS Image K Row Balance → KSampler` と接続します。
MODEL出力はslot0です。slot1のSTRING出力には変更量と数値レポートが入ります。

| 設定 | strength | max_change | 意味 |
| --- | ---: | ---: | --- |
| 無変更の対照 | 0 | 0.03 | 入力MODELのcloneのみ。対象tensorを読まず、新しいpatchを付けない |
| 再構成だけの対照 | +1 | 0 | 実編集と同じmodule置換経路を通すが、重みの数値を変えない |
| 偏り緩和の試験 | +1 | 0.03 | 各headの行ノルムを中央値に近付ける方向。ただし各行は最大±3% |
| 逆方向の対照 | -1 | 0.03 | 同じ制限で偏りを拡大する方向 |

まずノードなし、strength=0、再構成だけの対照の一致を生成機で確認します。
再構成対照が違う場合は、K補正の効果として解釈せず停止します。その後、同一prompt・seed・
モデル・VAE・steps・CFG等で正負を比較します。最初からResidual/AFM/LoRAを重ねず、
この処理の効果を分離してください。UIのstrength既定値は0です。

## v2: 強度・対象ブロック・patcher互換性

既存の入力順序と既定値を保持し、末尾に任意の `blocks` 欄を追加しました。
省略した古いworkflowは従来通りblock41を使います。

- `strength`: −10～10。既定0。
- `max_change`: 0～0.95。既定0.03。0.3なら倍率0.7～1.3、0.5なら0.5～1.5。
  1未満に制限し、log計算と正のscaleを維持します。
- `blocks`: 0始まりの整数、カンマ区切り、両端込み範囲、`all`。
  例 `0,1,2,22,24,26,30,31,40,41` / `0-2` / `41`。
  全指定blockに同じstrength/max_changeを使います。

強めの比較設定例は **strength=2 / max_change=0.3** です。まず単独blockで確認してから
複数blockへ広げると変化を切り分けやすくなります。strength>1は中央値を越える過補正が
起こり得るため、「大きいほど正しく均される」「大きいほど画質が良い」とは限りません。
`clipped_rows` / `clipped_fraction` は倍率上限に達した行です。同じcapのままstrengthだけを
上げても、すでに上限にある行の変化は増えません。

旧版の `K Row Balance currently requires the standard ModelPatcher` は、補正前に
派生型・動的patcherを一律に拒否したものです。v2は通常のModelPatcher派生型を受理し、
動的入力はComfyUIの `get_non_dynamic_delegate()` で通常モデルへ変換します。
最初はcached loaderからの再読み込みがあり、次回以降はネイティブdelegateのキャッシュを
再利用します。動的モデルの内部メモリーを直接書き換える実装ではありません。
cached loaderがない場合など、変換できない入力はエラーのまま明示します。

レポートには受信したstrength/cap、block一覧と個別の数値、入力/使用patcher型、
delegate使用の有無が入ります。実行ログにも概要を出します。GPU上の動的モデル全体での
検証は未実施で、CPUテストはネイティブdelegate/cloneメソッドを使ったharnessです。

戻すには、このノードの入力を元のUNETLoaderのままにしてstrengthを0へ戻すか、
ノードを外して元workflowへ戻します。strength=0は「入力モデルへの追加変更なし」であり、
他の上流ノードの変更を取り消す機能ではありません。チェックポイントは保存しません。

## 補正内容

対象はnative MOSSの **選択したblock（既定41、0始まり）** です。42ブロック、24heads、head_dim128、
融合QKV `[9216,3072]` のK部分 `[3072:6144]` を扱います。
どのblockでも同じhead内の行バランス処理です。block0を指定しても、K全体を一律に
引き上げる処理にはなりません。低いRMSだけから必要な補正方向を決めません。

K行iについて、保存されたINT8コードqと行scale sから、
`r_i = s_i × ||q_i||2` をCPUのFP64で集計します。
ConvRotは入力軸の直交回転なので、この値は復号後の行ノルムに対応します。
各128行のhead内の中央値mに対し、

```text
g_i = clip((m / r_i) ** strength, 1 - max_change, 1 + max_change)
scale_i' = scale_i × g_i
```

をlog空間で計算します。headをまたいで大きさを揃えません。
ゼロ・非有限・不正scale/normをepsilonで隠さず停止します。
レポートのchanged_rowsは、FP32のscale値が実際に変化した行数です。

INT8コードは値を変更せずコピーし、K部分のFP32 scaleだけを変更します。
`D(WH)=(DW)H` により、入力軸のConvRotを保ったまま出力行の倍率を変更できます。
Q/Vのscaleは保持され、既存の量子化計算経路とcompute dtypeを維持します。
通常のLoRA/weight patchによる復号・再量子化や、forwardへの出力倍率hookは使いません。

Kはこの後にRMSNormとRoPEを通ります。一律なhead倍率は相殺されやすい一方、
行ごとに異なる倍率はKベクトルの方向を変えます。学習済みの異方性や位置表現を
変えてしまう可能性もあるため、正方向が良くなるとは決めません。
モデル重みの変更なので、全step・画像/テキストtokens・正負のCFG側に共通して作用します。

## 対応範囲と復帰

- canonical `int8_tensorwise` / `TensorWiseINT8Layout`、ConvRot group256、
  FP32 `[9216,1]` scale、biasなしのQKVに限定。
- compute metadataはFP32/BF16。FP8、非融合QKV、compiled/multi-GPU、
  GEMV-packed targetや追加のtarget scale設定は初期版の対象外。
- 対象QKVへの既存LoRA/diff/set/object patch、weight wrapper、モデルhookなどの
  衝突はエラーにします。互換性のない入力を黙って別の計算へ変更しません。
- freshなnative Linearをcloneへobject patchとして登録し、標準ModelPatcherが
  apply/unpatchで元のmoduleへ戻します。variantごとにcache識別を更新します。
  独自に複数cloneを同時patchする使い方は検証していません。
- 元モデルとは別に、対象QKVのINT8コピー等の一時領域・保持領域が必要です。
  保持領域は1blockあたり約27 MiBで、指定数に比例します。全対象の準備が成功した後に
  1つのcloneへまとめて登録し、途中の失敗で一部の元moduleを変更しません。
  ノードの推定値は総VRAM保証ではありません。GPU全体のメモリー挙動は未検証です。

## 比較用ファイル

```powershell
python tools/build_k_balance_examples.py --workflow path/to/workflow.json --output new-controls --include-baseline
python tools/build_k_balance_examples.py --workflow path/to/workflow.json --output strong-controls --include-baseline --blocks "0,1,2,22,24,26,30,31,40,41" --strength 2 --max-change 0.3
```

API JSONまたはUI 0.4 JSONから、baseline、identity、compress、reverseを作ります。
再構成対照(rebuild_control)も含まれます。元のモデル・VAE・prompt・seed・生成設定を
保持し、新しいMODEL経路と保存prefixを分けます。
出力先は未作成のディレクトリを指定します。生成キューへの送信は行いません。

## 開発時の検証

現在の数値と試験件数は [v2検証記録](k-row-balance-multi-validation.md) にまとめています。
初版の結果は [旧検証記録](k-row-balance-validation.md) として保持しています。

CPU上の実際のComfyUIコアを使い、量子化Linear、V3登録、Q/V出力の保持、
native ModelPatcherのapply/unpatch、variant切替と非累積を検査します。
large geometryの適合検査はmeta-backed MOSSと対象projectionだけで行い、
フルモデルをロードした画像生成とは区別します。

```powershell
$env:PYTHONUTF8 = '1'
$env:MOSS_COMFY_ROOT = 'path/to/ComfyUI'
python -m pytest -q
```

実ファイルのblock41だけを検査するCPU probeもあります。

```powershell
python tools/probe_k_row_balance.py --comfy-root path/to/ComfyUI --checkpoint path/to/model.safetensors --output new-probe
python tools/probe_k_row_balance.py --comfy-root path/to/ComfyUI --checkpoint path/to/model.safetensors --output multi-probe --blocks "0,1,2,22,24,26,30,31,40,41" --strength 2 --max-change 0.3
```

元モデルの前後hash、コードとQ/Vのscaleの一致、数値変化、CPU projectionの出力を記録します。
VAE・text encoder・画像生成は使用しません。GPU/driver/backendや実画像の品質は、
生成機で別途確認する必要があります。
