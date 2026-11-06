# 画像別 Block Tune プリセット

`MOSS Image Block Tune Preset` を追加します。カテゴリは `model/patches/moss_image`。

```text
Load Diffusion Model → LoRA（任意）→ MOSS Image Block Tune Preset → KSampler
```

AFMも使う場合は、このノードとKSamplerの間へ接続します。
既存のResidual / Residual Test / K Row Balanceと重ねず、どれかを選んでください。
`off`は入力モデルのcloneを返し、このプリセットによる追加調整を行いません。

| preset | ブロック強度 | q/k/v |
|---|---|---|
| `image_1_individual_qkv` | 画像①の0～13・21～34。14～20・35～41は1.0 | 見えている28ブロックは各個別値。未確認14ブロックは1.70 / 1.00 / 0.80 |
| `image_2_global_qkv` | 画像②の全42ブロック | 全ブロック1.70 / 1.00 / 0.80 |

両方ともHeadは1.0です。画像①のHeadは未確認のため1.0、画像②は表示値1.0です。
2枚は別設定として扱い、画像②の強度で画像①の未確認箇所を埋めません。
q/k/vの未確認箇所・画像②の個別値は、ユーザーの指定によりグローバル欄の数値で補完しています。
これは画像から観測した個別値と区別し、JSONの`qkv_observed`へ記録しています。

例: ①のBlock 0は強度1.00、q=1.45、k=1.54、v=1.39。
①のBlock 35は強度1.00、q=1.70、k=1.00、v=0.80。
②のBlock 35は強度1.10、q=1.70、k=1.00、v=0.80です。

## 数値と適用方法

設定本体は次のJSONです。

- [画像①](../afm/presets/image_1_individual_qkv.json)
- [画像②](../afm/presets/image_2_global_qkv.json)

`moss block tune/moss_block_strength.py`の`applyGlobal()`は各ブロックの値を
一括上書きし、保存時は個別値を使用します。グローバル値をさらに掛ける処理ではありません。
このノードもJSONの各ブロックのq/k/vを最終倍率として使います。

強度は各joint blockの`proj.o`と`proj.ffn_down`に適用します。
Q/K/Vはprojection重みへ作用し、Q/K正規化の前です。融合QKVは行方向に3等分して適用します。
Attention計算後のq/k/vへ倍率を掛けるものではありません。
元チェックポイントへ書き込まず、ネイティブのweight wrapperをMODEL cloneへ登録します。
通常のLoRA適用後の重みに倍率を掛け、画像・テキスト、両CFG、全ステップを対象とします。
Headと正規化重みは変更しません。STRING出力には適用した全値と補完方針を記録します。

画像の数値は小数点以下2桁までの転記です。元ツール内部の非表示桁までは復元できません。
INT8などでは、対象のprojectionを計算時に逆量子化して倍率を適用します。
再量子化やモデルファイル保存はしませんが、計算時間・一時メモリが増える場合があります。
元ツールで保存したBF16チェックポイントとのbitwise一致や画質改善は保証しません。

## 使用例と確認範囲

`examples/moss_block_tune_image_1_ui.json`と`examples/moss_block_tune_image_2_ui.json`
はComfyUIへ読み込めるUIワークフローです。同名の`_api.json`はAPI用です。
既存例のモデル名・追加ノードはそのままなので、使用環境に合わせて変更してください。
既存のインストール先が別ディレクトリなら、今回の変更を含むパッケージをそこへ反映し、
ComfyUIを再起動してノードを読み込んでください。

テストは数値転記、融合・分離QKV、小型native MOSSでの重み倍率参照との比較、
元モデルへの切り替え、INT8 ConvRot projection、AFM接続、スキーマを対象とします。
実サイズのモデルでの画像生成・GPU動作はこの変更のテストには含みません。
