# 2026-11-06 Block Tune質感比較の検証

## 追加内容

- image2派生6種: 全体V=.90/1.00、Block34–41 V=.90/1.00、
  同区間のO/FFN-down強度=1.00、同区間のV=1.00と強度=1.00。
- 画像③: `image_3_individual_qkv_partial`。見えていないQKV・強度・Headは
  1.0で補完し、観測値と区別。独立した目視転記の照合を実施。
- 適用条件: `static`、`full_run`、`middle_30_70`、`identity_control`。
- `examples/block_tune_texture/`: 32組のUI/API比較例。

## テスト結果

PowerShellで実行:

```powershell
$env:PYTHONUTF8='1'
$env:MOSS_COMFY_ROOT=(Resolve-Path '..\ComfyUI').Path
$env:OMP_NUM_THREADS='1'
python -m pytest -q
```

**155 passed, 393 subtests passed**。GPUマーカーはプロジェクト設定により除外。
コンパイル検査と`git diff --check`も成功。Ruffは環境に未導入のため未実施。

UTF-8指定なしの初回実行では、既存`test_repository_contract.py`の次の3件が
WindowsのCP932読込で失敗した。コード変更で回避せず、UTF-8を指定した全体再実行で解消:

- `test_core_and_node_defaults_match_validated_presets`
- `test_example_generator_preserves_generation_inputs_and_default`
- `test_shipped_api_and_ui_use_current_default`

## 確認した契約

- 局所プリセットはBlock0–33を保持。Q/Kはimage2と同じ。
- 全区間/中盤/恒等対照でwrapper対象を一致させ、Vや強度を1.0に戻す派生でも
  必要な恒等wrapperを維持。
- 小型ネイティブMOSSの融合・分離QKVで、区間内は静的版、区間外は恒等対照と一致。
- INT8 ConvRotで区間内は倍率適用済み逆量子化参照、区間外は無倍率逆量子化参照と一致。
  保存テンソルは保持。
- 実Euler/ER-SDEサンプラーのコールバックが時間窓を駆動。
  25 stepsでは0始まり8–16が対象。区間端、複数回評価、例外後の再実行と復元を確認。
- Dynamic入力はネイティブ非動的delegateへ移行。時間指定とAFM・参照KVの併用は拒否。
- 新旧ノードの選択肢と既定値、比較例のmodel配線・widget値、生成条件保持を確認。

## 比較例の出典と限界

出典PNG: `edit_00001__20261106-050649_1523089_7612105a.png`。
未接続のK Row Balanceノードを除外し、元のmodel名、prompt、seed、25 steps、CFG2.5、
ER-SDE/simple、16:9、`flux2-vae.safetensors`を保持。
生成機に同名モデルが必要。モデルファイル同一性をPNGの名前だけから保証するものではない。
再生成: `python tools/build_block_tune_texture_examples.py`。

コードレビューはimage2/時間窓の範囲で指摘なし。画像③は独立した目視転記照合と
ロード/補完境界のテストを実施。実サイズモデルのGPU生成、画質改善、ComfyUIブラウザー上の
読み込み操作は未実施。時間窓の区間外は同じ逆量子化wrapper経路での倍率1.0であり、
wrapperなしの元INT8経路とのbitwise一致を保証しない。
