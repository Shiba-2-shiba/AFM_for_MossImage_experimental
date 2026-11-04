# K Row Balance 検証記録

2026-11-04。ComfyUI参考commit `7f09d3fa4a783868fe39d4d63cf4f815e58abecd`、
手元のPython3.10.11 / torch2.10.0+cpu / comfy-kitchen0.2.39でCPU試験。
既存プロジェクトの配布条件Python3.12+は変更していません。

## 試験結果

- AFM repo全体: 119 tests、361 subtests成功。MOSS_COMFY_ROOTを指定し、
  実際のComfyUIコアを使うCPU統合テストを有効にした結果。
- 既存の解析・画像インベントリ: 172 tests成功。新ノードのPNG条件識別を追加。
- 両プロジェクトのcompileall、git diff --check成功。Ruffは未導入のため未実施。
- V3登録、FP32/BF16 compute metadata、Q/V保持、再量子化を呼ばないこと、
  clone/cache識別、apply/unpatch、variant切替、非累積、RNG保持を検査。
- 独立レビューでfresh moduleがeval状態とhookを失う問題を確認。
  失敗テストで再現し、mode/cast flag保持、target hooks・instance overrides拒否を追加。
- clone-only OFFとは別に、max_change=0の同経路再構成対照を追加。

## 実チェックポイントのblock41だけを使用したCPU probe

| 条件 | 実際に変わったK scale行 | K row p99/p50 | median head log-norm std |
| --- | ---: | ---: | ---: |
| 元モデル / 再構成対照 | 0 | 4.11184044 | 0.29293859 |
| strength=+1 / cap0.03 | 3072 | 3.94625564 | 0.28270766 |
| strength=-1 / cap0.03 | 3072 | 4.25817422 | 0.30471152 |

INT8コードとQ/V scaleは全条件で一致。固定のCPU projection入力に対し、再構成対照は
QKV全出力が一致し、正負の補正でもQ/V出力は一致しました。
この数値変化は補正処理の動作確認であり、画像品質の改善を意味しません。

原本SHA256は前後とも
`f12df456dac38a69bedd290cee3a79895c48dcd4ca894d6f054d848c7e5ce2dd`。
計算・再構成用の最大作業領域見積もりは約87.3 MiB、追加projection保持領域は約27 MiB。
これらはフルモデル・GPU forward workspace・総VRAMを含む値ではありません。

フルモデルの画像生成、GPU/driver/backendの同等性、生成画像の改善、他の編集との併用は
未検証です。生成機ではbaseline、OFF、再構成対照の確認から始めます。
