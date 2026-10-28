# Residualノードの検証記録 — 2026-10-29

対象：MOSS Image Residual、AFMとの合成、ネイティブLoRAを含むMODEL経路。
仕様は[Residual仕様](superpowers/specs/2026-10-29-residual-block-scaling-design.md)、
実装順は[実装計画](superpowers/plans/2026-10-29-residual-block-scaling.md)を参照。

## 確認した実装

- `MossImageResidual`の2倍率で、0始まりの9～25／34～41を独立に調整する。
- nativeのmodulationのAttention/MLP gateだけを局所的に乗算し、元MODELや共有modを変更しない。
- LoRAの既存パッチをcloneへ保持し、LoRA込み出力にResidual倍率を適用する。
- AFMの24・25でResidualと合成し、native blockを各評価で1回だけ呼ぶ。
- off/zero/observe、2区間のgap、CFG非選択行、fallbackでもResidualを保持する。
- 未知・不整合パッチ、効果のある重複/逆順追加を拒否する。

## 自動テスト

Python 3.12.11、PyTorch 2.12.0+rocm7.14.0、ComfyUI 0.42.0の既存環境で実行した。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MOSS_COMFY_ROOT='C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI'
& 'C:/ComfyUI/.venv/Scripts/python.exe' -B -m unittest discover -s tests -t . -q
```

最終結果：**79 tests / OK、skipなし**。
この環境にはpytestがないためunittestで全件実行した。別のCPU環境でもpytestを実行し、
その時点の56件と66 subtestsが成功した（native専用テストは環境未指定でskip）。
構文確認と`git diff --check`も成功。Ruffは未導入、専用typecheck設定はなく、検査依存を追加していない。

新規処理の欠落、AFMの既存DiT拒否、設定ログ欠落、ノード未登録、smoke経路欠落をそれぞれ失敗として
確認してから実装した。native fused/unfused小型モデルでは、出力投影を別モデルで直接倍率変更した参照と
`rtol=1e-5, atol=1e-6`で一致した。synthetic LoRAを1個/2個適用した独立の行列差分参照も通過した。
倍率1.0とAFMのno-opはbitwise一致、分岐保護、異常後のcleanup、run間の状態分離を確認した。

## 実モデルGPU検証

GPU：AMD Radeon AI PRO R9700。モデル：`moodyMossMixFreeDownload_v2025STEPSCFG3_int8.safetensors`。
テキストエンコーダー：`qwen3.5_9b_int8_convrot.safetensors`、VAE：`flux2-vae_small.safetensors`。
元例のprompt/seed、ER-SDE/simple、CFG 2.5を保持した。
LoRAは`MoodyMoss/hatsunemiku_slider.safetensors`を0.6で適用し、2個条件では
`MoodyMoss/aging_slider.safetensors`を0.2で追加した。各ファイルで168キーの適用を確認した。

| 条件 | サイズ・steps | 結果 |
|---|---|---|
| LoRA＋Residual 0.75/1.15＋AFM 24-26 | 512×512・3 | baseline/off/zero/observe/edit全成功 |
| 同上 | 1024×1024・25 | baseline/edit成功、editのfallback 0 |
| LoRAなし・あり、Residual 1.0/1.0 | 各512×512・3 | 各5モード成功、off/zero/observe完全一致 |
| LoRAなし、Residual 0.75/1.15 | 512×512・3 | 5モード成功、no-op完全一致 |
| LoRA＋Residual、AFM 0-2 / 30-32 | 各512×512・3 | 各baseline/edit成功 |
| LoRA＋Residual、AFM 24-26→0-2 | 512×512・3 | baseline/edit成功。step 0/1は24-26、step 2は0-2のみ編集 |
| LoRA2個＋Residual＋AFM 24-26 | 512×512・3 | 5モード成功、no-op完全一致 |

計33回の成功したサンプリングを確認した。全出力が有限で、AFMを有効にした全条件でfallback 0。
baselineは同じLoRA＋Residualを維持しAFMだけを外したもの。画質改善や保存済み調整モデルとの同一画像は
この検証の主張に含めない。モデルは保存・再量子化していない。

ローカルの詳細証拠は`test-results/residual/`にある：`native-final-tests.log`、
`gpu-lora-512-r2/report.json`、`gpu-lora-1024/report.json`、`matrix-summary.json`と各条件のreport/log/latent/image。
これらとモデルファイルはGit管理・配布ZIPの対象外。
一部の終了ログにはComfyUIの`ModelPatcher.__del__`がPython終了時に出す警告があるが、
対象生成はreportのsuccess、終了コード0、latentとAFM summaryで判定した。

## 独立レビューと修正

独立した読み取り専用レビューでCritical 0、Important 2、Minor 0。Importantは以下のとおり処理した。

1. ComfyUIはFLOAT入力をノード検証前に変換する。生JSONのbool/数値文字列をコア側で拒否できるという
   計画上の前提を補正し、nativeの数値変換を維持した。直接executeでは型を厳密に検証し、
   APIでは変換後の有限性・範囲を検証する。APIのNaN受理を再現したテストを追加して修正し、成功を確認。
2. 一部のLoRAキーだけが適用された場合、UUID変更だけでは検出できなかった。
   nativeローダーの未適用キー警告を検出してsmokeを失敗させ、ファイルごとの適用キーを記録する。
   有効キー＋存在しないblockの混在テストで修正を確認。

実生成の初回にはsmoke内のnative LoRAメソッド指定誤りを検出した。
ノードの`FUNCTION`入口を使うよう修正し、実safetensorsファイルを使う回帰テストとGPU再実行で確認した。
新しいAPI例の必須advanced入力の省略もnative prompt validatorで検出し、例に既定値を明示して修正した。

## 実環境と確認範囲

ユーザーがComfyUIを再読み込みし、稼働中の8000番サーバーで新ノード登録を確認した。
新LoRA付きAPI例はlive preflightでモデル/LoRA名・ノード型・出力ノードを確認した。
ネイティブのLoraLoaderModelOnly/LoraLoader両方の接続をprompt validatorで確認し、
ユーザーから実機で正常に動作するとの確認を受けた。

ブラウザー接続ホストが利用できず、nativeウィンドウ操作も最小化状態から復帰できなかったため、
エージェントによるUI保存・再読込の往復操作は未実施。ユーザーの実機動作確認とこの自動UI未実施を区別する。
検証用CPUサーバーは停止済みで、ユーザーのComfyUIプロセスは変更・停止していない。

compile/cache/multi-GPU/HF/参照画像付きMOSS-Image-Editは引き続き対応対象外。
通常LoraLoaderのCLIP変更を含む実生成は今回のGPU matrixに含めず、MODEL経路とAPI接続の確認に限定した。
