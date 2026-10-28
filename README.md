# MOSS Image AFM for ComfyUI

Experimental Attention Frequency Modulation for native MOSS-Image.
The node edits image-query/text-key logits on the actual latent grid, followed by
joint normalization over image and text keys. The model weights are unchanged.

## プリセットと検証状況

既定値は **`block_scope=0-2`** です。ノードの選択肢は次の3種類です。

| block_scope | 開発者の実機比較で選定した用途 |
|---|---|
| `0-2`（既定） | 全体の改善 |
| `24-26` | 感情表現などの強調 |
| `30-32` | ポーズなどの変化 |

開発者は全層を3層単位で比較し、上記3群を選定しています。また、このリポの
ノードを実際のComfyUI環境で使用しています（2026-10-26確認）。検証画像は
ローカルに保存され、リポジトリを軽量に保つため意図的に含めていません。
画像が未収録であることは、未検証であることを意味しません。

用途名はその実機比較に基づく選定理由であり、すべてのモデル・プロンプト・seedに
同じ効果を保証するものではありません。全層比較の正確なモデル版、prompt、seed、
試行数はこのリポには記録されていません。既存の動作検証条件とは区別して扱います。
詳細は [検証状況](docs/implementation-status.md) を参照してください。

## Installation

Place this package's contents in `ComfyUI/custom_nodes/AFM_for_MossImage` and restart
ComfyUI using the existing launch configuration. No additional runtime dependencies.
Python 3.12+ and native ComfyUI MOSS support are required.

Use **MOSS Image AFM** between Load Diffusion Model and KSampler.
Start with `mode=observe`, then `mode=edit`, strength `0.05`, block_scope `0-2`.
Strength 0.05 is a conservative starting point, not a universal optimum.
The API and UI examples use the same current preset/default.

The supplied API workflows retain the model and 1024×1024, 25-step,
CFG 2.5, ER-SDE/simple generation settings. Replace model filenames with installed
files as needed. API JSON is submitted through the API; it is not a UI graph export.
The examples additionally use the `ResolutionSelector` node; install its provider
or replace it with explicit width/height inputs. The AFM node itself adds no runtime dependencies.

## Controls

- `off` and edit strength 0 return a clone without new AFM hooks.
- `observe` records selected joint blocks without FFT or manual attention.
- `edit` filters only selected CFG image rows. Text rows and unselected branch
  rows preserve the same call's native output; this is not a guarantee that all
  later states remain unchanged.
- The node offers only `0-2`, `24-26`, `30-32`. The internal parser can represent
  other scopes for tests, but those are not additional node presets.
- `start_percent/end_percent` use the actual executed sampler step schedule.
- `max_workspace_mib` limits an estimate of AFM workspace, not total VRAM.
- `fail_mode=raise` stops on edit failure. `fallback_original` returns that call's
  complete native output for supported edit failures and records the failure.
  Native backend and run-contract errors still propagate.
- Summary logs are in `output/moss_afm/<run_id>.jsonl`. Detailed QKV are not saved.

## 2区間モード

`two_interval_mode` をONにすると、1つのAFMノードで区間ごとにblockを切り替えられます。
既存の `block_scope` / `start_percent` / `end_percent` が第1区間、
advanced設定の `second_block_scope` / `second_start_percent` / `second_end_percent` が第2区間です。
OFFが既定で、既存ワークフローは従来の1区間として動作します。
第2区間の入力欄は常設ですが、OFFでは生成に使用しません。

例えば `mode=edit`、`two_interval_mode=True` とし、次のように指定します。

| 区間 | block | 開始 | 終了 |
|---|---|---:|---:|
| 第1区間 | `30-32` | `0.0` | `0.5` |
| 第2区間 | `0-2` | `0.5` | `1.0` |

**重複時は第1区間だけを適用します。** この例の進捗 `0.5` では `30-32` を選び、
`0.5` を超えてから `0-2` に切り替わります。区間が広く重なった場合も同じ規則で、
両区間に異なるblockを指定しても同時適用はしません。
第1区間を既定の `0–1` のままにすると第2区間は動作しないため、終了値も設定してください。
どちらの区間にも該当しない進捗ではAFMを適用しません。

進捗は実際に実行するステップ列の `step_index / max(total_steps - 1, 1)` です。
上の例を25ステップで実行すると、最初の13ステップが第1区間、残り12ステップが第2区間です。
ステップ列に存在しない進捗値へ丸めることはありません。
`strength`、周波数マスク、CFG分岐、メモリ制限、エラー処理は両区間で共通です。
低・高周波の倍率は生成全体の進捗から計算し、第2区間でもリセットしません。

API例は [moss_afm_two_intervals_api.json](examples/moss_afm_two_intervals_api.json) です。
従来と同じモデル・seed・1024×1024・25ステップ・CFG 2.5・ER-SDE/simpleを保持しています。
モデル名と追加ノードの扱いは上記Installationの説明に従ってください。
AFMノードを直列につなぐ場合の重複エラーは引き続き有効です。

## Scope

Native ComfyUI MOSS only; standard Euler and ER-SDE.
Fused/unfused QKV, rectangular image grids, CFG batching and separate text lengths
are covered by the existing test plan. HF plugin loaders, torch.compile, multi-GPU,
skip-cache and other block/attention patches are not qualified, except for the
MOSS Image Residual node described below. Native LoRA model patches are preserved.
The owner's real-device operation is established; an additional deployment gate is
not required by this update. This does not imply every possible API/UI round-trip
or model combination was independently rechecked here.

## Residualブロック倍率

**MOSS Image Residual**を、ネイティブのモデル／LoRA読込ノードとAFMの間へ接続できます。
追加後はComfyUIを再起動してノードを読み込んでください。

```text
Load Diffusion Model → Load LoRA（複数可）→ MOSS Image Residual → MOSS Image AFM → KSampler
```

LoRAなし、Residual単独でも使用できます。`LoraLoaderModelOnly`と`LoraLoader`のMODEL出力に対応します。
`LoraLoader`のCLIP出力は、通常どおりテキストエンコードへ接続します。

| 入力 | 対象 | 既定値 | 範囲 |
|---|---|---:|---|
| `mid_multiplier`（9-25の倍率） | joint block 9～25 | 1.0 | 0～2 |
| `late_multiplier`（34-41の倍率） | joint block 34～41 | 1.0 | 0～2 |

番号は0始まり・両端を含みます。1.0は変更なし、0は対象blockの残差更新を0にします。
有限の0～2を指定してください。APIの数値文字列・boolはComfyUI本体のFLOAT変換に従い、変換後の値を検証します。
それ以外のblock、QKVパラメータ、出力headを直接変更しません。
倍率は対象blockのAttention出力とMLP出力にそれぞれ作用し、**LoRA適用後の結果全体**を調整します。
画像・テキストの両トークン、CFGの両分岐、全ステップが対象です。
AFMのCFG選択・時間区間とは独立し、AFMがoff/observe/区間外でもResidualは有効です。

モデルファイルの保存・再量子化は不要です。cloneへ設定を持たせるため、元MODELから分岐した別経路へ
倍率が漏れません。両倍率1.0では追加パッチなしのcloneを返します。
AFMの`24-26`では24・25に両処理が作用し、各blockのnative計算は1回です。
AFMの`fallback_original`もLoRA＋Residualを維持します。

通常の42層native MOSSのT2I向けです。効果を持つResidualの二重追加、AFM→Residualの逆順、
未知のblock/attention patchとの併用はエラーです。参照画像付きモデル、compile、cache、multi-GPUは未対応です。
保存済みの調整モデルとは丸め位置が違うため、INT8を含めbitwise一致を保証しません。

使用例：[Residual＋AFM](examples/moss_residual_afm_api.json)、
[ネイティブLoRA＋Residual＋AFM](examples/moss_lora_residual_afm_api.json)。
0.75/1.15は参照設定の例で、推奨画質を保証する値ではありません。
LoRA例の`MoodyMoss\\hatsunemiku_slider.safetensors`は検証環境で存在を確認した名前です。
別環境では互換性のあるインストール済みモデル・LoRAへ置き換えてください。

smoke helperもこのMODEL経路を実行します。baselineはLoRA＋Residualを保持し、AFMだけを外して比較します。
helperのLoRA対応は`LoraLoaderModelOnly`に限定します。CLIPも変更する`LoraLoader`を渡すと明示エラーとし、
その経路は通常のComfyUI API/UIで検証してください。必要なら`--extra-model-paths-config`で既存のモデル検索設定を渡せます。
例の生成時は`--model-lora NAME STRENGTH`を繰り返して複数LoRAを指定できます。
smokeでは未適用のLoRAキーがあれば失敗とし、ファイルごとの適用キーをreportへ記録します。

## Tests and smoke helper

```powershell
$env:PYTHONUTF8='1'
python -m pytest -q
$env:MOSS_COMFY_ROOT='C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/ComfyUI/.venv/Scripts/python.exe' -m unittest discover -s tests -t . -v
```

CPU tests require an existing PyTorch/pytest installation. Native contract tests
require the ComfyUI Python environment. The GPU smoke helper uses a separate
process and writes only to its specified output directory:

```powershell
python tools/run_gpu_smoke.py --comfy-root C:/ComfyUI/ComfyUI --workflow examples/moss_afm_edit_api.json --output test-results/current --modes baseline,off,zero,observe,edit
```

The helper reads the AFM settings from the node connected to KSampler in the
supplied API graph, preserving the preset, strength, branch selection and advanced
settings. Each mode overrides only mode, plus strength=0 for `zero`. A baseline-only
graph uses current defaults. It supports the supplied graph structure, not arbitrary
workflow execution. Keep `logging=summary`; `--modes` must begin with `baseline` and
contain no duplicates. Off/zero/observe and edit with strength 0 must be bitwise equal to baseline or
the helper fails. Edit must complete with finite outputs and no fallback; differences
are recorded, not treated as proof of improved image quality.

## Source archives

The obsolete bundled 0.1.0 ZIP has been removed to avoid installing stale defaults.
Generate a deterministic archive from the current checkout:

```powershell
python tools/build_release.py --output dist/AFM_for_MossImage-0.1.1.zip
```

Generated ZIPs and local validation images are not tracked in Git.
