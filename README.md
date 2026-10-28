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
skip-cache, other block/attention patches and Slider interoperability are not qualified.
The owner's real-device operation is established; an additional deployment gate is
not required by this update. This does not imply every possible API/UI round-trip
or model combination was independently rechecked here.

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
