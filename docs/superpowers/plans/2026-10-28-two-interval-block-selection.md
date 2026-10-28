# MOSS Image AFM 2区間ブロック切り替え Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 単一AFMノードで2つの生成区間に異なるblockを指定し、重複時は第1区間を優先する。

**Architecture:** 既存AFMConfigに第2区間を追加し、全体進捗から優先区間を決定する。両区間のblockの和集合に一度だけフックを登録し、選択された区間に属するblockだけを動作させる。サンプラー状態と周波数計算は現在の仕組みを利用する。

**Tech Stack:** Python 3.12以上、PyTorch、ComfyUI V3 `ComfyExtension` / `io`、unittest / pytest。

**Spec:** [合意仕様・互換性方針](../specs/2026-10-28-two-interval-block-selection.md)

## Global Constraints

- 単一の `MossImageAFM` ノードに2区間モードを追加する。
- 両区間に該当する進捗では、第1区間だけを採用する。
- 周波数の倍率には生成全体の進捗を渡す。区間切り替え時に進捗を0へ戻さない。
- 新規4入力はスキーマ・実行引数・dataclassの末尾に追加し、入力省略時にも既定値で動くようにする。
- 両方のblock選択肢は `0-2`、`24-26`、`30-32`。
- 新規依存関係、独自JavaScript、直列ノード対応、区間別strengthは追加しない。
- 現在のnative MOSS、Euler / ER-SDE、CFG分岐、fallback、メモリ制限を維持する。
- この成果物は実装計画書。計画作成時点では製品コード・サンプル・テストを変更しない。

## Review Focus

1. 異なるblockを持つ区間が広く重なっても、第2区間のblockが漏れて適用されないこと。Task 1・2で固定する。
2. 旧UIの位置依存 `widgets_values` と新入力を持たないAPIが従来どおり動くこと。Task 3で固定する。
3. 同一block・同一ステップ内の複数モデル評価でも、二重編集や区間の先行切り替えが起きないこと。Task 2で固定する。
4. 第2区間でも全体進捗を使い、gap・1ステップ生成・境界のないステップ列を正しく扱うこと。Task 1・2で固定する。
5. 第2区間でのfallback、CFGの非対象行、observe、後続生成への状態持ち越しが従来の契約を守ること。Task 2・4で固定する。

---

## 現状の根拠と変更対象

調査時点：2026-10-28、HEAD `f47831f`。以下の行番号はこの時点のもの。

| ファイル | 確認した現在の動作 | 実装時の責務 |
|---|---|---|
| `afm/config.py:52` | 1組のblock・開始・終了を持つ | 第2区間の4フィールド、入力検証、優先区間の選択 |
| `afm/adapter.py:44` | 1組の時間範囲だけを判定 | 時間で優先区間を選んでから、その区間のblockか判定 |
| `afm/adapter.py:140` | 単一scopeのフック登録、二重インストール拒否 | 有効なscopeの和集合を登録。重複拒否は維持 |
| `nodes.py:16` / `nodes.py:49` | 既存入力順と位置引数による設定生成 | 新規optional入力を末尾追加、設定生成をキーワード引数化 |
| `tools/smoke_contract.py:9` | dataclassから入力名を取得し、第1blockを検証 | 有効な第2blockもノードの選択肢に照合 |
| `tools/build_examples.py:17` | baseline / observe / editを生成 | 既存3種を維持し、2区間のAPI例を追加生成 |
| `afm/runtime.py:49` / `afm/runtime.py:95` | 全体進捗、設定全体とstep/block/actionの記録 | 原則コード変更不要。追加設定の記録と状態分離をテスト |
| `afm/attention.py:19` / `afm/spectral.py:6` | 全体進捗から低・高周波の倍率を計算 | コード変更不要。回帰テスト対象 |

変更するテストは `tests/unit/test_runtime.py`、`test_repository_contract.py`、`test_smoke_contract.py`、
`tests/integration/test_native_hooks.py`。区間の純粋ロジック用に `tests/unit/test_intervals.py` を新設する。
ドキュメントは `README.md` と `docs/implementation-status.md` を更新する。
例は `examples/moss_afm_two_intervals_api.json` を新設し、既存の1区間UI/API例は旧形式の互換性確認にも使う。

## Task 1: 設定モデルと区間選択

**Files:** Modify `afm/config.py`。Create `tests/unit/test_intervals.py`。

**Interfaces:**
- 既存フィールドの後に `two_interval_mode: bool = False`、`second_block_scope: str = "0-2"`、`second_start_percent: float = 0.5`、`second_end_percent: float = 1.0` を追加。
- `AFMConfig.active_interval(self, progress: float) -> int | None` を追加。返値は第1区間 `1`、第2区間 `2`、対象外 `None`。
- この関数は時間だけで判断する。mode・branch・現在のblockは判断材料にしない。

- [x] **失敗するテストを追加する。** 境界テストの中心となる期待値は以下。

```python
config = AFMConfig(start_percent=0.0, end_percent=0.5,
                   two_interval_mode=True,
                   second_start_percent=0.5, second_end_percent=1.0)
assert [config.active_interval(p) for p in (0, 0.25, 0.5, 0.75, 1)] == [1, 1, 1, 2, 2]
```

  追加ケースを `test_disabled_mode_preserves_primary_window`、`test_overlap_gap_and_zero_width`、
  `test_interval_validation`、`test_executed_step_schedule_boundaries` として固定する。
  第1 `[0.2,0.8]`・第2 `[0.1,0.9]` では進捗0.1→2、0.5→1、0.9→2。
  第1 `[0,0.25]`・第2 `[0.75,1]` では0.5→None。
  25ステップは1が13回・2が12回、24ステップは各12回、1ステップは進捗0で第1区間。
  開始＝終了は一致時のみ有効。NaN、Inf、boolの数値代用、逆転範囲、0–1外を拒否する。
- [x] **RED確認：** `python -B -m unittest tests.unit.test_intervals -v`。未追加の引数・メソッドを原因として失敗すること。
- [x] **最小実装：** 第1範囲を最初に判定し、該当したら1を返す。該当しない場合だけ、ONの第2範囲を判定する。progress自体も有限な0–1の数値として検証する。OFFでも入力数値の型・有限性・範囲は検証する。
- [x] **GREEN確認：** 上記コマンドと `python -B -m unittest tests.unit.test_spectral -v` が成功すること。従来の位置引数呼び出しの対応が変わっていないこともテストする。
- [x] **変更をレビューし、作業ブランチの方針に従ってコミットする。** 推奨名：`feat: add two-interval AFM configuration`。

## Task 2: 1つのサンプリング実行内でblockを切り替える

**Files:** Modify `afm/adapter.py`、`tests/unit/test_runtime.py`、`tests/integration/test_native_hooks.py`。

**Interfaces:**
- Task 1の `config.active_interval(progress)` を使用する。
- `install_afm(model, config)` の公開シグネチャは維持する。
- フックの `context` に `interval_ids: tuple[int, ...]` を追加。第1専用 `(1,)`、第2専用 `(2,)`、共通 `(1, 2)`。
- `AttentionOverride` を直接生成する既存テストにも明示的な `interval_ids=(1,)` を追加する。

- [x] **失敗するテストを追加する。** `test_second_interval_uses_global_progress` で進捗0.75の編集関数への引数が0.75であることを検証する。`test_overlap_suppresses_secondary_blocks` では進捗0.5で第1blockだけが編集され、第2専用blockはnative出力そのものであることを確認する。modeをobserveに変えた場合も第2専用blockはobservedではなくskippedとする。
- [x] **経路の回帰テストを追加する。** `test_shared_blocks_edit_once_per_call`、`test_gap_dispatches_native`、`test_multiple_evaluations_keep_interval`、`test_second_interval_preserves_cfg_rows_and_fallback` を追加。共通blockで編集呼び出しは1回、gapでは0回、同一step内の評価では区間固定、非対象CFG行・text行はnative出力を維持し、編集失敗では部分変更を含まないbaselineを返すこと。
- [x] **native統合テストを追加する。** 既存3層モデルで内部scope `0` と `2` を使い、進捗0/0.5/1で、選択区間だけを持つ単一区間モデルとの出力一致をfused/unfused双方で確認する。非重複scope・共通scope・gapを含める。33層の小型fixtureでは `30-32` / `0-2` の登録先が6層、wrapperが各種1つであることを検証する。
- [x] **RED確認：** `python -B -m unittest tests.unit.test_runtime -v`。第2区間が実行されない、または優先制御がないことによる失敗を確認する。
- [x] **登録処理を拡張する。** 第1scopeを従来どおりparseし、第2scopeはON時だけparseする。和集合の各blockに1つだけreplacementを登録し、所属する区間番号をcontextへ渡す。重複AFM拒否・off/zero bypass・元モデルの保護・sampler制限を維持する。
- [x] **適用判定を変更する。** 最初に `active_interval` を決め、次に `interval_ids` への所属を確認する。現在のblockが第1区間にないことを理由に第2区間へフォールスルーさせない。既存branch選択と併せてedit / observe / skippedを決める。
- [x] **GREEN確認：** `python -B -m unittest tests.unit.test_intervals tests.unit.test_runtime tests.unit.test_attention tests.unit.test_spectral -v`。native環境では `python -B -m unittest tests.integration.test_native_hooks tests.integration.test_sampler_contract -v` も実行し、スキップの有無を記録する。
- [x] **変更をレビューしてコミットする。** 推奨名：`feat: select AFM blocks by prioritized sampling interval`。

## Task 3: ノード入力・旧ワークフロー・smokeヘルパーの互換性

**Files:** Modify `nodes.py`、`tools/smoke_contract.py`、`tests/unit/test_repository_contract.py`、`tests/unit/test_smoke_contract.py`、`tests/integration/test_native_hooks.py`。

**Interfaces:**
- Task 1の4フィールドと同名・同値の入力を既存入力列の末尾に追加する。
- `execute` の末尾に `two_interval_mode=False, second_block_scope="0-2", second_start_percent=0.5, second_end_percent=1.0` を追加する。
- `workflow_config(workflow) -> AFMConfig` / `mode_config(config, mode) -> AFMConfig` の契約を維持する。

- [x] **失敗する互換性テストを追加する。** `test_new_inputs_are_optional_and_appended` で旧入力列が新列の完全なprefixであること、新4入力の既定値・optional・block選択肢が一致することを確認する。`test_old_api_omits_second_interval` では追加フィールドのない入力がOFF設定となることを確認する。
- [x] **smoke設定の保存テストを追加する。** `test_two_interval_settings_survive_mode_variants` で4フィールドがoff/zero/observe/editの全モードへ引き継がれることを検証する。ONで第2blockが `all` / `0` ならノード選択肢違反として拒否、正規の3選択肢なら受理。OFFでは第2blockを解析・適用しないことも確認する。
- [x] **RED確認：** `python -B -m unittest tests.unit.test_repository_contract tests.unit.test_smoke_contract -v`。新入力・検証の未実装による失敗を確認する。
- [x] **スキーマと実行入口を実装する。** 新4入力を `optional=True` で末尾追加する。表示名は「2区間モード」「第2区間の効果」「第2区間の開始」「第2区間の終了」。第2区間の3設定は `advanced=True`。開始・終了は0–1、刻み0.01。既存 `AFMConfig(...)` 生成は全てキーワード引数へ変える。
- [x] **説明とsmoke検証を更新する。** tooltipに「重複時は第1区間を優先」「第1区間が0–1なら第2区間は動作しない」を記載する。`workflow_config` はON時に第2blockも選択肢に照合する。`run_gpu_smoke.py` は既存の設定読込・mode変換を利用し、不要な変更を加えない。
- [x] **GREENとAPI検証：** 上記CPUテスト、nativeスキーマテストを実行する。ComfyUIのprompt validationを通して旧入力のみのAPIを受理できることを確認する。Pythonの `execute` 直接呼び出しだけでAPI互換性を証明したことにしない。
- [ ] **UI往復確認：** 旧 `examples/moss_afm_edit_ui.json` を読み込み、既存widget値と配線が保持されることを確認する。ONにして例の2区間を設定し、保存・再読込・API出力で全4値が保たれることを確認する。環境不足なら未検証として記録する。
- [x] **変更をレビューしてコミットする。** 推奨名：`feat: expose optional two-interval AFM controls`。

## Task 4: 使用例・全体検証・完了記録

**Files:** Modify `tools/build_examples.py`、`tests/unit/test_repository_contract.py`、`tests/unit/test_runtime.py`、`README.md`、`docs/implementation-status.md`。Create `examples/moss_afm_two_intervals_api.json`。

**Interfaces:**
- `python tools/build_examples.py --workflow examples/moss_afm_baseline_api.json --output <directory>` から、既存3種類に加えて `moss_afm_two_intervals_api.json` を出力する。
- 新例は第1 `30-32 / 0.0–0.5`、第2 `0-2 / 0.5–1.0`、ON、edit、strength 0.05。モデル・seed・25ステップ・CFG等の生成条件は元例を維持する。

- [x] **失敗する例生成テストを追加する。** `test_two_interval_example_preserves_sampling_inputs` で新例の上記値、loader→AFM→KSamplerの接続、元例の生成入力の保持を検証する。
- [x] **ログ・状態回帰テストを追加する。** `test_two_interval_run_logs_config_and_resets` でrun_startに新4設定が記録され、run_summaryのstep/block/actionが選択結果と一致すること、2回の生成でstepが0に戻りrun_idが別であること、例外後も元のmodel_optionsに戻ることを確認する。既存ログの必須キーとCounter形式は維持する。
- [x] **RED確認：** `python -B -m unittest tests.unit.test_repository_contract tests.unit.test_runtime -v`。未生成の例と不足する挙動を確認する。
- [x] **例と説明を更新する。** 生成ツールで新例を作り、READMEにON/OFF、重複優先、gap、共有strength、全体進捗、25ステップ時の13/12分割を記載する。既存1区間例の意味を変えない。第1が0–1の場合の第2区間遮蔽も明記する。
- [x] **全体検証を実行する。** 下記のコマンド・手動確認を行い、実行環境・成功件数・skip・未実行を記録する。
- [x] **差分レビューと完了記録を行う。** `docs/implementation-status.md` に今回の実測結果を追記する。周波数処理への差分がないこと、依存追加がないこと、対象外の機能が混入していないことを確認してコミットする。推奨名：`docs: document and verify two-interval AFM scheduling`。

## 検証コマンドと成功条件

リポジトリ直下で、Python 3.12以上とPyTorchを持つ環境を使う。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m pytest -q -p no:cacheprovider
python -m compileall -q afm tools tests nodes.py __init__.py
git diff --check
```

成功条件：新旧CPUテストが成功し、構文・差分チェックにエラーがない。
nativeテストがskipした場合は、CPU成功とnative未検証を分けて記録する。
`pyproject.toml` に現時点でtypecheck専用コマンドはない。利用可能な既存環境にRuffがある場合は
`python -m ruff check afm/config.py afm/adapter.py nodes.py tools/smoke_contract.py tools/build_examples.py tests/unit/test_intervals.py tests/unit/test_runtime.py tests/unit/test_repository_contract.py tests/unit/test_smoke_contract.py tests/integration/test_native_hooks.py` も実行する。
新たな検査依存関係は追加しない。

native検証は、既存ComfyUI Python環境と実在する `MOSS_COMFY_ROOT` を指定して実行する。
全nativeテストでskipがなく成功したことを確認する。fixtureだけの成功と実画像生成は区別する。

利用可能な実機環境では、既存モデル名を解決した上で次を実行する。

```powershell
python tools/run_gpu_smoke.py --comfy-root "$env:MOSS_COMFY_ROOT" --workflow examples/moss_afm_two_intervals_api.json --output test-results/two-intervals --width 1024 --height 1024 --steps 25 --modes baseline,off,zero,observe,edit
```

成功条件：off/zero/observeがbaselineとbitwise一致、editは有限値でfallbackが0。
標準例の25ステップでは、summary内のedit対象blockがstep 0–12では30/31/32、13–24では0/1/2であること。
CFG・モデル評価回数によりcallsの件数は変わるため、固定の総件数ではなくstep/block/actionで判定する。
既存のGPU helperはこの区間別block条件を自動検証しないため、JSONLを別途確認して検証記録へ残す。

## 完了判定・リスクへの対応

| リスク | 対応・受け入れ条件 |
|---|---|
| blockごとに優先順位を判断して第2区間が漏れる | 時間だけで区間を先に選び、異なるscopeの重複テストを必須化 |
| ONにするだけでは第2区間が動かない | 第1の従来値を勝手に変えず、tooltip・README・2区間例で設定を明示 |
| 新入力追加で旧UI/APIを壊す | optional・末尾追加・旧例のprompt validationとUI往復で確認 |
| 区間ごとに周波数進捗をリセットする | 編集関数に渡る全体進捗をテストし、spectral/attentionの既存テストを維持 |
| 実機環境がなく見かけ上だけ完了とする | CPU・native・UI・GPUの各結果を区別し、未検証項目を最終報告に残す |

実装順はTask 1→2→3→4。中心の3ファイルが密に関連するため、同時編集を避けた逐次実装を基本とする。
停止条件は、合意仕様の実装、実行可能な検証、実行できなかった確認の明示、差分レビューと文書更新が揃った時点。
計画作成時点では、ここに列挙した新機能のテスト・UI確認・GPU検証は未実行。

## 実装時の確認記録（2026-10-28）

- [x] Task 1：設定、入力検証、優先区間の純粋ロジック。未実装時の失敗と実装後の成功を確認。
- [x] Task 2：scopeの和集合、区間に応じた適用、全体進捗、CFG/fallback、native出力の一致。
- [x] Task 3：末尾へのoptional入力追加、旧・新payloadのComfyUI prompt validation、smoke設定の引き継ぎ。
- [x] Task 4の例・ログ確認：新API例の生成、設定保持、run間の状態分離、README更新。
- [x] 全体テスト：57 passed / 45 subtests passed / skipped 0。構文・差分チェック成功。
- [x] 独立レビュー：指摘0件。レビュー側でも全57テストの成功を確認。
- [ ] ブラウザUIの保存・再読込：未実行。スキーマ順序・API検証とは区別する。
- [ ] Python 3.12環境での再実行：未実行。今回の実行環境は3.10.11。
- [ ] 実モデルGPU smoke：未実行。今回のPyTorchはCPU版。

実装時の補正：smokeヘルパーの既定値は512×512・3ステップのため、上の25ステップ検証コマンドに
`--width 1024 --height 1024 --steps 25` を明示した。ヘルパー自体の既定値は変更しない。
詳細な検証環境と結果は [implementation-status.md](../../implementation-status.md) を参照。
