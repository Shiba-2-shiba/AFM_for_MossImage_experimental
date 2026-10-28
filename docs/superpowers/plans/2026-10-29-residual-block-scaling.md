# MOSS Image Residual Block Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. コアの変更箇所を共有するため逐次実装を基本とする。

**Goal:** ネイティブLoRA適用済みMOSS MODELの9～25／34～41を独立したResidual倍率で調整し、既存AFMへ接続できる追加ノードを実装する。

**Architecture:** ModelPatcher.cloneへ対象blockのgate倍率パッチを登録し、既存native forwardを呼ぶ。AFMはこの自作パッチだけを識別して合成する。モデル・LoRAの読込はネイティブノードを使い、重みやモデルファイルを変更しない。

**Tech Stack:** Python 3.12以上、PyTorch、ComfyUI V3 / ModelPatcher、既存unittest・pytest。追加依存関係なし。

**Spec:** [Residual仕様](../specs/2026-10-29-residual-block-scaling-design.md)。根拠E1～E10、受け入れ条件A1～A10はこの仕様を参照。

作成日：2026-10-29。基準HEAD：`f10ab33`。作成時は未実装。現在の実装・検証結果は末尾を参照。

## Global Constraints

- Python 3.12以上、通常の42 joint blocksを持つnative MOSSのテキスト画像生成。
- AFM併用は既存と同じEuler / ER-SDE。HF、compile、skip-cache、multi-GPU、参照画像入力は対象外。
- `MossImageResidual`、`mid_multiplier=1.0`、`late_multiplier=1.0`、0～2、UI刻み0.01。
- block番号は0始まり・両端を含む。対象は9～25／34～41。画像・テキスト、CFG両分岐、全期間へ適用。
- 両倍率1.0は新規パッチを付けない。LoRAは保持し、入力モデル・共有mod・native module・モデルファイルを直接変更しない。
- 既存AFMノードの入力・既定値・3プリセット・2区間の動作を維持。未知パッチの拒否を解除しない。
- 追加依存関係・独自JavaScript・重み保存・汎用グラフ実行器を追加しない。
- 以下のチェック欄は実装時に根拠を確認して更新する。今回の成果物は仕様書と計画書のみ。

## Review Focus

1. LoRAが複数／出力投影以外にも掛かる場合：既存patchesを保持してからResidualを適用する（Task 1/4）。
2. block 24・25でAFMと重なる場合：native blockを1回だけ呼び、両方の効果を維持する（Task 2）。
3. AFM off/observe・区間外・fallbackの場合：比較基準はLoRA＋Residualであり、素のモデルへ戻さない（Task 2/4）。
4. 同じloaderの分岐と反復生成：倍率を共有modへ蓄積せず、別経路や次のrunに漏らさない（Task 1/2）。
5. UI/API入力の異常、逆順、未知patch：範囲外・非対応を明示し、黙ったclampや処理省略をしない（Task 1/3/4）。

## ファイルと責務

| ファイル | 変更 | 責務・根拠 |
|---|---|---|
| `afm/residual.py` | 新規 | frozen設定、範囲選択、mod変換、ResidualBlockPatch、install（E1～E4） |
| `afm/adapter.py` | 変更 | 既知Residualの検証とAFM合成。既存native検出を再利用（E4） |
| `afm/runtime.py` | 変更 | Residual有効時のrun_start設定記録。既存run lifecycleは維持（E5） |
| `nodes.py` | 変更 | 新V3ノードとget_node_listへの追加。AFMクラスは既存位置を維持（E6） |
| `tests/unit/test_residual.py` | 新規 | 数値・範囲・mod・非破壊性のテスト |
| `tests/integration/test_native_hooks.py` | 変更 | 既存fixtureを使ったnative、LoRA、AFM合成、V3/API検証（E7） |
| `tests/unit/test_runtime.py`、`tests/unit/test_repository_contract.py` | 変更 | fallback/ログ、classを限定したschema検証、例の回帰 |
| `tools/smoke_contract.py`、`tools/run_gpu_smoke.py` | 変更 | 対応するMODEL配線を解析し、LoRA→Residualをbaselineから保持（E8） |
| `tests/unit/test_smoke_contract.py` | 変更 | 配線解析、順序、baseline、非対応の拒否 |
| `tools/build_examples.py`、`examples/moss_residual_afm_api.json` | 変更／新規 | 既存例を維持してResidual付きAPI例を追加（E9） |
| `examples/moss_lora_residual_afm_api.json` | 条件付き新規 | 実在LoRAを解決できた場合のみ具体的なMODEL-only LoRA例を生成 |
| `tests/unit/test_release.py`、`README.md`、`docs/implementation-status.md` | 変更 | 配布内容、操作方法、今回の検証結果。既存履歴を上書きしない |

`afm/config.py`、`afm/attention.py`、`afm/spectral.py`の計算変更は不要。
`__init__.py`とrelease builderは現行の拡張登録・afmディレクトリ収録を利用する。

## Task 1: Residualの数学的契約とMODELパッチ

**Files:** Create `afm/residual.py`、`tests/unit/test_residual.py`。Modify `tests/integration/test_native_hooks.py`。

**Interfaces:**
- `ResidualConfig(mid_multiplier: float = 1.0, late_multiplier: float = 1.0)`：frozen dataclass。`bypass: bool`、`multiplier_for(index: int) -> float`。
- `RESIDUAL_CONFIG_KEY = "moss_residual_config"`。非identity時だけtransformer_optionsへ設定。
- `scale_residual_mod(mod: tuple[torch.Tensor, ...], multiplier: float) -> tuple[torch.Tensor, ...]`：6要素の2/5だけを新tensorへ置換。
- `ResidualBlockPatch(block_index: int, multiplier: float).__call__(args: dict, extra: dict) -> dict`：引数を局所コピーし、original_blockを1回呼ぶ。
- `install_residual(model, config: ResidualConfig)`：cloneしたMODELを返す。ComfyUI importは実行時に行う。

- [x] **既存baselineを確認。** 後述CPU/nativeコマンドを実行し、既存失敗・skipを記録。nativeソースの6要素mod、gate位置、replacement引数を確認する。契約が違えば対応可否を調査し、別の近似式へ黙って切り替えない。
- [x] **単体の失敗テストを追加。** `test_fixed_ranges_and_boundaries`：`ResidualConfig(0.75, 1.15)`で8/26/33/42は1、9/25は0.75、34/41は1.15。`test_invalid_multiplier`：bool/文字列/NaN/±Inf/-0.01/2.01を拒否し、0/1/2を受理。`test_modulation_is_local`：元6tensorの値不変、0/1/3/4の参照維持、2/5だけ乗算、original_block呼出し1回、未知のargs項目を保持。`test_bad_mod_contract`：欠損・6要素以外・gate shape不整合を明示エラー。
- [x] **RED確認。** `python -B -m unittest tests.unit.test_residual -v`で新機能欠落による失敗を確認。
- [x] **最小実装。** E4のnative検出を再利用し、42層を確認。循環importを避け、`detect_native_moss`は`install_residual`内で遅延importする。非identityは既存AFM/Residual・未知DiTパッチ・非対応wrapper/cache等を検出して拒否。identityは型確認後cloneのみ。0.75/1.15なら25blockへ登録し、共有native moduleへhook/forward変更を加えない。
- [x] **native検証を追加。** `test_residual_native_noop_and_branch_isolation`：LoRAなし/ありで1.0出力bitwise一致、元MODEL/options/patches不変。`test_residual_matches_projection_scaling`：既存fixtureの42層小型モデルを使い、fused/unfused・0/0.75/1.15/2で、別モデルの出力投影を直接倍率変更した参照と比較する。まずFP32、`rtol=1e-5, atol=1e-6`を初期基準にし、失敗を理由なく閾値緩和しない。`test_residual_lora_is_scaled_after_application`：ネイティブLoRAローダーでsynthetic low-rank差分をo/ffn_downと非出力投影へ適用し、1個/2個のLoRAを含む参照と比較。`test_residual_rejects_invalid_models`：非MOSS・41層・43層・compileを拒否。
- [x] **GREEN確認。** 単体テストとnative統合を実行。共有modを連続2blockへ渡すケースでも倍率が累積しないことを確認。
- [x] **Task 1差分レビュー。** A2～A5/A8、原モデル不変、ComfyUIなしでの単体importを確認。

## Task 2: ResidualとAFMの合成

**Files:** Modify `afm/adapter.py:151,171`、`afm/runtime.py:63`、`tests/integration/test_native_hooks.py`、`tests/unit/test_runtime.py`。

**Interfaces:** 既存の`install_afm(model, config)`と`make_outer_wrapper(...)`を維持。
Task 1の`ResidualBlockPatch`、`ResidualConfig`、`RESIDUAL_CONFIG_KEY`を利用する。
追加の汎用patch registryやサンプラーwrapperは作らない。

- [x] **合成の失敗テストを追加。** `test_residual_afm_overlap_calls_native_once`：42層fixtureでResidual(0.75,1.15)＋AFM 24-26を使い、24/25は合成、26はAFMのみ、34/41はResidualのみを確認。計測用call counterで同一評価内の二重実行を検出。
- [x] **回帰テストを追加。** `test_residual_afm_noop_modes_keep_residual`でAFM off/zero/observeをResidual-onlyとbitwise比較。`test_residual_afm_two_intervals_keep_residual_in_gaps`でAFM 24-26→0-2、境界・gap・重複優先を確認。`test_residual_afm_cfg_and_fallback`でpositive_onlyとfallback時にResidualが両CFGへ残ることを確認。`test_residual_afm_repeated_runs_cleanup`で例外後と2run目の状態分離・ログの設定を確認。
- [x] **拒否テストを追加。** `test_only_known_residual_patches_are_accepted`で未知callable、設定と倍率/位置が食い違うResidualパッチ、未知diffusion wrapper、重複AFM、効果を持つResidualの二重追加/逆順追加を拒否。1.0/1.0のno-op追加は既存パッチを変更しない。
- [x] **RED確認。** native統合とruntimeテストで現在の一律DiT拒否による失敗を確認。
- [x] **AFM登録を修正。** E4の一律拒否を自作Residual型＋設定/登録先整合の検査へ変更。非重複パッチを保持し、重複箇所はAFMのoptionsを設定した後に既存Residualへ委譲する。委譲先は登録時に固定して再帰を防ぎ、`extra['original_block']`は元native blockのまま渡す。
- [x] **ログを拡張。** 有効なResidual設定が存在するrun_startだけに`residual_config`を追加。既存`config`・summaryキーとAFMカウントの意味を変えず、Residual単独ではAFMのrun状態や出力ディレクトリを要求しない。
- [x] **GREEN確認・レビュー。** runtime/nativeと既存attention/spectral/intervalテストを実行し、A6/A7/A8を確認。AFM非対象のtext/CFG行に関する保証は、Residual適用後の同じ呼び出しを基準に説明する。

## Task 3: V3追加ノードとワークフロー互換性

**Files:** Modify `nodes.py:78`、`tests/unit/test_repository_contract.py`、`tests/integration/test_native_hooks.py`。

**Interfaces:** `MossImageResidual.define_schema()`、`MossImageResidual.execute(model, mid_multiplier=1.0, late_multiplier=1.0) -> io.NodeOutput`。
既存`MossAFMExtension.get_node_list()`は`[MossImageAFM, MossImageResidual]`を返す。

- [x] **失敗テストを追加。** `test_residual_v3_schema`でID/category/MODEL入出力・2倍率の名前/既定値/範囲/刻みを固定。`test_existing_afm_schema_is_unchanged`はAST検索を`MossImageAFM`クラス内に限定し、新クラスの追加による誤検出を防ぐ。
- [x] **API経路を追加。** `test_native_lora_residual_afm_prompt_validation`でLoraLoaderModelOnlyとLoraLoaderの両方のMODEL出力→Residual→AFMを検証する。実在する選択肢を使う環境確認と、選択肢をstubした構造テストは区別する。旧AFM payloadと既存例の検証も維持。
- [x] **RED確認。** repository contract/native schemaテストで未登録の新ノードを検出。
- [x] **ノードを追加。** 入力はmodel/mid_multiplier/late_multiplierの順。倍率tooltipに0始まり、1.0=変更なし、Attention＋MLP、LoRA込み・両CFGへの適用を記載。独自JS・動的widget・AFM入力追加は行わない。
- [x] **GREEN確認。** スキーマとComfyUI prompt validationを実行。Pythonのexecute直接呼出しだけでAPI受付の成功としない。
- [ ] **UI確認。** 検索→追加→LoRA→Residual→AFMの接続→倍率0.75/1.15で保存→再読込→API出力を確認。既存AFM UI例の値と配線も保持。UIが使えない場合は未実行と理由を記録。

## Task 4: 実経路を検証するsmokeと使用例

**Files:** Modify `tools/smoke_contract.py`、`tools/run_gpu_smoke.py`、`tools/build_examples.py`、`tests/unit/test_smoke_contract.py`、`tests/unit/test_repository_contract.py`。Create `examples/moss_residual_afm_api.json`。実在LoRAが解決できる場合はLoRA付き例も作成。

**Interfaces:**
- `ModelOnlyLoraSpec(lora_name: str, strength_model: float)`と`ModelPipeline(loras: tuple[ModelOnlyLoraSpec, ...], residual: ResidualConfig | None, afm: AFMConfig)`を`tools/smoke_contract.py`へ追加。
- `workflow_pipeline(workflow: dict) -> ModelPipeline`：既存KSampler 31からMODELリンクを遡る。`UNETLoader 28 → LoraLoaderModelOnly* → Residual? → AFM? → KSampler 31`に限定。ノードIDは中間ノードについて固定しない。
- 既存`workflow_config(workflow) -> AFMConfig`は`workflow_pipeline(...).afm`を返し、旧呼出しと既定値を維持。
- 例生成CLIへ`--model-lora NAME STRENGTH`（繰り返し可）を追加。既存4例には影響させず、新LoRA例だけへロード順で挿入。

smoke専用parserはMODEL-only LoRAに限定する。通常LoraLoaderから出るMODELは新ノード本体で受理するが、
CLIPも変更する経路の実生成確認は通常のComfyUI API/UIで行う。helperへ通常LoraLoaderを渡した場合、
CLIPを無視して走らせず、対応経路と通常API/UIでの確認方法を示す明示エラーにする。

- [x] **配線の失敗テストを追加。** `test_model_pipeline_preserves_lora_order`でLoRA2個の順序・名前・強度、Residual0.75/1.15、AFM設定を保持。`test_pipeline_rejects_unsupported_links`で循環・欠損node・出力index不正・逆順・重複・未知node・リンク化されたwidget入力を拒否。既存loader直結/loader→AFMを維持。
- [x] **baselineの失敗テストを追加。** `test_smoke_baseline_keeps_lora_and_residual`で全modeの入力モデルに同じLoRA/Residualが残り、baselineだけAFMを付けないことを確認。新経路でResidualを省略すると比較テストが失敗することも確認。
- [x] **RED確認。** `python -B -m unittest tests.unit.test_smoke_contract tests.unit.test_repository_contract -v`。
- [x] **parserと実行を拡張。** loader→ネイティブLoRA適用→install_residualまでを共通モデルとして構築し、その後に各AFM modeを適用。LoRAはfolder_pathsの既存登録先で解決し、モデル保存を行わない。reportにLoRA順序/強度、Residual設定、実ファイル識別情報と環境を記録する。
- [x] **例を生成。** 新Residual例は0.75/1.15、AFM edit/strength0.05/24-26として重複blockを通す。元例のseed・prompt・sampler・scheduler・steps・CFG・VAE等を保持。LoRA例はnative `LoraLoaderModelOnly`で、実在する名前を確認できた場合のみ生成。未確認の架空名を実行可能例として配布しない。
- [x] **GREEN確認。** parser、mode生成、例、既存例の回帰を実行。reportだけでなく、LoRA適用・Residual登録を実際に呼んだこともテストで確認。
- [x] **実機matrixを実行。** 下表の生成とlatent/AFMログ比較を行う。実LoRAのkeyが未適用なら成功扱いせず、互換性またはマッピングを調査する。

| 入力モデル | Residual倍率 | AFM条件 | 判定 |
|---|---|---|---|
| 元MOSS／MOSS＋LoRA | 1.0/1.0 | baseline/off/zero/observe | Residual追加前とbitwise一致 |
| 元MOSS／MOSS＋LoRA | 0.75/1.15 | baseline/off/zero/observe | 同じLoRA＋Residualのbaselineとbitwise一致 |
| MOSS＋LoRA | 0.75/1.15 | edit 24-26 | 有限出力、AFM edit呼出しあり、fallback 0、重複箇所で二重実行なし |
| MOSS＋LoRA | 0.75/1.15 | edit 0-2 / 30-32 / 2区間 | 既存範囲・時間選択とResidual両立 |

少なくともINT8実モデル＋互換実LoRAで実行する。複数LoRAはnative小型テストを必須とし、
実ファイルも2個利用できれば同じmatrixの代表条件で確認する。
FP32小型参照の一致とINT8生成成功は別の証拠として記録する。
参照の調整済みファイルと比較する場合、LoRAの適用順が違う条件を同値比較に使わない。

## Task 5: 文書・配布・完了検証

**Files:** Modify `README.md`、`docs/implementation-status.md`、`tests/unit/test_release.py`。

- [x] **配布テストを追加。** 生成ZIPに`afm/residual.py`と新API例が入り、参考フォルダーやモデル/LoRAファイルは入らず、同じソースから同一ZIPができることを確認。
- [x] **操作説明を追加。** native loader/LoRA→Residual→AFM、2つの倍率、0始まり、0/1の意味、両CFG・全期間への作用、AFMとの相互作用、対応順序、0.75/1.15が例であることを記載。通常LoraLoaderのCLIP配線とsmokeのMODEL-only制限を区別する。
- [x] **全体検証を実行。** 下記コマンドとA1～A10の対応表を使い、成功/失敗/skip/未実行を記録。既存結果を今回の結果として流用しない。
- [x] **差分レビュー。** weight/forwardの恒久変更なし、既存AFM入力・数式・区間優先の維持、未知patchの拒否、既存API例の保持、追加依存なしを確認。
- [x] **完了記録。** `docs/implementation-status.md`にPython/ComfyUI/PyTorch/GPU、テスト件数、実モデル/LoRA、UI・GPUの証拠と不足を追記。別の検証済み条件からINT8＋LoRA＋Residual＋AFM全体の成功を推定しない。

## 検証コマンド

既存のPython 3.12以上・PyTorch環境を使う。検査ツールを追加インストールしない。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m unittest tests.unit.test_residual tests.unit.test_runtime tests.unit.test_smoke_contract tests.unit.test_repository_contract -v
python -B -m pytest -q -p no:cacheprovider
python -m compileall -q afm tools tests nodes.py __init__.py
git diff --check
```

native検証には実在するComfyUI checkoutを指定する。下記は調査時点で存在した環境の例であり、
実装コードやテストへパスを埋め込まない。成功条件はnative testsがskipなしで通ること。

```powershell
$env:MOSS_COMFY_ROOT='C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI'
& 'C:/ComfyUI/.venv/Scripts/python.exe' -B -m unittest tests.integration.test_native_hooks tests.integration.test_sampler_contract -v
```

改修後のGPU helperは、同じ入力経路をbaselineから保持して使う。
まず512×512・3steps、通過後に1024×1024・25stepsで代表条件を確認する。

```powershell
& 'C:/ComfyUI/.venv/Scripts/python.exe' tools/run_gpu_smoke.py --comfy-root "$env:MOSS_COMFY_ROOT" --workflow examples/moss_lora_residual_afm_api.json --output test-results/residual-lora-smoke --width 512 --height 512 --steps 3 --modes baseline,off,zero,observe,edit
```

このコマンドのLoRA例はTask 4で実在LoRAを指定して生成してから実行する。
各条件で出力先を分け、seed・CFG等を揃える。ログはResidual倍率、AFM block/step/action、
fallback 0を確認する。画像が違うことだけで実装正当性や画質改善を証明したことにしない。

Ruffが既存環境にある場合は変更Pythonファイルへ実行する。typecheck専用設定は現状ないため、
未設定であることを記録し、この機能のためだけに検査依存を導入しない（E10）。

## 受け入れ条件とタスクの対応

| 条件 | タスク | 主な証拠 |
|---|---|---|
| A1 | 3/5 | V3スキーマ、旧/新prompt validation、旧例差分 |
| A2～A4 | 1 | 範囲境界・共有mod非破壊・native参照一致・1.0 bitwise |
| A5 | 1/4 | synthetic LoRA1/2個、実LoRAロード結果と生成 |
| A6～A7 | 2/4 | 重複呼出し回数、2区間・CFG・fallback・反復run |
| A8 | 1/2/3/4 | 入力・モデル・patch・配線の拒否テスト |
| A9 | 4 | pipeline解析、共通baseline、実経路report |
| A10 | 3/4/5 | UI保存再読込、INT8実生成、環境別検証記録 |

## リスクと停止条件

| リスク | 対応 |
|---|---|
| 最新ComfyUIのmodや参照画像処理が変わる | 使用中のnative契約を先に確認。通常T2Iだけを認定し、非対応の契約は明示エラー |
| gate共有tensorのin-place変更 | 2/5だけ新tensor、他要素保持。連続block・分岐・反復runで検証 |
| AFMがResidualを上書き、または二重実行 | 既知パッチのみ委譲、24/25でcall counter、専用block保持 |
| LoRAが外れる／別順序の計算になる | cloneのpatches保持と、LoRA適用後の独立参照を比較 |
| INT8の丸め差を障害と誤認 | FP32の数学検証とINT8実生成を分離。保存済みモデルの完全一致を要求しない |
| smokeが実際と違う配線を実行 | 対応配線を限定し、未知ノードやCLIP変更を省略せず拒否 |
| 他環境の過去の成功を今回の成功と混同 | 実行時の環境、skip、未実行を分けて記録 |

順序はTask 1→2→3→4→5。現在の作業ディレクトリの未コミット変更を実装開始時に確認し、
他者の変更を戻さない。コミット/公開/ComfyUI再起動を計画作成の一部として実行しない。
将来の実装は適用中のワークスペース権限と実行方針に従う。

完了は合意機能の実装、実行可能な検証、文書・例の更新、差分レビューが揃った時点。
環境不足でnative/UI/GPU検証ができない場合は、その項目を完了扱いせず未検証として明示する。
計画作成時点では未実行だった。現在の結果は以下の実装記録を参照。

## 実装・検証記録（2026-10-29）

- Task 1/2/3のコード、Task 4/5は実装・検証済み。native全79テスト成功、skipなし。
- INT8実モデル＋実LoRAで512pxの5モード、1024px/25stepsのbaseline/editが成功。
  追加7条件も成功し、合計33サンプリング、有限出力、AFM fallback 0を確認。
- LoRA1/2個、倍率1.0、残差単独、AFM全3プリセット、2区間のblock/step選択を確認。
- 独立レビューのImportant 2件は修正・回帰検証済み。詳細は[検証記録](../../residual-validation.md)。
- 計画補正：ComfyUI APIはFLOATを先に数値化するため、元のbool/文字列型の拒否は直接executeの契約とする。
  API側ではnativeの変換後に有限性と範囲を検証する。ComfyUI本体は変更しない。
- 新API例にはnative validatorが必要とするAFM既定値を全て記載した。既存4例は変更しない。
- ユーザーはComfyUI再読み込み後の正常動作を確認済み。エージェントのUI保存・再読込操作は、
  UI接続ホストと最小化ウィンドウの操作制約で未実施。自動確認済みとはしない。
- 指定された作業ツリーで実装した。ユーザーの追加指示によりコミット・pushを行う。
