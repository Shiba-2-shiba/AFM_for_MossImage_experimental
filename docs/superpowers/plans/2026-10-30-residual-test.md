# Residual Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全42ブロックを3個刻み、全14倍率・既定値1.0で調整する別testノード。

**Architecture:** 名前付き倍率を持つ`ResidualTestConfig`を追加し、既存のgate倍率処理を共有。
AFMが受け付ける既知設定型へ追加し、V3ノードから利用する。

**Tech Stack:** Python、PyTorch、既存ComfyUI V3。

**Spec:** [設計](../specs/2026-10-30-residual-test-design.md)

## Global Constraints

- 0–41を重複・欠落なしで3個ずつ、全14入力、default=1.0、min=0、max=2、step=0.01。
- 既存ノードのスキーマ・計算・IDを変更しない。追加依存なし。
- 現在のインストール先へ直接実装する。ユーザーの自律実行指示により確認待ちは挟まない。

## Review Focus

- 先頭・末尾を含む全区間の入力が正しい3ブロックだけへ届く。
- 全1.0で追加パッチなし、元MODELと出力が変わらない。
- NaN/Inf・範囲外はAPIと実行時で拒否する。
- AFM重複区間で二重実行せず、Residualだけの区間も維持する。
- 新旧Residualの重複・AFM後の追加・不整合なパッチを引き続き拒否する。

## Task 1: ノードと残差設定

**Files:** `afm/residual.py`、`afm/adapter.py`、`nodes.py`、`tests/unit/test_residual.py`、
`tests/integration/test_native_hooks.py`、`README.md`。

**Interfaces:** `ResidualTestConfig(blocks_00_02=1.0, …, blocks_39_41=1.0)`、
`bypass`、`multiplier_for(index)`。`MossImageResidualTest.execute(model, **multipliers)`。

- [x] 区間独立性・入力検証・V3/API・native参照一致・AFM合成の失敗テストを書く。
- [x] テストを実行して新機能未実装による失敗を確認する。
- [x] 設定、ノード登録、既知型の受理を最小限の差分で実装する。
- [x] 対象テスト、全テスト、native統合、利用可能なlintと差分チェックを実行する。
- [x] READMEに区間・配線・操作例と検証範囲を記載し、差分レビューを行う。
  独立レビュアーはモデル非対応で起動できず、実装担当の確認へ切り替えた。

## 実施記録

- 開始時の作業ツリーはclean。3ブロック刻みは実装担当の判断。
- repo lookup用exploreロールはモデル非対応で起動失敗したため、直接調査へ切り替えた。
- RED: 新設定型・ノードが未実装のため6件失敗。既存の4件は成功。
- 対象の単体/native/APIテスト: 11件成功。
- 通常環境の`python -B -m pytest -q -p no:cacheprovider`: 58件成功、27件skip、
  192 subtests成功。skipは`MOSS_COMFY_ROOT`未設定のnativeテスト。
- ComfyUI環境の`python -B -m unittest discover -s tests -t . -v`: 85件成功、skipなし。
  Pythonは`C:/ComfyUI/.venv/Scripts/python.exe`、
  `MOSS_COMFY_ROOT=C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI`。
  ログは`test-results/residual-test/native-tests.log`。依存ライブラリのSWIG非推奨警告あり。
- 22 PythonファイルのAST構文チェックと、リポジトリ既定設定での`git diff --check`が成功。
  途中の`core.autocrlf=false`指定によるチェックは既存CRLFを空白として誤検出したため、
  設定を上書きせず再確認した。ファイルの改行形式の一括変更は行っていない。
- Ruffは環境に未導入、専用typecheck設定なし。検査用依存の追加は行っていない。
- `code-reviewer`ロールもモデル非対応で起動失敗。独立レビュー済みとは扱わない。
  実装担当が旧スキーマ不変、区間ルーティング、AFMの既知型制限、clone分離、READMEを確認。
- UIでの追加・保存・再読込、実モデルGPU生成、画像の副作用低減／最適倍率は未検証。
  ComfyUIの再起動、コミット、公開は行っていない。
