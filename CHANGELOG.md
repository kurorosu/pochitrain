# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- N/A.

### Changed
- `onnx`, `onnxscript`, `onnxruntime-gpu` を `[dependency-groups] onnx` から `dependencies` へ移動し, `uv sync` のみで ONNX 関連コマンドが利用可能になるようにした.
  - `[dependency-groups] onnx` グループを削除した.
- ONNX/TRT の `pipeline=gpu` 前処理で使う `gpu_non_blocking` 設定を導入し, 非同期転送の A/B 比較を構成ファイルで切り替え可能にした.
  - `configs/pochi_train_config.py` に `gpu_non_blocking = True` を追加し, 既定挙動を明示した.

### Fixed
- N/A.

### Removed
- `requirements.txt` から未使用の `scipy>=1.11.0` を削除した.

## v1.5.0 (2026-02-19)

### 概要
- `PochiConfig` を Pydantic `BaseModel` に移行し, 手動バリデーション層を宣言的バリデーションに置換したリリースです.

### Added
- changelog の分割運用を導入し, 旧履歴を `changelogs/` 配下で管理する構成を追加した ([#237](https://github.com/kurorosu/pochitrain/pull/237)).
  - `changelogs/README.md` と `changelogs/1.0.x.md` から `changelogs/1.3.x.md` を新規作成し, 過去履歴を系列単位で参照できるようにした.
  - changelog の運用ルールを整理し, 最新履歴は `CHANGELOG.md`, 旧履歴は `changelogs/` に置く方針へ統一した.
  - 各変更項目の記法を統一し, 1行目を要点, 2行目をインデント補足とする形式へ揃えた.
- GitHub の Issue Template と PR Template を導入した ([#236](https://github.com/kurorosu/pochitrain/pull/236)).
  - `bug`, `feature`, `refactor`, `test`, `documentation` のテンプレートを追加し, 起票とレビュー記述を標準化した.
- GPU 環境セットアップガイドを追加し, CUDA/cuDNN/TensorRT のインストール手順と検証済み環境を整理した.
  - `pochitrain/docs/gpu_environment_setup.md` を新規作成し, 環境変数設定やトラブルシューティングを含むガイドを提供した.

### Changed
- `PochiConfig` と全サブ設定を `dataclass` から Pydantic `BaseModel` に移行し, 宣言的バリデーションへ統一した ([#241](https://github.com/kurorosu/pochitrain/pull/241)).
  - `Field` 制約, `Literal` 型, `field_validator`, `model_validator` により, 設定値の不正を型レベルで検出できるようにした.
  - CLI の `train`, `infer`, `optimize` 各コマンドで `ValidationError` をキャッチし, エラーメッセージを表示する統一ハンドリングを導入した.
  - `dataclasses.asdict` / `dataclasses.fields` を `model_dump` / `model_fields` に置換し, dataclasses 依存を除去した.
- `pochi infer` の推論ビジネスロジックを `PyTorchInferenceService` へ分離し, CLI を薄いラッパーに整理した ([#238](https://github.com/kurorosu/pochitrain/pull/238)).
  - 推論器生成・DataLoader 作成・入力サイズ検出・結果集約を Service 層に移し, 単体テストを可能にした.
- `PochiTrainer` の訓練フローを `TrainingLoop` と `EpochRunner` へ分離し, 状態管理を改善した ([#235](https://github.com/kurorosu/pochitrain/pull/235)).
  - 訓練ループ責務を分割し, 目的関数とユニットテストの保守性を高めた.
- TensorRT 変換ガイドを `pochitrain/tensorrt/docs/` から `pochitrain/docs/` へ移動し, ドキュメント配置を統合した.
- Optuna 設定ドキュメントをネスト辞書形式 (`optuna = {...}`) に更新し, フラットキー形式の記述を廃止した ([#241](https://github.com/kurorosu/pochitrain/pull/241)).

### Fixed
- N/A.

### Removed
- `pochitrain/validation/` ディレクトリを削除し, Chain of Responsibility パターンによる手動バリデーションを Pydantic の宣言的バリデーションに置換した ([#241](https://github.com/kurorosu/pochitrain/pull/241)).
  - 13のバリデータソースファイルと対応する13のテストファイルを除去し, バリデーションロジックを `PochiConfig` に集約した.

## Archived Changelogs

- [`changelogs/1.4.x.md`](./changelogs/1.4.x.md)
- [`changelogs/1.3.x.md`](./changelogs/1.3.x.md)
- [`changelogs/1.2.x.md`](./changelogs/1.2.x.md)
- [`changelogs/1.1.x.md`](./changelogs/1.1.x.md)
- [`changelogs/1.0.x.md`](./changelogs/1.0.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
