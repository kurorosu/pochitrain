# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- テスト品質に関する改善を行った ([#308](https://github.com/kurorosu/pochitrain/pull/308)).
  - `model_loading.py` の専用ユニットテストを追加した.
  - `json_utils.py` の専用ユニットテストを追加した.
  - エンドツーエンド訓練フローの統合テストを追加した.

### Changed
- テスト品質に関する改善を行った ([#308](https://github.com/kurorosu/pochitrain/pull/308)).
  - `test_layer_wise_lr_validation_error` のテスト名とアサーションを実際の振る舞いに合わせて修正した.
  - `test_core` の重複フィクスチャ (`trainer`, `logger`) を `conftest.py` に共通化した.
  - テスト全体の一時ディレクトリ作成方法を `tempfile.TemporaryDirectory` から `tmp_path` に統一した.

### Fixed
- なし.

### Removed
- なし.

## v1.7.3 (2026-03-14)

### 概要
- コード品質・設計パターンの改善と mypy エラーの全件解消を行ったパッチリリースです.

### Added
- なし.

### Changed
- コード品質・設計パターンを改善した ([#305](https://github.com/kurorosu/pochitrain/pull/305)).
  - `Evaluator.calculate_accuracy` の戻り値型を `Dict[str, float]` から `Dict[str, Any]` に修正した.
  - `TrainingLoop.run()` の引数16個を `TrainingContext` dataclass に集約した.
  - `PochiImageDataset.get_class_counts` を `Counter` ベースに書き換え, 計算量を O(n*m) から O(n) に改善した.
  - `TrainingConfigurator._build_optimizer` を if-elif チェーンから辞書マッピング + `functools.partial` 方式に変更し, `_build_scheduler` とスタイルを統一した.
  - `pochitrain/` 配下の全ファイルで typing の `Dict`, `List`, `Tuple` を built-in 型 (`dict`, `list`, `tuple`) に統一した.
- mypy エラーを全件解消した ([#305](https://github.com/kurorosu/pochitrain/pull/305)).
  - `pyproject.toml` の mypy 設定に `benchmark_runs/` と `work_dirs/` の除外を追加した.
  - `DefaultParamSuggestor` / `LayerWiseLRSuggestor` の `suggest()` 引数型を `optuna.Trial` から `optuna.trial.BaseTrial` に変更した.
  - `create_calibration_dataset` の戻り値型を `Dataset[Any]` から `PochiImageDataset | Subset[Any]` に変更した.

### Fixed
- なし.

### Removed
- なし.

## Archived Changelogs

- [`changelogs/1.7.x.md`](./changelogs/1.7.x.md)

- [`changelogs/1.6.x.md`](./changelogs/1.6.x.md)
- [`changelogs/1.5.x.md`](./changelogs/1.5.x.md)
- [`changelogs/1.4.x.md`](./changelogs/1.4.x.md)
- [`changelogs/1.3.x.md`](./changelogs/1.3.x.md)
- [`changelogs/1.2.x.md`](./changelogs/1.2.x.md)
- [`changelogs/1.1.x.md`](./changelogs/1.1.x.md)
- [`changelogs/1.0.x.md`](./changelogs/1.0.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
