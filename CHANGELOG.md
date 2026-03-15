# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- TensorBoard 統合を実装した ([#312](https://github.com/kurorosu/pochitrain/pull/312)).
  - `pochitrain/visualization/tensorboard/` パッケージを追加した.
  - `PochiConfig` に `enable_tensorboard` オプションを追加した.
  - `MetricsTracker` に `TensorBoardWriter` を統合し, loss, accuracy, 学習率を記録するようにした.
  - `tensorboard>=2.14.0` を依存関係に追加した.

### Changed
- 例外処理を改善した ([#326](https://github.com/kurorosu/pochitrain/pull/326)).
  - `pochi_predictor.py` の例外チェーンを `from e` 付きに修正し, デバッグ時のスタックトレースを保持するようにした.
  - `inference_utils.py` の `sys.exit(1)` を `FileNotFoundError` / `RuntimeError` に置き換え, Jupyter 等での利用を可能にした.
  - CLI 側で例外をキャッチしてエラーログ出力後に安全に終了するようにした.
- `PochiPredictor.predict()` の複雑度を削減した ([#327](https://github.com/kurorosu/pochitrain/pull/327)).
  - ウォームアップ, 初回バッチ実行, タイミング計測をヘルパーメソッドに分離した.
- `pochi_dataset.py` の transform フィルタリングロジックの重複を解消した ([#328](https://github.com/kurorosu/pochitrain/pull/328)).
  - 共通の `_filter_transforms()` 関数を抽出し, `build_gpu_preprocess_transform` と `convert_transform_for_fast_inference` から呼び出すようにした.
- `FastInferenceDataset._transform_error_logged` をクラス変数からインスタンス変数に変更した ([#329](https://github.com/kurorosu/pochitrain/pull/329)).

### Tests
- `benchmark/` モジュールのテストを追加した (`N/A.`).
  - `models.py`: CaseConfig/SuiteConfig のフィールド保持, frozen 制約のテストを追加した.
  - `utils.py`: configure_logger, タイムスタンプ形式, write_json, to_float のテストを追加した.
  - `loader.py`: バリデーション関数と load_suite_config の正常系/エラー系テストを追加した.
  - `aggregator.py`: パス収集, ケース名抽出, 集計ロジック (平均, 標準偏差, グループ分離, 不正JSON) のテストを追加した.
  - `runner.py`: config パス解決, config コピー, コマンド構築のテストを追加した.

### Fixed
- なし.

### Removed
- なし.

## v1.7.4 (2026-03-14)

### 概要
- テスト品質, セキュリティ・依存関係, ドキュメント・API設計に関する問題点を改善したパッチリリースです.

### Added
- テスト品質に関する改善を行った ([#308](https://github.com/kurorosu/pochitrain/pull/308)).
  - `model_loading.py` の専用ユニットテストを追加した.
  - `json_utils.py` の専用ユニットテストを追加した.
  - エンドツーエンド訓練フローの統合テストを追加した.
- `work_dirs/.gitkeep` を追加し, clone 直後にディレクトリが存在するようにした ([#309](https://github.com/kurorosu/pochitrain/pull/309)).

### Changed
- テスト品質に関する改善を行った ([#308](https://github.com/kurorosu/pochitrain/pull/308)).
  - `test_layer_wise_lr_validation_error` のテスト名とアサーションを実際の振る舞いに合わせて修正した.
  - `test_core` の重複フィクスチャ (`trainer`, `logger`) を `conftest.py` に共通化した.
  - テスト全体の一時ディレクトリ作成方法を `tempfile.TemporaryDirectory` から `tmp_path` に統一した.
- セキュリティ・依存関係に関する改善を行った ([#309](https://github.com/kurorosu/pochitrain/pull/309)).
  - `pillow` のバージョン制約を `>=10.0.0` に引き上げた.
  - `torch>=2.6.0`, `torchvision>=0.21.0` のバージョン制約を追加した.
  - `requires-python` と `black`/`mypy` の Python バージョン不整合の理由をコメントで補足した.
  - `.pre-commit-config.yaml` の mypy rev を `v1.19.1` に更新し `pyproject.toml` と整合させた.
  - `scikit-learn` のバージョン制約を `>=1.0.0` に引き上げた.
- ドキュメント・API設計に関する改善を行った ([#310](https://github.com/kurorosu/pochitrain/pull/310)).
  - bench コマンドの使用方法を README に追記した.
  - `export-onnx` のオプション名を `--no-verify` から `--skip-verify` に修正した.
  - `enable_gradient_tracking` のコメントを実際の値と整合させた.
  - `pochi_predictor.py` の docstring の型表記を `dict[str, Any]` に修正した.
  - `pochi infer` の `--pipeline` ヘルプを `infer-onnx`/`infer-trt` と書式統一した.
  - `configuration.md` の `device` パラメータ説明を実装に合わせて修正した.
  - Python バージョンバッジを `pyproject.toml` の `>=3.10` に合わせて修正した.
  - `__init__.py` の `__all__` から内部クラスを除外した.

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
