# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- なし.

### Changed
- マジックナンバーを定数化した ([#355](https://github.com/kurorosu/pochitrain/pull/355)).
  - `pochi_predictor.py`: ウォームアップ反復回数 `_WARMUP_ITERATIONS = 10`.
  - `epoch_runner.py`: ログ出力バッチ間隔 `_LOG_BATCH_INTERVAL = 100`.
  - `pochi_dataset.py`: Resize スケール倍率 `_RESIZE_SCALE_FACTOR = 1.14`, ColorJitter パラメータ.
- 型アノテーションを補完した (`N/A.`).
  - `pochi_dataset.py`: `get_class_counts() -> dict[str, int]`.
  - `directory_manager.py`: `save_dataset_paths(train_paths: list[str], ...)`.
  - `param_group_builder.py`: `layer_params: dict[str, list[torch.nn.Parameter]]`.

### Fixed
- `InputShapeResolver._detect_from_onnx()` の無言例外無視を修正した ([#353](https://github.com/kurorosu/pochitrain/pull/353)).
  - `except Exception: pass` を `logger.debug()` に変更し, デバッグ時にエラー原因を追跡可能にした.

### Tests
- `input_shape_resolver.py` と `int8_config.py` のテストを追加した ([#354](https://github.com/kurorosu/pochitrain/pull/354)).
  - `InputShapeResolver`: CLI入力, 静的/動的 ONNX, onnx 未インストール, 破損ファイル, extract_static_shape のテストを追加した.
  - `INT8CalibrationConfigurer`: 明示パス指定, config 自動検出, calib_data 未指定, パス不在, val_transform 未設定, config 読み込みエラー, キャッシュファイルパスのテストを追加した.

### Removed
- なし.

## v1.8.1 (2026-03-20)

### 概要
- CLI 構造のリファクタリング, 設計改善, バグ修正を含むパッチリリースです.

### Added
- なし.

### Changed
- `TrainingConfigurator` から層別学習率ロジックを分離した ([#337](https://github.com/kurorosu/pochitrain/pull/337)).
  - `training/layer_wise_lr/` パッケージを新設した (`ILayerGrouper`, `ResNetLayerGrouper`, `ParamGroupBuilder`).
  - Strategy パターンにより, 将来の非 ResNet モデル対応が拡張のみで可能になった.
  - `ResNetLayerGrouper`, `ParamGroupBuilder` のユニットテストを追加した.
- デッドコード・未使用公開メソッドを整理した ([#339](https://github.com/kurorosu/pochitrain/pull/339)).
  - `CheckpointStore.save_checkpoint()` 等 7メソッドを private 化した.
  - テストを public API 経由に書き換えた.
- `pochi.py` をサブコマンドごとに分割した ([#341](https://github.com/kurorosu/pochitrain/pull/341)).
  - `cli/commands/` に `train.py`, `infer.py`, `optimize.py`, `convert.py` を分離した.
  - `cli/cli_commons.py` に共有ユーティリティ (`setup_logging`, `create_signal_handler`) を抽出した.
  - `pochi.py` を 956行から 181行に削減した.
- `convert_command` のビジネスロジックをサービス層に分離した ([#342](https://github.com/kurorosu/pochitrain/pull/342)).
  - `tensorrt/input_shape_resolver.py`: ONNX 動的シェイプ検出を CLI から分離した.
  - `tensorrt/int8_config.py`: INT8 キャリブレーション設定の組み立てを CLI から分離した.
  - ベンチマーク JSON 出力の重複を `export_benchmark_json()` に共通化した.

### Fixed
- `metrics_exporter.py` の colormap を層数に応じて自動選択するよう修正した (`N/A.`).
  - 10層以下は `tab10`, 11層以上は `tab20` を使用する.

### Removed
- `PochiTrainer.setup_training_from_config()` を削除した (呼び出し元なし) ([#339](https://github.com/kurorosu/pochitrain/pull/339)).
- `find_best_model()` を削除した (呼び出し元なし) ([#339](https://github.com/kurorosu/pochitrain/pull/339)).
- `EarlyStopping.get_status()` を削除した (呼び出し元なし) ([#339](https://github.com/kurorosu/pochitrain/pull/339)).

## Archived Changelogs

- [`changelogs/1.8.x.md`](./changelogs/1.8.x.md)
- [`changelogs/1.7.x.md`](./changelogs/1.7.x.md)

- [`changelogs/1.6.x.md`](./changelogs/1.6.x.md)
- [`changelogs/1.5.x.md`](./changelogs/1.5.x.md)
- [`changelogs/1.4.x.md`](./changelogs/1.4.x.md)
- [`changelogs/1.3.x.md`](./changelogs/1.3.x.md)
- [`changelogs/1.2.x.md`](./changelogs/1.2.x.md)
- [`changelogs/1.1.x.md`](./changelogs/1.1.x.md)
- [`changelogs/1.0.x.md`](./changelogs/1.0.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
