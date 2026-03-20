# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- なし.

### Changed
- なし.

### Fixed
- なし.

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
