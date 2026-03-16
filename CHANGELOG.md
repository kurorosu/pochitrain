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

## v1.8.0 (2026-03-16)

### 概要
- TensorBoard 統合, リファクタリング, テスト拡充, ドキュメント修正を含むマイナーリリースです.

### Added
- TensorBoard 統合を実装した ([#325](https://github.com/kurorosu/pochitrain/pull/325)).
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
- `benchmark/` モジュールのテストを追加した ([#330](https://github.com/kurorosu/pochitrain/pull/330)).
  - `models.py`: CaseConfig/SuiteConfig のフィールド保持, frozen 制約のテストを追加した.
  - `utils.py`: configure_logger, タイムスタンプ形式, write_json, to_float のテストを追加した.
  - `loader.py`: バリデーション関数と load_suite_config の正常系/エラー系テストを追加した.
  - `aggregator.py`: パス収集, ケース名抽出, 集計ロジック (平均, 標準偏差, グループ分離, 不正JSON) のテストを追加した.
  - `runner.py`: config パス解決, config コピー, コマンド構築のテストを追加した.
- `epoch_runner.py` のテストを追加した ([#331](https://github.com/kurorosu/pochitrain/pull/331)).
  - 単一バッチ/複数バッチの損失計算, 空 DataLoader の防御的ガード, クラス重み付き損失, 勾配更新のテストを追加した.
- `visualize_gradient.py` のテストを追加した ([#332](https://github.com/kurorosu/pochitrain/pull/332)).
  - CSV 読み込み (正常系, 不正CSV), PNG 出力生成 (timeline, heatmap, statistics, snapshots), CLI ユーティリティ関数のテストを追加した.
- エッジケーステストを追加した ([#333](https://github.com/kurorosu/pochitrain/pull/333)).
  - `early_stopping`: min_delta 境界値 (極小0.0001/極大10.0) のテストを追加した.
  - `pochi_dataset`: 画像1枚データセット, 破損画像ファイルのテストを追加した.
  - `training_configurator`: 学習率0.0, 負の学習率, 極端な層別学習率のテストを追加した.
  - `checkpoint_store`: 存在しないディレクトリへの保存, optimizer=None のテストを追加した.
  - `evaluator`: 空 DataLoader, 単一サンプルの精度計算/混同行列のテストを追加した.

### Fixed
- ドキュメント不整合を修正した ([#334](https://github.com/kurorosu/pochitrain/pull/334)).
  - CLAUDE.md: `pochi infer` コマンドの引数を位置引数に修正した.
  - README.md: `--opset` を `--opset-version` に修正した.
  - README.md: `create_data_loaders` の使用例に `train_transform`/`val_transform` パラメータを追加した.
- CLI の使用例を実装と整合させた ([#335](https://github.com/kurorosu/pochitrain/pull/335)).
  - `pochi.py`: `pochi infer` の epilog を位置引数形式に修正した.
  - `infer_trt.py`: `--pipeline gpu` のデフォルト表記を削除した (実際のデフォルトは `auto`).

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
