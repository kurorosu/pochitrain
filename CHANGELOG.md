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

## v1.8.2 (2026-03-21)

### 概要
- コード品質改善 (バグ修正, テスト追加, 定数化, 型補完) を含むパッチリリースです.

### Added
- なし.

### Changed
- マジックナンバーを定数化した ([#355](https://github.com/kurorosu/pochitrain/pull/355)).
  - `pochi_predictor.py`: ウォームアップ反復回数 `_WARMUP_ITERATIONS = 10`.
  - `epoch_runner.py`: ログ出力バッチ間隔 `_LOG_BATCH_INTERVAL = 100`.
  - `pochi_dataset.py`: Resize スケール倍率 `_RESIZE_SCALE_FACTOR = 1.14`, ColorJitter パラメータ.
- 型アノテーションを補完した ([#356](https://github.com/kurorosu/pochitrain/pull/356)).
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
