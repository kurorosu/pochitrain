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

## v1.7.1 (2026-02-28)

### 概要
- ONNX GPU 推論の精度劣化バグ修正と, テストコードのリファクタリングを行ったパッチリリースです.

### Added
- なし.

### Changed
- 推論テストの重複ケースを共通基底テストへ集約し, runtime 固有差分の検証へ整理した ([#280](https://github.com/kurorosu/pochitrain/pull/280)).
  - `test_base_inference_service.py` を追加し, `IInferenceService` の共通ロジック検証を一元化した.
  - `test_infer_trt.py` を実運用に近いワークスペース構造で再構成し, 状態ベースの導線検証へ整理した.
  - `test_pochi_config.py`, `test_sub_configs.py`, `test_pochi_trainer.py` の同種検証を `parametrize` 化し, 重複ケースを削減した.
- 古典派テストへの再シフトを進め, CLI分割と重複ケース集約を実施した ([#282](https://github.com/kurorosu/pochitrain/pull/282)).
  - `test_pochi_cli.py` を共通責務へ縮小し, `test_pochi_cli_infer.py` / `test_pochi_cli_train.py` へ物理分割した.
  - `test_convert_cli.py` に成功系の引数伝播検証を追加し, セットアップをヘルパー化した.
  - `test_inference_utils.py` を `parametrize` ベースへ集約し, 同型重複を削減した.
  - ONNX/TRT/PyTorch の service テストを runtime 固有差分のみの最小維持セットへ再編した.

### Fixed
- ONNX `pipeline=gpu` で `gpu_non_blocking=True` 時に精度低下が再現する問題を修正した ([#284](https://github.com/kurorosu/pochitrain/pull/284)).
  - `OnnxInference.set_input_gpu` で入力テンソル参照を保持し, 推論完了まで入力バッファ寿命を保証するようにした.
  - `EngineRuntimeAdapter` から `gpu_non_blocking=True` 時のみ ONNX 入力同期フックを呼ぶ経路を追加し, 転送完了を保証した.
  - ONNX 関連テストを追加し, 入力バッファ寿命管理と同期フック呼び出しを検証可能にした.

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
