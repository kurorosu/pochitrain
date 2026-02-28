# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- なし.

### Changed
- `test_objective.py` の白箱テストを古典派テストへ移行した ([#293](https://github.com/kurorosu/pochitrain/pull/293)).
  - `FakeTrainer` から `last_init_kwargs` / `last_setup_kwargs` による内部呼び出し検証を削除し, 観測可能な出力 (戻り値, trial.reported) を基準とするテストへ再構成した.
- `test_pochi_cli_infer.py` の `_ServiceStub` を簡素化し, 設定ヘルパーの重複を解消した ([#294](https://github.com/kurorosu/pochitrain/pull/294)).
  - `_ServiceStub` から `captured` dict による内部引数検証を削除し, 観測可能な副作用 (ファイル作成, 集計実行有無) のみで検証するテストへ移行した.
  - `_build_config_dict` / `_build_minimal_config` を `test_cli/conftest.py` の `build_cli_config()` に集約した.
- `test_training_loop.py` / `test_training_configurator.py` のプライベートメソッド直接テストを公開 API 経由に移行した ([#295](https://github.com/kurorosu/pochitrain/pull/295)).
  - `_update_best_and_check_early_stop()` の直接テストを `run()` 経由の古典派テストに置き換えた.
  - `_get_layer_group()` / `_build_layer_wise_param_groups()` の直接テストを `configure()` 経由の検証に移行した.
- `test_runtime_adapters.py` の TRT/ONNX 重複テストを統合し, inference 系テストのモックを削減した ([#296](https://github.com/kurorosu/pochitrain/pull/296)).
  - TRT/ONNX `gpu_non_blocking` 重複テストを `@pytest.mark.parametrize` で統合した.
  - `test_base_inference_service.py` の `MagicMock` を `_DummyAdapter` スタブと実 `DataLoader` に置換した.
  - `test_pytorch_inference_service.py` の `create_dataset_and_params` 完全モックを実データテストに移行した.
- テスト間で重複するフィクスチャ・ヘルパーを共通化した ([#297](https://github.com/kurorosu/pochitrain/pull/297)).
  - `logger` フィクスチャを `test_core/conftest.py` に集約した.
  - `SimpleModel` クラスを `test_onnx/conftest.py` に集約した.
  - `test_pipeline_consistency.py` と `test_calibrator.py` のデータ生成ヘルパーを `create_dummy_dataset` フィクスチャに統合した.
  - `test_convert_cli.py` の重複 `monkeypatch.setattr` を `_patch_convert_deps` ヘルパーに集約した.
  - `test_export_onnx_cli.py` の `FakeOnnxExporterVerifyFail` テストを `run_export` フィクスチャ経由に統合した.
- `test_tensorrt/` のスタブ密結合と `test_infer_onnx.py` の内部メソッド patch を改善した ([#298](https://github.com/kurorosu/pochitrain/pull/298)).
  - `TestResolveIoBindings` の6テストと `TestResolveDynamicShape` の7テストを `@pytest.mark.parametrize` で集約した.
  - `test_infer_onnx.py` の `set_input_gpu` 内部メソッド patch テストを削除した.

- `pochi_dataset.py` の Transform 検証ロジックと Dataset `__getitem__` の重複を排除した ([#299](https://github.com/kurorosu/pochitrain/pull/299)).
  - `_PIL_ONLY_TRANSFORMS` をモジュールレベル定数に集約した.
  - PIL専用 transform の検出ロジックを `_check_pil_transform` に抽出した.
  - `GpuInferenceDataset.__getitem__` を削除し, `FastInferenceDataset.__getitem__` の実装を再利用するようにエラーハンドリング処理を統一した.

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
