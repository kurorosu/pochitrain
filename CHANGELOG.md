# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- なし.

### Changed
- 推論テストの重複ケースを共通基底テストへ集約し, runtime 固有差分を ONNX/TRT/PyTorch 側へ分離した ([#N/A.](https://github.com/kurorosu/pochitrain/pull/N/A.)).
  - `test_base_inference_service.py` を追加し, `IInferenceService` の共通ロジック検証を一元化した.
  - `test_infer_trt.py` を実運用に近いワークスペース構造で再構成し, 状態ベースの導線検証へ整理した.
  - `test_pochi_config.py`, `test_sub_configs.py`, `test_pochi_trainer.py` の同種検証を `parametrize` 化し, 重複ケースを削減した.

### Fixed
- なし.

### Removed
- なし.

## v1.7.0 (2026-02-25)

### 概要
- ベンチマーク機能の CLI 統合, 推論オーケストレーションの共通化, `pin_memory` 設定の学習/推論分離, および本番コード全体のリファクタリングを行ったリリースです.

### Added
- ベンチマーク機能を `bench` 独立コマンドとして CLI 登録し, `uv run bench --suite base` で実行可能にした ([#272](https://github.com/kurorosu/pochitrain/pull/272)).
  - `tools/benchmark/` のモジュール群を `pochitrain/benchmark/` へ移動し, 絶対インポートに統一した.
  - `pochitrain/cli/bench.py` にエントリポイントを作成し, `pyproject.toml` の `[project.scripts]` に登録した.
  - `tools/benchmark/suites.yaml` を `configs/bench_suites.yaml` へ移動した.
- ベンチマーク実行基盤で `pytorch` runtime を追加し, `base` スイートから `pochi infer` を実行できるようにした ([#261](https://github.com/kurorosu/pochitrain/pull/261)).
  - `tools/benchmark/suites.yaml` の `model_paths` と `cases` に `pytorch` を追加した.
  - `tools/benchmark/loader.py` と `tools/benchmark/runner.py` を更新し, runtime 検証とコマンド分岐を対応した.
- `pochi infer` に `--benchmark-json`, `--benchmark-env-name`, `--pipeline` を追加し, ベンチマーク結果の JSON 出力に対応した ([#261](https://github.com/kurorosu/pochitrain/pull/261)).
  - `build_pytorch_benchmark_result` を追加し, ONNX/TRT と同じ `benchmark_result.json` スキーマへ統一した.

### Changed
- `.github/ISSUE_TEMPLATE/refactor_request.md` のラベル表記を `refactoring` から `refactor` へ統一した ([#260](https://github.com/kurorosu/pochitrain/pull/260)).
- 推論CLIのオーケストレーション境界を整理し, `run(request) -> result` を3ランタイムで共通化した ([#263](https://github.com/kurorosu/pochitrain/pull/263)).
  - `InferenceRunResult` をランタイム横断の集約型として統一した.
  - パス解決と runtime option 解決の責務を Service 層へ集約した.
  - `pochi infer` の `--output` 指定時の出力挙動を ONNX/TRT と同じルールに統一した.
  - `infer-onnx` と `infer-trt` の CLI から手動の DataLoader / ExecutionRequest / 結果エクスポート処理を削除し, Service 委譲フローに統一した.
- 本番経路で未使用だったメソッドを削除し, 関連テストと README の推論サンプルを現行仕様へ整理した ([#264](https://github.com/kurorosu/pochitrain/pull/264)).
  - `CheckpointStore.load_checkpoint`, `PochiWorkspaceManager.save_image_list`, `PochiWorkspaceManager.get_available_workspaces`, `LoggerManager.reset` を削除した.
  - 削除メソッドに依存するユニットテストを整理し, `PochiPredictor` ベースの推論例へ更新した.
- JSON出力, モデルロード, ベンチマーク結果構築の重複ロジックを統合し, 保守性を改善した ([#267](https://github.com/kurorosu/pochitrain/pull/267)).
  - `json_utils.write_json_file` を追加し, 推論/ベンチ結果のJSON書き出しを共通化した.
  - `model_loading.load_model_from_checkpoint` を追加し, `PochiPredictor` と `OnnxExporter` のモデル読み込みを統一した.
  - `result_builder` の3ランタイム向けビルダーを共通内部ビルダーへ集約した.
- 周辺ユーティリティの冗長実装を整理し, ワークスペース生成と時刻書式の重複を解消した ([#268](https://github.com/kurorosu/pochitrain/pull/268)).
  - `InferenceWorkspaceManager.create_workspace` の重複処理を親クラスへ集約した.
  - 時刻書式を `timestamp_utils` 定数へ集約し, `base_csv_exporter` のハードコードを置換した.
  - 混同行列計算の2系統実装について, 訓練系(Torch Tensor)と推論系(NumPy/list)の使い分け方針をdocstringに明記した.
  - テストコード全体の低価値コメントを整理し, 意図説明コメントのみを残した.
- 本番コード全体の低価値コメント (whatコメント) を整理し, whyコメントのみを残した ([#270](https://github.com/kurorosu/pochitrain/pull/270)).
- Jetson の推論ベンチマーク再現性向上のため, `README.md` と `GPU環境セットアップガイド` に `nvpmodel` / `jetson_clocks` の運用手順を追記した ([#275](https://github.com/kurorosu/pochitrain/pull/275)).
- `pin_memory` 設定を学習と推論で分離し, `train_pin_memory` / `infer_pin_memory` で個別制御できるようにした ([#277](https://github.com/kurorosu/pochitrain/pull/277)).
  - 学習CLIは `train_pin_memory` を `create_data_loaders` へ渡すように変更した.
  - 推論Serviceは `infer_pin_memory` を参照する.
  - 既定値は `True` を推奨. Jetson (統合メモリ) 環境でも `True` の方が E2E 性能が同等以上であることを実測で確認した.

### Fixed
- なし.

### Removed
- 本番未使用の補助メソッドを削除した ([#264](https://github.com/kurorosu/pochitrain/pull/264)).

## Archived Changelogs

- [`changelogs/1.6.x.md`](./changelogs/1.6.x.md)
- [`changelogs/1.5.x.md`](./changelogs/1.5.x.md)
- [`changelogs/1.4.x.md`](./changelogs/1.4.x.md)
- [`changelogs/1.3.x.md`](./changelogs/1.3.x.md)
- [`changelogs/1.2.x.md`](./changelogs/1.2.x.md)
- [`changelogs/1.1.x.md`](./changelogs/1.1.x.md)
- [`changelogs/1.0.x.md`](./changelogs/1.0.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
