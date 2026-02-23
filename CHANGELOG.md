# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- ベンチマーク実行基盤で `pytorch` runtime を追加し, `base` スイートから `pochi infer` を実行できるようにした.
  - `tools/benchmark/suites.yaml` の `model_paths` と `cases` に `pytorch` を追加した.
  - `tools/benchmark/loader.py` と `tools/benchmark/runner.py` を更新し, runtime 検証とコマンド分岐を対応した.
- `pochi infer` に `--benchmark-json`, `--benchmark-env-name`, `--pipeline` を追加し, ベンチマーク結果の JSON 出力に対応した.
  - `build_pytorch_benchmark_result` を追加し, ONNX/TRT と同じ `benchmark_result.json` スキーマへ統一した.

### Changed
- `.github/ISSUE_TEMPLATE/refactor_request.md` のラベル表記を `refactoring` から `refactor` へ統一した ([#260](https://github.com/kurorosu/pochitrain/pull/260)).
- 推論CLIのオーケストレーション境界を整理し, `run(request) -> result` を3ランタイムで共通化した.
  - `InferenceRunResult` をランタイム横断の集約型として統一した.
  - パス解決と runtime option 解決の責務を Service 層へ集約した.
  - `pochi infer` の `--output` 指定時の出力挙動を ONNX/TRT と同じルールに統一した.

### Fixed
- N/A.

### Removed
- N/A.

## v1.6.0 (2026-02-22)

### 概要
- 推論ベンチマークの任意実行基盤を追加し, TensorRT GPU パイプラインの計測バグ修正, ONNX/TRT 依存パッケージの管理と `gpu_non_blocking` 設定を整理したリリースです.

### Added
- 推論ベンチマーク記録用ドキュメントを追加し, `gpu_non_blocking` の実測結果と `pin_memory` 計測テンプレートを整理した ([#249](https://github.com/kurorosu/pochitrain/pull/249)).
  - `pochitrain/docs/benchmark.md` を追加し, Windows/Jetson の計測値と計測日を記録した.
- 推論ベンチマークの任意実行基盤を追加し, 条件定義から実行・集計までを自動化した ([#252](https://github.com/kurorosu/pochitrain/pull/252)).
  - `infer-trt` / `infer-onnx` に `--benchmark-json`, `--benchmark-env-name` を追加し, オプトインで `benchmark_result.json` を出力できるようにした.
  - `tools/benchmark/suites.yaml`, `tools/benchmark/run_benchmark.py`, `loader.py`, `runner.py`, `aggregator.py` を追加し, suite 定義に基づく実行と集計を実装した.

### Changed
- `onnx`, `onnxscript`, `onnxruntime-gpu` を `[dependency-groups] onnx` から `dependencies` へ移動し, `uv sync` のみで ONNX 関連コマンドが利用可能になるようにした ([#244](https://github.com/kurorosu/pochitrain/pull/244)).
  - `[dependency-groups] onnx` グループを削除した.
- ONNX/TRT の `pipeline=gpu` 前処理で使う `gpu_non_blocking` 設定を導入し, 非同期転送の A/B 比較を構成ファイルで切り替え可能にした ([#248](https://github.com/kurorosu/pochitrain/pull/248)).
  - `configs/pochi_train_config.py` に `gpu_non_blocking = True` を追加し, 既定挙動を明示した.
- ベンチマーク出力形式を整理し, 実運用時に不要な生成物を削減した ([#252](https://github.com/kurorosu/pochitrain/pull/252)).
  - 出力先を `benchmark_runs/<suite>_<timestamp>/` に固定し, `.gitignore` で `benchmark_runs/` を無視対象にした.
  - `benchmark_result.json` の時刻を JST (`YYYY-MM-DD HH:MM:SS`) へ統一し, `schema_version=1.0.0` を維持した.
  - `benchmark_summary.csv/json` から `config.py` と重複する設定値を除外し, 性能指標中心の集計に変更した.
  - `execution_manifest.json`, `stdout.log`, `stderr.log` の出力を廃止した.
  - `pochitrain/docs/benchmark.md` に suites 設定, 実行, 再集計, 出力物の運用手順を追記した.

### Fixed
- TensorRT GPU パイプラインの精度低下と計測異常を修正した ([#245](https://github.com/kurorosu/pochitrain/pull/245)).
  - GPU 入力時にデフォルトストリーム待機を追加し, INT8 精度低下を解消した.
  - CUDA Event 計測を TRT 実行ストリームに統一し, 推論時間が異常に短く記録される問題を修正した.

### Removed
- `requirements.txt` から未使用の `scipy>=1.11.0` を削除した ([#244](https://github.com/kurorosu/pochitrain/pull/244)).

## Archived Changelogs

- [`changelogs/1.5.x.md`](./changelogs/1.5.x.md)
- [`changelogs/1.4.x.md`](./changelogs/1.4.x.md)
- [`changelogs/1.3.x.md`](./changelogs/1.3.x.md)
- [`changelogs/1.2.x.md`](./changelogs/1.2.x.md)
- [`changelogs/1.1.x.md`](./changelogs/1.1.x.md)
- [`changelogs/1.0.x.md`](./changelogs/1.0.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
