# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- FastAPI ベースの推論 API サーバーを追加した (NA.).
  - `pochi serve` コマンドでサーバーを起動し, HTTP 経由で画像分類を実行できる.
  - `POST /api/v1/predict` で cv2 キャプチャ画像 (numpy 配列) を受け取り, クラス予測+信頼度を返す.
  - `GET /api/v1/health`, `/model-info`, `/version` のヘルスチェック・情報エンドポイントを提供する.
  - raw / JPEG の画像シリアライズ形式を Strategy パターンで切り替え可能.
- `types-PyYAML` スタブパッケージを追加した ([#362](https://github.com/kurorosu/pochitrain/pull/362)).

### Changed
- なし.

### Fixed
- なし.

### Removed
- なし.

## v1.8.3 (2026-03-23)

### 概要
- CLI 推論モジュールの重複解消とベンチマーク結果構築の統合を含むパッチリリースです.

### Added
- なし.

### Changed
- ベンチマーク JSON 出力の env_name 解決パターンを共通化した ([#358](https://github.com/kurorosu/pochitrain/pull/358)).
  - `resolve_benchmark_env_name()` を `result_exporter.py` に追加し, 3箇所の CLI から呼び出すようにした.
- `infer_onnx.py` と `infer_trt.py` の共通処理を抽出した ([#359](https://github.com/kurorosu/pochitrain/pull/359)).
  - `cli_commons.py` に `run_inference_pipeline()` を追加し, パス解決からエクスポートまでの共通フローを一元化した.
  - ログ初期化を `setup_logging()` に統一した.
  - `infer_onnx.py`: 222行 → 172行, `infer_trt.py`: 222行 → 179行.
- `result_builder.py` のベンチマーク結果構築関数を統合した ([#360](https://github.com/kurorosu/pochitrain/pull/360)).
  - `build_onnx/trt/pytorch_benchmark_result()` 3関数を `build_benchmark_result()` 1関数に統合した.
  - `_resolve_trt_precision()` を `resolve_trt_precision()` としてパブリック化した.

### Fixed
- なし.

### Removed
- `build_onnx_benchmark_result()`, `build_trt_benchmark_result()`, `build_pytorch_benchmark_result()` を削除した ([#360](https://github.com/kurorosu/pochitrain/pull/360)).

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
