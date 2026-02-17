# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- changelog の分割運用を導入し, 旧履歴を `changelogs/` 配下で管理する構成を追加した ([#237](https://github.com/kurorosu/pochitrain/pull/237)).
  - `changelogs/README.md` と `changelogs/1.0.x.md` から `changelogs/1.3.x.md` を新規作成し, 過去履歴を系列単位で参照できるようにした.
  - changelog の運用ルールを整理し, 最新履歴は `CHANGELOG.md`, 旧履歴は `changelogs/` に置く方針へ統一した.
  - 各変更項目の記法を統一し, 1行目を要点, 2行目をインデント補足とする形式へ揃えた.
- GitHub の Issue Template と PR Template を導入した ([#236](https://github.com/kurorosu/pochitrain/pull/236)).
  - `bug`, `feature`, `refactor`, `test`, `documentation` のテンプレートを追加し, 起票とレビュー記述を標準化した.
- GPU 環境セットアップガイドを追加し, CUDA/cuDNN/TensorRT のインストール手順と検証済み環境を整理した.
  - `pochitrain/docs/gpu_environment_setup.md` を新規作成し, 環境変数設定やトラブルシューティングを含むガイドを提供した.

### Changed
- `pochi infer` の推論ビジネスロジックを `PyTorchInferenceService` へ分離し, CLI を薄いラッパーに整理した ([#238](https://github.com/kurorosu/pochitrain/pull/238)).
  - 推論器生成・DataLoader 作成・入力サイズ検出・結果集約を Service 層に移し, 単体テストを可能にした.
- `PochiTrainer` の訓練フローを `TrainingLoop` と `EpochRunner` へ分離し, 状態管理を改善した ([#235](https://github.com/kurorosu/pochitrain/pull/235)).
  - 訓練ループ責務を分割し, 目的関数とユニットテストの保守性を高めた.
- TensorRT 変換ガイドを `pochitrain/tensorrt/docs/` から `pochitrain/docs/` へ移動し, ドキュメント配置を統合した.

### Fixed
- N/A.

### Removed
- N/A.

## v1.4.3 (2026-02-14)

### 概要
- 推論パイプライン拡張, 推論サービス層分離, テストスイート再編を中心に品質と保守性を改善したリリースです.

### Added
- 推論CLIに `--pipeline` 切替を追加し, `current`, `fast`, `gpu` を選択可能にした ([#195](https://github.com/kurorosu/pochitrain/pull/195)).
  - 推論前処理の比較検証をCLI引数だけで切替できるようにした.
- ONNX/TensorRT 推論に `DataLoader` ベース実行を導入した ([#194](https://github.com/kurorosu/pochitrain/pull/194)).
  - 推論対象の読み込みをバッチ処理へ統一し, 実運用時の流量に近い経路を確立した.
- ONNX/TensorRT 推論に `ExecutionService`, `ResultExportService` を導入し, 実行と出力の責務を分離した ([#209](https://github.com/kurorosu/pochitrain/pull/209), [#213](https://github.com/kurorosu/pochitrain/pull/213), [#217](https://github.com/kurorosu/pochitrain/pull/217)).
  - CLIと推論実装の結合を弱め, テストと拡張のしやすさを高めた.
- 推論CLIのパイプライン整合性テストを追加した ([#203](https://github.com/kurorosu/pochitrain/pull/203)).
  - ONNX/TRT間で同条件時に同等のパイプライン解決になることを検証可能にした.

### Changed
- 推論CLIをサービス層中心に再編し, CLIは入出力制御中心へ整理した ([#219](https://github.com/kurorosu/pochitrain/pull/219)).
  - CLI神モジュール化を抑え, 推論オーケストレーションの責務を分離した.
- `gpu_normalize` の `mean/std` 再生成を解消し, 前処理オーバーヘッドを削減した ([#205](https://github.com/kurorosu/pochitrain/pull/205)).
  - バッチごとの同一テンソル再生成を避け, GPU前処理の安定した性能を確保した.
- テストスイート全体を棚卸しし, 重複テスト削減と実経路テスト強化を実施した ([#226](https://github.com/kurorosu/pochitrain/pull/226)).
  - 低価値テストを減らしつつ, 回帰検知力を維持する構成へ再編した.
- `pytest-xdist` 導入と分散モード最適化により, テスト実行時間を短縮した ([#227](https://github.com/kurorosu/pochitrain/pull/227)).
  - `worksteal` と `-n 6` を既定化し, ローカル検証サイクルを短縮した.

### Fixed
- `infer-onnx` の CUDA EP 不可時に, GPUパイプラインがCPUへ正しくフォールバックしない問題を修正した ([#201](https://github.com/kurorosu/pochitrain/pull/201)).
  - GPU指定時でも実行不能環境でクラッシュせず, CPU経路へ継続できるようにした.
- `export-onnx` の `--input-size` に正の整数バリデーションを追加した ([#190](https://github.com/kurorosu/pochitrain/pull/190)).
  - 0や負値による不正入力をCLI段階で早期に検出できるようにした.
- `matplotlib_fontja` 未導入時でも混同行列出力が継続できるよう修正した ([#223](https://github.com/kurorosu/pochitrain/pull/223)).
  - 追加フォント依存の有無にかかわらず, 可視化処理の継続性を確保した.
- 3つの推論CLIで CUDA 計測方式を統一し, 計測値の不整合を修正した ([#202](https://github.com/kurorosu/pochitrain/pull/202)).
  - バックエンド間で比較可能なレイテンシ指標を得られるようにした.

### Removed
- 旧推論出力モジュールの重複責務を整理し, `ResultExportService` へ統合した ([#217](https://github.com/kurorosu/pochitrain/pull/217)).
  - 旧APIの二重メンテナンスを解消し, 出力処理の実装箇所を一本化した.

## v1.4.2 (2026-02-08)

### 概要
- TensorRT 実行パスと CUDA ストリーム計測の整合性を改善し, Jetson 向け依存整備を進めたリリースです.

### Added
- なし.

### Changed
- TensorRT 推論を `execute_v2` から `execute_async_v3` に移行した ([#185](https://github.com/kurorosu/pochitrain/pull/185)).
  - 新しい実行APIに合わせ, 非同期実行前提の推論経路へ更新した.
- 非デフォルト CUDA ストリームへ統一し, 入出力と計測の整合性を改善した ([#185](https://github.com/kurorosu/pochitrain/pull/185)).
  - 実行ストリームと計測ストリームの不一致を減らし, 計測精度を改善した.
- Jetson 向け `requirements.txt` を整備した ([#186](https://github.com/kurorosu/pochitrain/pull/186)).
  - Jetsonで必要な依存の導入手順を揃え, 環境差分の吸収を進めた.

### Fixed
- `pochi convert` の `--input-size` に `positive_int` バリデーションを追加した ([#182](https://github.com/kurorosu/pochitrain/pull/182)).
  - 変換時の入力サイズ不正値を実行前に弾けるようにした.

### Removed
- なし.

## v1.4.1 (2026-02-07)

### 概要
- 動的シェイプ変換と変換CLIバリデーションの不具合修正を中心に, 変換系の安定性を高めたリリースです.

### Added
- なし.

### Changed
- README と conversion ドキュメントの動的シェイプ説明を改善した ([#183](https://github.com/kurorosu/pochitrain/pull/183)).
  - 変換時に必要な入力条件がユーザーに伝わるよう説明を明確化した.

### Fixed
- 動的シェイプ ONNX の INT8 変換時に `--input-size` を適用できるよう修正した ([#174](https://github.com/kurorosu/pochitrain/pull/174)).
  - INT8変換でも空間次元を適切に解決し, 変換失敗を防止した.
- 動的シェイプ ONNX 変換時に H/W が `1` 固定になる問題を修正した ([#175](https://github.com/kurorosu/pochitrain/pull/175)).
  - 実入力サイズに沿った shape 解決へ変更し, 不正なプロファイル生成を回避した.
- TensorRT の単一I/O前提を明示的にガードした ([#180](https://github.com/kurorosu/pochitrain/pull/180)).
  - 非対応のI/O構成を早期検出し, わかりやすい失敗へ変えた.
- `find_best_model` のエポック比較を文字列比較から数値比較へ修正した ([#171](https://github.com/kurorosu/pochitrain/pull/171)).
  - `epoch10` と `epoch2` の比較誤りを防ぎ, 正しい最新モデル選択を保証した.

### Removed
- calibrator の未使用 import を削除した ([#170](https://github.com/kurorosu/pochitrain/pull/170)).
  - 不要依存を削減し, モジュール責務を簡潔にした.

## v1.4.0 (2026-02-07)

### 概要
- ネイティブ TensorRT 推論導入と学習・推論責務分離を実施した, 1.4 系列の基盤リリースです.

### Added
- ネイティブ TensorRT 推論を追加した ([#161](https://github.com/kurorosu/pochitrain/pull/161)).
  - ONNX経由ではなくTensorRTエンジンを直接利用する推論経路を提供した.
- 推論CLIの統一仕様, 出力先管理, 混同行列, クラス別レポートを追加した ([#161](https://github.com/kurorosu/pochitrain/pull/161)).
  - バックエンド差分を減らし, 出力物の比較と運用をしやすくした.
- Early Stopping を追加した ([#160](https://github.com/kurorosu/pochitrain/pull/160)).
  - 改善が止まった時点で訓練を終了できるようにし, 過学習を抑制した.
- `CheckpointStore`, `MetricsTracker`, `Evaluator`, `TrainingConfigurator` を追加した ([#160](https://github.com/kurorosu/pochitrain/pull/160)).
  - 学習処理の副責務を分離し, 構造的な保守性を向上させた.

### Changed
- `PochiTrainer` の責務を段階的に分離し, 保守性を改善した ([#160](https://github.com/kurorosu/pochitrain/pull/160)).
  - 設計を役割単位へ整理し, 変更時の影響範囲を局所化した.
- 推論後処理と出力処理の共通化を実施した ([#160](https://github.com/kurorosu/pochitrain/pull/160)).
  - 予測後の集計と書き出し仕様をバックエンド間で統一した.

### Fixed
- 空 DataLoader 時の平均計算ガードを追加した ([#160](https://github.com/kurorosu/pochitrain/pull/160)).
  - サンプル数ゼロ時の例外を防ぎ, 学習ループの安全性を高めた.

### Removed
- なし.

## Archived Changelogs

- [`changelogs/1.3.x.md`](./changelogs/1.3.x.md)
- [`changelogs/1.2.x.md`](./changelogs/1.2.x.md)
- [`changelogs/1.1.x.md`](./changelogs/1.1.x.md)
- [`changelogs/1.0.x.md`](./changelogs/1.0.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
