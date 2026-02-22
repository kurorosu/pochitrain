# pochitrain

[![Version](https://img.shields.io/badge/version-1.6.0-blue.svg)](https://github.com/kurorosu/pochitrain)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-yellow.svg)](https://www.python.org/)
[![Jetson](https://img.shields.io/badge/Jetson-JetPack%206.2.1%20%28Python%203.10%29-76B900.svg)](https://developer.nvidia.com/embedded/jetpack)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)

A tiny but clever CNN pipeline for images — as friendly as Pochi!

**シンプルで親しみやすいCNNパイプラインフレームワーク**

## 📚 ドキュメント

- [GPU環境セットアップガイド](pochitrain/docs/gpu_environment_setup.md) - CUDA/cuDNN/TensorRT のインストールと環境構築
- [設定ファイルガイド](configs/docs/configuration.md) - 詳細な設定方法とカスタマイズ
- [TensorRT変換ガイド](pochitrain/docs/conversion_guide.md) - 動的シェイプ対応と精度モード別の変換手順

## 🚀 クイックスタート

最速で検証まで到達するためのシンプルなガイド

### 1. データの準備

以下のフォルダ構造でデータを準備してください:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

### 2. 設定ファイルの編集

`configs/pochi_train_config.py` を編集してください:

```python
# モデル設定
model_name = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'
num_classes = 10  # 分類クラス数 (自動で設定されます)
pretrained = True  # 事前学習済みモデルを使用

# データ設定
train_data_root = 'data/train'  # 訓練データのパス
val_data_root = 'data/val'      # 検証データのパス
batch_size = 32                 # バッチサイズ

# 訓練設定
epochs = 50                   # エポック数
learning_rate = 0.001         # 学習率
optimizer = 'Adam'            # 最適化器
```

### 3. 訓練実行

`uv run pochi` コマンドを使用します.

訓練の実行 (デフォルト設定ファイルを使用):
```bash
uv run pochi train
```

カスタム設定ファイルを使用する場合:
```bash
uv run pochi train --config configs/my_custom_config.py
```

デバッグログを有効化する場合:
```bash
uv run pochi train --debug
```

これだけで訓練が開始されます!

### 4. 結果の確認

訓練結果は `work_dirs/` に保存されます:

- `best_model.pth`: 最高精度のモデル
- `checkpoint_epoch_*.pth`: 各エポックのチェックポイント

### 5. 推論の実行

基本的な推論 (config・データパスはモデルパスから自動検出):
```bash
uv run pochi infer work_dirs/20251018_001/models/best_epoch40.pth
```

データパスや出力先を上書きする場合:
```bash
uv run pochi infer work_dirs/20251018_001/models/best_epoch40.pth \
  --data data/test \
  --output results/
```

推論完了時に以下の情報が表示されます:
- 入力解像度とチャンネル数
- 精度 (%)
- 純粋推論時間 (ms/image) とスループット (images/sec) — モデルの forward pass のみ
- End-to-End全処理時間 (ms/image) とスループット — I/O・前処理・転送を含む実効性能
- 計測詳細 (ウォームアップ除外サンプル数)

### 6. 結果と出力

訓練結果は `work_dirs/<timestamp>` に保存されます.

- `models/best_epoch*.pth`: ベストモデル
- `training_metrics_*.csv`: 学習率や精度を含むメトリクス
- `training_metrics_*.png`: 損失/精度グラフ (層別学習率が有効な場合は別グラフ)
- `visualization/`: 層別学習率グラフ, 勾配トレースなど

推論結果は `work_dirs/<timestamp>/inference_results/` に保存されます.

- `*_inference_results.csv`: 画像ごとの詳細結果 (ファイルパス, 正解, 予測, 信頼度)
- `*_inference_summary.txt`: 推論サマリー (精度, 推論時間, スループット等を日本語で出力)
- `classification_report.csv`: クラス別精度レポート (precision, recall, f1-score)
- `confusion_matrix.png`: 混同行列

### 7. 勾配トレースの可視化

訓練時に出力された勾配トレースCSVから詳細な可視化グラフを生成できます.

```bash
uv run vis-grad work_dirs/20251018_001/visualization/gradient_trace.csv
```

出力されるグラフ:
- 時系列プロット (全層/前半層/後半層)
- ヒートマップ
- 統計情報 (初期vs最終, 安定性, 最大値, 最小値)
- エポックスナップショット

## 📖 詳細な使用方法

### 個別に使用する場合

```python
from pochitrain import PochiTrainer, create_data_loaders

# データローダーの作成
train_loader, val_loader, classes = create_data_loaders(
    train_root='data/train',
    val_root='data/val',
    batch_size=32
)

# トレーナーの作成
trainer = PochiTrainer(
    model_name='resnet18',
    num_classes=len(classes),
    pretrained=True
)

# 訓練設定
trainer.setup_training(
    learning_rate=0.001,
    optimizer_name='Adam'
)

# 訓練実行
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)
```

### 予測の実行

```python
# モデルの読み込み
trainer.load_checkpoint('best_checkpoint.pth')

# 予測の実行
predictions, confidences = trainer.predict(test_loader)
```

## 🎯 特徴

- **シンプルなAPI**: 3ステップで訓練開始
- **torchvisionモデル**: ResNet18/34/50を直接使用
- **事前学習済み**: ImageNet事前学習済みモデル
- **基本的なtransform**: データ拡張と正規化を内蔵
- **最速検証**: 複雑な設定不要

## 🛠️ サポート機能

### モデル
- ResNet18
- ResNet34
- ResNet50

### 最適化器
- Adam
- AdamW
- SGD

### スケジューラー
- StepLR
- MultiStepLR
- CosineAnnealingLR
- ExponentialLR
- LinearLR

### 高度な機能
- **層別学習率 (Layer-wise Learning Rates)**: 各層の学習率を個別設定し, 専用グラフを出力
- **メトリクス記録**: 学習率や損失を CSV/グラフに自動保存
- **勾配トレース**: 層ごとの勾配推移を可視化
- **クラス重み**: 不均衡データセットへ柔軟に対応
- **ハイパーパラメータ最適化**: Optunaによる自動パラメータ探索
- **Early Stopping**: 過学習を自動検知して訓練を早期終了
- **クラス別精度レポート**: 推論時にクラスごとの精度を詳細出力
- **TensorRT推論**: ONNXモデルをTensorRTエンジンに変換し高速推論 (FP32/FP16/INT8量子化対応)

## 📋 要件

- Python 3.10+
- PyTorch 2.9+ (CUDA 13.0)
- torchvision 0.21+
- pandas 2.0+ (勾配トレース可視化用)
- Optuna 3.5+ (ハイパーパラメータ最適化用)

## 📦 インストール

### uv を使用する場合 (推奨)

uv のインストール (未インストールの場合):
```bash
pip install uv
```

依存関係のインストール:
```bash
uv sync
```

仮想環境の有効化:
```bash
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac
```

開発用依存関係も含める場合:
```bash
uv sync --group dev
```

### Jetson環境で依存混在を避ける手順

Jetson では `tensorrt` を system package から利用するため,
`--system-site-packages` 付き venv を使います.
このとき `numpy` / `scipy` が混在しやすいので, venv 側を優先するために
再インストールを明示します.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip install --force-reinstall --no-cache-dir "numpy==1.26.1" "scipy>=1.11.0"
python -m pip check
python -c "import numpy, scipy; print(numpy.__file__, numpy.__version__); print(scipy.__file__, scipy.__version__)"
```

`numpy.__file__` と `scipy.__file__` が `.venv` 配下を指していれば,
混在は解消されています.
`matplotlib` は Jetson では system package (`/usr/lib/...`) を使って問題ありません
(必要なら `python -c "import matplotlib; print(matplotlib.__file__, matplotlib.__version__)"` で確認).

## 🔬 ハイパーパラメータ最適化

Optunaを使ったハイパーパラメータ自動探索機能です.

### 基本的な使い方

最適化の実行 (デフォルト設定ファイルを使用):
```bash
uv run pochi optimize
```

カスタム設定ファイルを使用する場合:
```bash
uv run pochi optimize --config configs/my_custom_config.py
```

出力先を変更する場合:
```bash
uv run pochi optimize --output work_dirs/custom_results
```

出力ディレクトリ (`work_dirs/optuna_results`) が既に存在する場合, 自動的に連番が付与されます (`optuna_results_001`, `optuna_results_002`...).

### 出力ファイル

| ファイル | 説明 |
|----------|------|
| `best_params.json` | 最適パラメータ |
| `trials_history.json` | 全試行履歴 |
| `optimized_config.py` | 最適パラメータを反映した設定ファイル |
| `study_statistics.json` | 統計情報 + パラメータ重要度 |
| `optimization_history.html` | 最適化履歴グラフ (Plotly) |
| `param_importances.html` | パラメータ重要度グラフ (Plotly) |
| `contour.html` | パラメータ間の等高線プロット (Plotly) |

### 最適化後の訓練

最適化されたパラメータで本格訓練:
```bash
uv run pochi train --config work_dirs/optuna_results/optimized_config.py
```

### 探索空間のカスタマイズ

`configs/pochi_train_config.py` の `search_space` で探索範囲を設定できます:

```python
search_space = {
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-1,
        "log": True,  # 対数スケール
    },
    "batch_size": {
        "type": "categorical",
        "choices": [16, 32, 64],
    },
    "optimizer": {
        "type": "categorical",
        "choices": ["SGD", "Adam", "AdamW"],
    },
}
```

詳細は [設定ファイルガイド](configs/docs/configuration.md#optunaハイパーパラメータ最適化設定) を参照してください.

## 🔄 ONNXエクスポート・推論

学習済みモデルをONNX形式にエクスポートし, ONNX Runtimeで高速推論を行う機能です.

### ONNX依存関係のインストール

```bash
uv sync --group onnx
```

### モデルのエクスポート

PyTorchチェックポイント (.pth) をONNX形式に変換:
```bash
uv run export-onnx work_dirs/20251018_001/models/best_epoch40.pth
```

入力サイズを指定する場合:
```bash
uv run export-onnx work_dirs/20251018_001/models/best_epoch40.pth --input-size 224 224
```

出力先とopsetバージョンを指定:
```bash
uv run export-onnx work_dirs/20251018_001/models/best_epoch40.pth \
  --output model.onnx \
  --opset 17
```

### ONNX推論の実行

エクスポートしたONNXモデルで推論 (config・データパスはモデルパスから自動検出):
```bash
uv run infer-onnx work_dirs/20251018_001/models/best_epoch40.onnx
```

データパスや出力先を上書きする場合:
```bash
uv run infer-onnx work_dirs/20251018_001/models/best_epoch40.onnx \
  --data data/test -o results/
```

### コマンドオプション

**export-onnx:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--output` | 出力ONNXファイルパス | `<入力ファイル名>.onnx` |
| `--input-size` | 入力画像サイズ (H W) | `224 224` |
| `--opset` | ONNX opsetバージョン | `17` |
| `--no-verify` | エクスポート後の検証をスキップ | - |

**infer-onnx:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--debug` | デバッグログを有効化 | - |
| `--data` | 推論データのパス | configの`val_data_root` |
| `--output` | 結果の出力先ディレクトリ | モデルパスから自動決定 |

## ⚡ TensorRT高速推論

ONNXモデルをTensorRTエンジンに変換し, ネイティブTensorRTで高速推論を行う機能です. 環境によってはONNX Runtimeと比較して数倍高速な推論が可能です.

### 前提条件

TensorRT SDKのインストールが必要です.

1. [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)からSDKをダウンロード
2. SDKをインストール後, `trtexec`がPATHに通っていることを確認
3. Python APIをインストール:
```bash
uv pip install <TensorRT_SDK_PATH>/python/tensorrt-10.x.x-cpXX-none-win_amd64.whl
```

### 使用フロー

#### 1. ONNXモデルのエクスポート

```bash
uv run export-onnx work_dirs/20251018_001/models/best_epoch40.pth --input-size 512 512
```

#### 2. TensorRTエンジンのビルド

FP32で変換:
```bash
uv run pochi convert best_epoch40.onnx
```

FP16で変換:
```bash
uv run pochi convert best_epoch40.onnx --fp16
```

INT8量子化で変換 (キャリブレーションデータを自動取得):
```bash
uv run pochi convert best_epoch40.onnx --int8
```

キャリブレーションデータを明示的に指定:
```bash
uv run pochi convert best_epoch40.onnx --int8 --calib-data data/val --calib-samples 300
```

動的シェイプONNXモデルを変換 (入力サイズの明示指定が必要):
```bash
uv run pochi convert best_epoch40.onnx --fp16 --input-size 512 512
```

#### 3. TensorRT推論の実行

基本的な使い方 (config・データパスはエンジンパスから自動検出):
```bash
uv run infer-trt work_dirs/20251018_001/models/model.engine
```

データパスや出力先を上書きする場合:
```bash
uv run infer-trt work_dirs/20251018_001/models/model.engine \
  --data data/test -o results/
```

### コマンドオプション

**pochi convert:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--fp16` | FP16精度で変換 | - |
| `--int8` | INT8精度で変換 (キャリブレーション必要) | - |
| `--output` | 出力エンジンファイルパス | `<入力ファイル名>.engine` |
| `--config-path` | 設定ファイルパス (INT8時にtransformとデータパスを取得) | ONNXパスから自動検出 |
| `--calib-data` | キャリブレーションデータディレクトリ | configの`val_data_root` |
| `--input-size` | 入力画像サイズ H W (動的シェイプONNXモデルの変換時に必要) | - |
| `--calib-samples` | キャリブレーションサンプル数 (1以上) | `500` |
| `--calib-batch-size` | キャリブレーションバッチサイズ (1以上) | `1` |
| `--workspace-size` | TensorRTワークスペースサイズ bytes (1以上) | `1GB` |

**infer-trt:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--debug` | デバッグログを有効化 | - |
| `--data` | 推論データのパス | configの`val_data_root` |
| `--output` | 結果の出力先ディレクトリ | エンジンパスから自動決定 |

### 注意事項

- TensorRTエンジンはGPUアーキテクチャ固有です (異なるGPUでは再ビルドが必要)
- `uv sync`を実行するとTensorRTがアンインストールされます. その場合は再度`uv pip install`でインストールしてください

## 🔧 設定オプション

設定ファイル (`configs/pochi_train_config.py`) で以下の項目を調整できます:

| 項目 | 説明 | デフォルト |
|------|------|-----------|
| `model_name` | モデル名 | 'resnet18' |
| `pretrained` | 事前学習済みモデル使用 | True |
| `batch_size` | バッチサイズ | 32 |
| `epochs` | エポック数 | 50 |
| `learning_rate` | 学習率 | 0.001 |
| `optimizer` | 最適化器 | 'Adam' |

## 📝 注意点

- グレースケールやRGBA画像は自動的にRGBに変換されます
- クラス数は自動で検出されます
- 推論では最初のバッチがウォームアップとして計測から除外されます
- `--debug` フラグを付けると, 推論時のTransform内容やバッチ単位の処理時間など詳細ログが表示されます

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています.
