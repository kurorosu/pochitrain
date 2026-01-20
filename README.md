# pochitrain

A tiny but clever CNN pipeline for images — as friendly as Pochi!

**シンプルで親しみやすいCNNパイプラインフレームワーク**

## 📚 ドキュメント

- [設定ファイルガイド](configs/docs/configuration.md) - 詳細な設定方法とカスタマイズ

## 🚀 クイックスタート

最速で検証まで到達するためのシンプルなガイド

### 1. データの準備

以下のフォルダ構造でデータを準備してください：

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

`configs/pochi_train_config.py` を編集してください：

```python
# モデル設定
model_name = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'
num_classes = 10  # 分類クラス数（自動で設定されます）
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

これだけで訓練が開始されます!

### 4. 結果の確認

訓練結果は `work_dirs/` に保存されます：

- `best_model.pth`: 最高精度のモデル
- `checkpoint_epoch_*.pth`: 各エポックのチェックポイント

### 5. 推論の実行

基本的な推論:
```bash
uv run pochi infer \
  --model-path work_dirs/20251018_001/models/best_epoch40.pth \
  --data data/val \
  --config-path work_dirs/20251018_001/config.py
```

出力先を指定する場合:
```bash
uv run pochi infer \
  --model-path work_dirs/20251018_001/models/best_epoch40.pth \
  --data data/test \
  --config-path work_dirs/20251018_001/config.py \
  --output results/
```

推論完了時に1枚あたりの平均推論時間 (ms/image) が表示されます. 実運用での1枚ずつの推論速度を計測したい場合は, configの`batch_size`を1に設定してください.

### 6. 結果と出力

訓練結果は `work_dirs/<timestamp>` に保存されます。

- `models/best_epoch*.pth`: ベストモデル
- `training_metrics_*.csv`: 学習率や精度を含むメトリクス
- `training_metrics_*.png`: 損失/精度グラフ（層別学習率が有効な場合は別グラフ）
- `visualization/`: 層別学習率グラフ、勾配トレースなど

### 7. 勾配トレースの可視化

訓練時に出力された勾配トレースCSVから詳細な可視化グラフを生成できます.

```bash
python tools/visualize_gradient_trace.py work_dirs/20251018_001/visualization/gradient_trace.csv
```

出力されるグラフ:
- 時系列プロット（全層/前半層/後半層）
- ヒートマップ
- 統計情報（初期vs最終、安定性、最大値、最小値）
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

## 📋 要件

- Python 3.13+
- PyTorch 2.6+ (CUDA 13.0)
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

## 🔧 設定オプション

設定ファイル（`configs/pochi_train_config.py`）で以下の項目を調整できます：

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

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。
