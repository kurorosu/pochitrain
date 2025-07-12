# pochitrain クイックスタート

最速で検証まで到達するためのシンプルなガイド

## 1. データの準備

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

## 2. 設定ファイルの編集

`configs/simple_config.py` を編集してください：

```python
# モデル設定
model_name = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'
num_classes = 10  # 分類クラス数（自動で設定されます）
pretrained = True  # 事前学習済みモデルを使用

# データ設定
train_data_root = 'data/train'  # 訓練データのパス
val_data_root = 'data/val'      # 検証データのパス
image_size = 224               # 画像サイズ
batch_size = 32               # バッチサイズ

# 訓練設定
epochs = 50                   # エポック数
learning_rate = 0.001         # 学習率
optimizer = 'Adam'            # 最適化器
```

## 3. 訓練実行

```bash
python quick_start.py
```

これだけで訓練が開始されます！

## 4. 結果の確認

訓練結果は `work_dirs/` に保存されます：

- `best_model.pth`: 最高精度のモデル
- `checkpoint_epoch_*.pth`: 各エポックのチェックポイント

## 5. より詳細な使用方法

### 個別に使用する場合

```python
from pochitrain.simple_trainer import SimpleTrainer
from pochitrain.simple_dataset import create_data_loaders

# データローダーの作成
train_loader, val_loader, classes = create_data_loaders(
    train_root='data/train',
    val_root='data/val',
    batch_size=32,
    image_size=224
)

# トレーナーの作成
trainer = SimpleTrainer(
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

## サポートされているモデル

- ResNet18
- ResNet34
- ResNet50

## サポートされている最適化器

- Adam
- SGD

## サポートされているスケジューラー

- StepLR
- CosineAnnealingLR

## 注意点

- 画像は自動的にRGBに変換されます
- ImageNet用の正規化が適用されます
- データ拡張は訓練時のみ適用されます 