"""pochitrain Pochi設定ファイル."""

import torchvision.transforms as transforms

# pochitrain Pochi設定ファイル

# モデル設定
model_name = "resnet18"  # 'resnet18', 'resnet34', 'resnet50'
num_classes = 4  # 分類クラス数
pretrained = True  # 事前学習済みモデルを使用

# データ設定
train_data_root = "data/train"  # 訓練データのパス
val_data_root = "data/val"  # 検証データのパス（Noneの場合は検証なし）
image_size = 224  # 画像サイズ
batch_size = 16  # バッチサイズ
num_workers = 4  # データローダーのワーカー数

# 訓練設定
epochs = 50  # エポック数
learning_rate = 0.001  # 学習率
optimizer = "Adam"  # 最適化器 ('Adam', 'SGD')

# スケジューラー設定（オプション）
scheduler = "StepLR"  # スケジューラー ('StepLR', 'CosineAnnealingLR', None)
scheduler_params = {"step_size": 30, "gamma": 0.1}

# その他設定
work_dir = "work_dirs"  # 作業ディレクトリ
device = None  # デバイス (None で自動選択)

# === データ変換設定 ===
# 基本パラメータ
mean = [0.485, 0.456, 0.406]  # ImageNet標準値
std = [0.229, 0.224, 0.225]  # ImageNet標準値

# 訓練用変換（最小限構成）
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# 検証用変換（リサイズあり）
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# リサイズなしの例（コメントアウト）
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std),
# ])
#
# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std),
# ])
