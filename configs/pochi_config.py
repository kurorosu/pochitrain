"""pochitrain 設定ファイル.

詳細な使い方は configs/docs/configuration.md を参照してください。
"""

import torchvision.transforms as transforms

# モデル設定
model_name = "resnet18"  # 使用するモデル名
num_classes = 4  # 分類クラス数
pretrained = True  # 事前学習済みモデルを使用

# データ設定
train_data_root = "data/train"  # 訓練データのパス
val_data_root = "data/val"  # 検証データのパス
batch_size = 2  # バッチサイズ
num_workers = 4  # データローダーのワーカー数

# 訓練設定
epochs = 50  # エポック数
learning_rate = 0.001  # 学習率
optimizer = "Adam"  # 最適化器

# スケジューラー設定
scheduler = "StepLR"  # スケジューラー名
scheduler_params = {"step_size": 30, "gamma": 0.1}  # スケジューラーパラメータ

# 損失関数設定
class_weights = None  # クラス毎の損失重み

# その他設定
work_dir = "work_dirs"  # 作業ディレクトリ
device = "cuda"  # デバイス

# データ変換設定
mean = [0.485, 0.456, 0.406]  # 正規化平均値
std = [0.229, 0.224, 0.225]  # 正規化標準偏差

# 訓練用変換
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# 検証用変換
val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)
