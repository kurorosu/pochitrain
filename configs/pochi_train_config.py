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
batch_size = 16  # バッチサイズ
num_workers = 0  # データローダーのワーカー数（Ctrl+Cエラー回避のため0に設定）

# 訓練設定
epochs = 50  # エポック数
learning_rate = 0.001  # 学習率
optimizer = "SGD"  # 最適化器

# スケジューラー設定
scheduler = "StepLR"  # スケジューラー名
scheduler_params = {"step_size": 30, "gamma": 0.1}  # スケジューラーパラメータ

# 損失関数設定
class_weights = None  # クラス毎の損失重み

# その他設定
work_dir = "work_dirs"  # 作業ディレクトリ
device = "cuda"  # デバイス

# 訓練メトリクス可視化設定
enable_metrics_export = True  # メトリクスのCSV出力とグラフ生成を有効化

# 勾配トレース設定
enable_gradient_tracking = True  # デフォルトOFF（計算コスト考慮）
gradient_tracking_config = {
    "record_frequency": 1,  # 記録頻度（1 = 毎エポック）
    "exclude_patterns": ["fc\\.", "\\.bias"],  # 除外する層名パターン（正規表現）
    "group_by_block": True,  # ResNetブロック単位で集約（layer1.*, layer2.*など）
    "aggregation_method": "median",  # 集約方法: "median", "mean", "max", "rms"
}

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

# 混同行列可視化設定
confusion_matrix_config = {
    "title": "Confusion Matrix",  # タイトル
    "xlabel": "Predicted Label",  # x軸ラベル
    "ylabel": "True Label",  # y軸ラベル
    "fontsize": 14,  # セル内数値のフォントサイズ
    "title_fontsize": 16,  # タイトルのフォントサイズ
    "label_fontsize": 12,  # 軸ラベルのフォントサイズ
    "figsize": (8, 6),  # 図のサイズ (幅, 高さ)
    "cmap": "Blues",  # カラーマップ
}

# 日本語表示の例:
# confusion_matrix_config = {
#     "title": "混同行列",
#     "xlabel": "予測ラベル",
#     "ylabel": "実際ラベル",
#     "fontsize": 14,
#     "title_fontsize": 16,
#     "label_fontsize": 12,
#     "figsize": (8, 6),
#     "cmap": "Blues",
# }
