# pochitrain Pochi設定ファイル

# モデル設定
model_name = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'
num_classes = 10  # 分類クラス数
pretrained = True  # 事前学習済みモデルを使用

# データ設定
train_data_root = 'data/train'  # 訓練データのパス
val_data_root = 'data/val'      # 検証データのパス（Noneの場合は検証なし）
image_size = 224               # 画像サイズ
batch_size = 32               # バッチサイズ
num_workers = 4               # データローダーのワーカー数

# 訓練設定
epochs = 50                   # エポック数
learning_rate = 0.001         # 学習率
optimizer = 'Adam'            # 最適化器 ('Adam', 'SGD')

# スケジューラー設定（オプション）
scheduler = 'StepLR'          # スケジューラー ('StepLR', 'CosineAnnealingLR', None)
scheduler_params = {
    'step_size': 30,
    'gamma': 0.1
}

# その他設定
work_dir = 'work_dirs'        # 作業ディレクトリ
save_every = 10               # チェックポイント保存間隔
device = None                 # デバイス (None で自動選択)
