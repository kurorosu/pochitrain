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
# GPU前処理時の float32 変換を non_blocking で行うかどうか
gpu_non_blocking = True

# 訓練設定
epochs = 50  # エポック数
learning_rate = 0.001  # 学習率
optimizer = "SGD"  # 最適化器

# スケジューラー設定例:
# StepLR: 指定されたステップでγ倍に減衰
# scheduler = "StepLR"
# scheduler_params = {"step_size": 30, "gamma": 0.1}

# MultiStepLR: 複数のマイルストーンで減衰
# scheduler = "MultiStepLR"
# scheduler_params = {"milestones": [30, 60, 90], "gamma": 0.1}

# CosineAnnealingLR: コサイン関数で減衰
# scheduler = "CosineAnnealingLR"
# scheduler_params = {"T_max": 50}

# ExponentialLR: 毎エポック指数減衰（gamma倍）
scheduler = "ExponentialLR"
scheduler_params = {"gamma": 0.95}

# LinearLR: 線形減衰
# scheduler = "LinearLR"
# scheduler_params = {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 50}

# 損失関数設定
class_weights = None  # クラス毎の損失重み

# その他設定
work_dir = "work_dirs"  # 作業ディレクトリ
device = "cuda"  # デバイス
cudnn_benchmark = False  # cuDNN自動チューニング（固定サイズ入力で高速化）

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

# Early Stopping設定
early_stopping = {
    "enabled": False,  # Early Stoppingを有効化
    "patience": 30,  # 改善なしの許容エポック数
    "min_delta": 3.0,  # この値以上の変化がないと改善と見なさない(0.0なら少しでも良くなれば改善扱い)
    "monitor": "val_accuracy",  # 監視メトリクス ("val_accuracy" or "val_loss")
}

# 層別学習率設定
enable_layer_wise_lr = True
layer_wise_lr_config = {
    "layer_rates": {
        "conv1": 0.0001,
        "bn1": 0.0001,
        "layer1": 0.0002,
        "layer2": 0.0005,
        "layer3": 0.001,
        "layer4": 0.002,
        "fc": 0.01,
    },
    "graph_config": {
        "use_log_scale": True,  # Trueで対数スケール、Falseで線形スケール
    },
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

# === Optuna ハイパーパラメータ最適化設定 ===
# 以下の設定は optimize_hyperparams.py 実行時のみ使用されます.
# 通常の訓練（pochi.py train）では無視されます.

# optuna は必ずネスト辞書で指定
optuna = {
    # Study設定
    "study_name": "pochitrain_optimization",
    "direction": "maximize",  # "maximize"（精度最大化）or "minimize"（損失最小化）
    "n_trials": 20,  # 試行回数
    "n_jobs": 1,  # 並列ジョブ数（1 = 順次実行）
    # サンプラー設定
    # 利用可能: "TPESampler", "RandomSampler", "CmaEsSampler", "GridSampler"
    "sampler": "TPESampler",
    # プルーナー設定（オプション）
    # 利用可能: "MedianPruner", "PercentilePruner", "SuccessiveHalvingPruner",
    #           "HyperbandPruner", "NopPruner", None（プルーニングなし）
    "pruner": "MedianPruner",
    # 最適化時のエポック数（本格訓練より短く設定することで探索を高速化）
    "optuna_epochs": 10,
    # Storage設定（オプション）
    # 指定すると探索結果をDBに保存し、中断・再開が可能
    # 例: "sqlite:///optuna_study.db"
    "storage": None,
    # 探索空間
    # 各パラメータの type, low, high, log, choices を指定
    "search_space": {
        # 学習率（対数スケール）
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-1,
            "log": True,
        },
        # バッチサイズ（カテゴリカル）
        "batch_size": {
            "type": "categorical",
            "choices": [16, 32],
        },
        # オプティマイザー（カテゴリカル）
        "optimizer": {
            "type": "categorical",
            "choices": ["SGD", "Adam", "AdamW"],
        },
    },
}

# 探索空間の例（コメントアウト）
# スケジューラーパラメータの探索例:
# optuna["search_space"] = {
#     "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
#     "scheduler_gamma": {"type": "float", "low": 0.85, "high": 0.99},
#     "scheduler_step_size": {"type": "int", "low": 5, "high": 30},
# }
