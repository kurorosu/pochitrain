r"""Optunaハイパーパラメータ最適化設定ファイル.

このファイルでは以下を設定できます：
- 探索空間（search_space）: 最適化するパラメータとその範囲
- 最適化設定: 試行回数、サンプラー、プルーナーなど

使用例:
    python tools/optimize_hyperparams.py \
        --config configs/pochi_train_config.py \
        --optuna-config configs/optuna_config.py \
        --output work_dirs/optuna_results
"""

# === Study設定 ===
study_name = "pochitrain_optimization"
direction = "maximize"  # "maximize"（精度最大化）or "minimize"（損失最小化）
n_trials = 20  # 試行回数
n_jobs = 1  # 並列ジョブ数（1 = 順次実行）

# === サンプラー設定 ===
# 利用可能: "TPESampler", "RandomSampler", "CmaEsSampler", "GridSampler"
sampler = "TPESampler"

# === プルーナー設定（オプション）===
# 利用可能: "MedianPruner", "PercentilePruner", "SuccessiveHalvingPruner",
#           "HyperbandPruner", "NopPruner", None（プルーニングなし）
pruner = "MedianPruner"

# === 最適化時のエポック数 ===
# 本格訓練より短く設定することで探索を高速化
optuna_epochs = 10

# === Storage設定（オプション）===
# 指定すると探索結果をDBに保存し、中断・再開が可能
# 例: "sqlite:///optuna_study.db"
storage = None

# === 探索空間 ===
# 各パラメータの type, low, high, log, choices を指定
search_space = {
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
}

# === 探索空間の例（コメントアウト）===

# スケジューラーパラメータの探索例
# search_space = {
#     "learning_rate": {
#         "type": "float",
#         "low": 1e-5,
#         "high": 1e-1,
#         "log": True,
#     },
#     "scheduler_gamma": {
#         "type": "float",
#         "low": 0.85,
#         "high": 0.99,
#     },
#     "scheduler_step_size": {
#         "type": "int",
#         "low": 5,
#         "high": 30,
#     },
# }

# 層別学習率の探索例（LayerWiseLRSuggestorを使用する場合）
# 注: この場合はLayerWiseLRSuggestorをparam_suggestorとして使用する必要があります
# layer_wise_lr_config = {
#     "layers": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
#     "base_lr_range": (1e-5, 1e-2),
#     "layer_lr_scale_range": (0.1, 10.0),
# }
