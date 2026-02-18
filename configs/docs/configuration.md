# pochitrain 設定ファイルガイド

## 概要

`configs/pochi_train_config.py` は, pochitrainの訓練用設定を管理するPythonファイルです.
`importlib.util`を使用してPythonコードとして読み込まれるため, 関数やオブジェクトを直接定義できます.

## 基本設定

### モデル設定

| パラメータ | 型 | 説明 | 選択肢 |
|------------|----|----- |--------|
| `model_name` | str | 使用するモデル名 | `'resnet18'`, `'resnet34'`, `'resnet50'` |
| `num_classes` | int | 分類クラス数 | 任意の正整数 |
| `pretrained` | bool | 事前学習済みモデルを使用するか | `True`, `False` |

### データ設定

| パラメータ | 型 | 説明 |
|------------|----|----- |
| `train_data_root` | str | 訓練データのパス |
| `val_data_root` | str/None | 検証データのパス (Noneで検証なし) |
| `batch_size` | int | バッチサイズ |
| `num_workers` | int | データローダーのワーカー数 |

### 訓練設定

| パラメータ | 型 | 説明 | 選択肢 |
|------------|----|----- |--------|
| `epochs` | int | エポック数 | 任意の正整数 |
| `learning_rate` | float | 学習率 | 0.0 ~ 1.0 |
| `optimizer` | str | 最適化器 | `'Adam'`, `'AdamW'`, `'SGD'` |

### メトリクス・可視化設定

| パラメータ | 型 | 説明 | デフォルト |
|------------|----|----- |------------|
| `enable_metrics_export` | bool | メトリクスのCSV出力とグラフ生成を有効化 | `True` |

### その他設定

| パラメータ | 型 | 説明 | デフォルト |
|------------|----|----- |------------|
| `work_dir` | str | 作業ディレクトリ | `"work_dirs"` |
| `device` | str/None | デバイス (Noneで自動選択) | `None` |
| `cudnn_benchmark` | bool | cuDNN自動チューニング (固定サイズ入力で高速化) | `False` |

## スケジューラー設定

### 基本設定

```python
scheduler = "StepLR"  # スケジューラー名
scheduler_params = {"step_size": 30, "gamma": 0.1}  # パラメータ
```

### 対応スケジューラー

#### StepLR
一定エポック間隔で学習率を減衰

```python
scheduler = "StepLR"
scheduler_params = {"step_size": 30, "gamma": 0.1}
```

- `step_size`: 学習率を減衰するエポック間隔
- `gamma`: 減衰率

#### MultiStepLR
指定したエポックで学習率を減衰

```python
scheduler = "MultiStepLR"
scheduler_params = {"milestones": [30, 60, 90], "gamma": 0.1}
```

- `milestones`: 学習率を減衰するエポックのリスト
- `gamma`: 減衰率

#### CosineAnnealingLR
コサイン関数に従って学習率を変化

```python
scheduler = "CosineAnnealingLR"
scheduler_params = {"T_max": 50}
```

- `T_max`: 最大エポック数 (コサイン周期の半分)

#### ExponentialLR
学習率を指数的に減衰

```python
scheduler = "ExponentialLR"
scheduler_params = {"gamma": 0.95}
```

- `gamma`: 毎エポックの減衰率 (0 < gamma < 1)
- 例: `gamma=0.95` ならエポックごとに 0.95 倍

#### LinearLR
指定したイテレーション数で線形に減衰

```python
scheduler = "LinearLR"
scheduler_params = {
    "start_factor": 1.0,
    "end_factor": 0.1,
    "total_iters": 100,
}
```

- `start_factor`: 減衰開始時の係数 (通常 1.0)
- `end_factor`: 減衰終了時の係数 (終了時の学習率は `learning_rate * end_factor`)
- `total_iters`: 線形減衰を行う総ステップ数 (pochitrain ではエポック数想定)

#### スケジューラーなし

```python
scheduler = None
scheduler_params = None
```

## 層別学習率 (Layer-wise Learning Rates)

### 概要
層 (パラメータグループ) ごとに異なる学習率を設定し, 微調整や段階的な学習制御を可能にする機能です. 事前学習済みモデルの前段を保守的に更新し, 後段を積極的に学習させる **Discriminative Fine-tuning** などに有効です.

### 基本設定
```python
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
        "use_log_scale": False,  # True で対数スケール
    },
}
```

### パラメータ
- `enable_layer_wise_lr`: 層別学習率機能の有効/無効
- `layer_rates`: 層名 (正規表現対応) と学習率を保持する辞書. 記述順にマッチングされるため, より具体的な層名を先に書く
- `graph_config.use_log_scale`: 層別学習率グラフを対数スケールで表示するか

### 動作
- 層別学習率が有効でも, CSV の `learning_rate` カラムには設定した値が固定出力される
- 実際の各層の学習率は `lr_<layer_name>` カラムとして CSV に記録され, 専用グラフにも描画される
- 正答率グラフには学習率を描画せず, 層別学習率専用グラフが生成される

### スケジューラーとの併用例
```python
learning_rate = 0.001
enable_layer_wise_lr = True
layer_wise_lr_config = {
    "layer_rates": {
        "conv1": 0.0001,
        "layer1": 0.0005,
        "layer2": 0.001,
        "layer3": 0.002,
        "layer4": 0.005,
        "fc": 0.01,
    },
    "graph_config": {"use_log_scale": True},
}

scheduler = "ExponentialLR"
scheduler_params = {"gamma": 0.95}
```

```python
learning_rate = 0.001
enable_layer_wise_lr = True
layer_wise_lr_config = {
    "layer_rates": {
        "conv1": 0.0001,
        "layer1": 0.0002,
        "layer2": 0.0005,
        "layer3": 0.001,
        "layer4": 0.002,
        "fc": 0.01,
    },
    "graph_config": {"use_log_scale": False},
}

scheduler = "LinearLR"
scheduler_params = {
    "start_factor": 1.0,
    "end_factor": 0.1,
    "total_iters": 100,
}
```

## Early Stopping設定

過学習を自動検知して訓練を早期終了する機能です.

### 基本設定

```python
early_stopping = {
    "enabled": False,       # Early Stoppingを有効化
    "patience": 30,         # 改善なしの許容エポック数
    "min_delta": 3.0,       # この値以上の変化がないと改善と見なさない
    "monitor": "val_accuracy",  # 監視メトリクス
}
```

### パラメータ

| パラメータ | 型 | 説明 | デフォルト |
|------------|----|----- |------------|
| `enabled` | bool | Early Stoppingの有効/無効 | `False` |
| `patience` | int | 改善なしの許容エポック数 | `30` |
| `min_delta` | float | 改善と見なす最小変化量 (0.0なら少しでも良くなれば改善扱い) | `3.0` |
| `monitor` | str | 監視メトリクス | `"val_accuracy"` |

`monitor` の選択肢:
- `"val_accuracy"`: 検証精度を監視 (値が増加すれば改善)
- `"val_loss"`: 検証損失を監視 (値が減少すれば改善)

## 勾配トレース設定

訓練中の各層の勾配推移を記録・可視化する機能です.

### 基本設定

```python
enable_gradient_tracking = True  # デフォルトOFF (計算コスト考慮)
gradient_tracking_config = {
    "record_frequency": 1,                     # 記録頻度 (1 = 毎エポック)
    "exclude_patterns": ["fc\\.", "\\.bias"],   # 除外する層名パターン (正規表現)
    "group_by_block": True,                     # ResNetブロック単位で集約
    "aggregation_method": "median",             # 集約方法
}
```

### パラメータ

| パラメータ | 型 | 説明 | デフォルト |
|------------|----|----- |------------|
| `enable_gradient_tracking` | bool | 勾配トレースの有効/無効 | `False` |
| `record_frequency` | int | 記録頻度 (エポック単位) | `1` |
| `exclude_patterns` | list[str] | 除外する層名の正規表現パターン | `["fc\\.", "\\.bias"]` |
| `group_by_block` | bool | ResNetブロック単位で集約 (layer1.*, layer2.* など) | `True` |
| `aggregation_method` | str | 集約方法 | `"median"` |

`aggregation_method` の選択肢:
- `"median"`: 中央値
- `"mean"`: 平均値
- `"max"`: 最大値
- `"rms"`: 二乗平均平方根

## 損失関数設定

### 基本設定 (自動重み)

```python
class_weights = None  # 自動的にバランスを調整
```

### クラス重みの手動設定

不均衡データセット用にクラス毎の重みを設定できます:

```python
class_weights = [1.0, 2.0, 1.5, 0.8]  # 4クラスの例
```

- インデックスがクラス番号に対応
- 大きい値ほどそのクラスの損失を重視
- クラス数と配列の長さが一致しない場合はエラー

### 使用例

```python
# 少数クラス (クラス1) を重視する場合
class_weights = [1.0, 3.0, 1.0, 1.0]

# 多数クラス (クラス0) の影響を抑える場合
class_weights = [0.5, 1.0, 1.0, 1.0]
```

## データ変換 (Transform) 設定

### 重要な注意点

⚠️ **transformは必須です. 未設定 (None) の場合はエラーになります**

transformを設定しない場合, PIL Imageがそのままの状態で返されるため, PyTorchのテンソルに変換されずに訓練時にエラーが発生します. 最低限でも `transforms.ToTensor()` を含む変換を設定してください.

### 基本パラメータ

```python
mean = [0.485, 0.456, 0.406]  # ImageNet標準値
std = [0.229, 0.224, 0.225]   # ImageNet標準値
```

### 訓練用変換の例

#### 最小限構成 (必須設定)
```python
train_transform = transforms.Compose([
    transforms.ToTensor(),  # 必須: PIL Image → PyTorchテンソルに変換
    transforms.Normalize(mean=mean, std=std),  # 推奨: 正規化
])
```

**重要**: `transforms.ToTensor()` は必須です. これがないとPIL ImageがPyTorchテンソルに変換されず, 訓練時にエラーが発生します.

#### データ拡張あり
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
```

### 検証用変換の例

#### リサイズあり
```python
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
```

#### リサイズなし (最小限構成)
```python
val_transform = transforms.Compose([
    transforms.ToTensor(),  # 必須: PIL Image → PyTorchテンソルに変換
    transforms.Normalize(mean=mean, std=std),  # 推奨: 正規化
])
```

**注意**: 検証用変換でも `transforms.ToTensor()` は必須です.

## カスタム正規化

### グレースケール寄りのRGB画像用
pochitrain はグレースケールやRGBA画像を自動的にRGBに変換するため, チャンネル数は常に3です.
元がグレースケールの画像 (RGBに変換済み) には以下の正規化が適しています.
```python
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
```

### 医用画像用
```python
mean = [0.485, 0.485, 0.485]  # グレースケール寄りのRGB
std = [0.229, 0.229, 0.229]
```

## 混同行列設定

推論時に出力される混同行列の表示をカスタマイズできます.

### 基本設定

```python
confusion_matrix_config = {
    "title": "Confusion Matrix",    # タイトル
    "xlabel": "Predicted Label",    # x軸ラベル
    "ylabel": "True Label",         # y軸ラベル
    "fontsize": 14,                 # セル内数値のフォントサイズ
    "title_fontsize": 16,           # タイトルのフォントサイズ
    "label_fontsize": 12,           # 軸ラベルのフォントサイズ
    "figsize": (8, 6),              # 図のサイズ (幅, 高さ)
    "cmap": "Blues",                # カラーマップ
}
```

### 日本語表示の例

```python
confusion_matrix_config = {
    "title": "混同行列",
    "xlabel": "予測ラベル",
    "ylabel": "実際ラベル",
    "fontsize": 14,
    "title_fontsize": 16,
    "label_fontsize": 12,
    "figsize": (8, 6),
    "cmap": "Blues",
}
```

### パラメータ

| パラメータ | 型 | 説明 | デフォルト |
|------------|----|----- |------------|
| `title` | str | グラフタイトル | `"Confusion Matrix"` |
| `xlabel` | str | x軸ラベル | `"Predicted Label"` |
| `ylabel` | str | y軸ラベル | `"True Label"` |
| `fontsize` | int | セル内数値のフォントサイズ | `14` |
| `title_fontsize` | int | タイトルのフォントサイズ | `16` |
| `label_fontsize` | int | 軸ラベルのフォントサイズ | `12` |
| `figsize` | tuple | 図のサイズ (幅, 高さ) | `(8, 6)` |
| `cmap` | str | matplotlib カラーマップ名 | `"Blues"` |

## 設定例

### 高速訓練用 (軽量設定)
```python
model_name = "resnet18"
batch_size = 32
epochs = 20
learning_rate = 0.01
optimizer = "SGD"
scheduler = "StepLR"
scheduler_params = {"step_size": 10, "gamma": 0.1}

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
```

### 高精度狙い (重訓練設定)
```python
model_name = "resnet50"
batch_size = 16
epochs = 100
learning_rate = 0.001
optimizer = "Adam"
scheduler = "MultiStepLR"
scheduler_params = {"milestones": [30, 60, 90], "gamma": 0.1}

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
```

### 不均衡データセット用
```python
# クラス重みで少数クラスを重視
class_weights = [0.5, 2.0, 1.5, 1.0]  # クラス1を最重視

# CosineAnnealingLRで安定した学習
scheduler = "CosineAnnealingLR"
scheduler_params = {"T_max": 50}
```

### 層別学習率 + ExponentialLR
```python
learning_rate = 0.001
enable_layer_wise_lr = True
layer_wise_lr_config = {
    "layer_rates": {
        "conv1": 0.0001,
        "layer1": 0.0003,
        "layer2": 0.0005,
        "layer3": 0.001,
        "layer4": 0.003,
        "fc": 0.01,
    },
    "graph_config": {"use_log_scale": True},
}

scheduler = "ExponentialLR"
scheduler_params = {"gamma": 0.95}
```

### 層別学習率 + LinearLR
```python
learning_rate = 0.001
enable_layer_wise_lr = True
layer_wise_lr_config = {
    "layer_rates": {
        "conv1": 0.0002,
        "layer1": 0.0004,
        "layer2": 0.0006,
        "layer3": 0.0008,
        "layer4": 0.001,
        "fc": 0.002,
    },
    "graph_config": {"use_log_scale": False},
}

scheduler = "LinearLR"
scheduler_params = {
    "start_factor": 1.0,
    "end_factor": 0.1,
    "total_iters": 50,
}
```

## トラブルシューティング

### よくあるエラー

1. **クラス重みエラー**
   ```
   ValueError: クラス重みの長さ(3)がクラス数(4)と一致しません
   ```
   → `class_weights`の要素数を`num_classes`と一致させる

2. **サポートされていないスケジューラー**
   ```
   ValueError: サポートされていないスケジューラー: CustomLR
   ```
   → 対応スケジューラー (StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR, LinearLR) を使用

3. **Transform設定エラー**
   → `torchvision.transforms`のインポートを確認

4. **層別学習率の設定エラー**
   ```
   ValueError: 層別学習率が有効ですが layer_rates が定義されていません
   ```
   → `layer_wise_lr_config["layer_rates"]` に層名と学習率を指定

5. **層別学習率グラフが見づらい**
   → `graph_config["use_log_scale"]` を調整
     - 学習率差が大きい場合: `True` (対数)
     - 学習率差が小さい場合: `False` (線形)

### パフォーマンス最適化

- **num_workers**: CPUコア数の1/2〜1倍に設定
- **batch_size**: GPUメモリに応じて調整
- **pin_memory**: GPU使用時は`True`に設定 (現在は自動)

## Optunaハイパーパラメータ最適化設定

v1.1.0より, Optuna設定を`pochi_train_config.py`に統合しました. `pochi optimize`コマンドで使用されます.
`study_name`や`search_space`などのフラットキー形式は廃止され, `optuna = {...}` で指定します.

### 基本設定

| パラメータ | 型 | 説明 | デフォルト |
|------------|----|----- |------------|
| `optuna["study_name"]` | str | Study名 | `"pochitrain_optimization"` |
| `optuna["direction"]` | str | 最適化方向 | `"maximize"` |
| `optuna["n_trials"]` | int | 試行回数 | `20` |
| `optuna["n_jobs"]` | int | 並列ジョブ数 | `1` |
| `optuna["optuna_epochs"]` | int | 最適化時のエポック数 | `10` |

```python
optuna = {
    "study_name": "pochitrain_optimization",
    "direction": "maximize",  # "maximize"(精度最大化) or "minimize"(損失最小化)
    "n_trials": 20,
    "n_jobs": 1,
    "optuna_epochs": 10,  # 本格訓練より短く設定して探索を高速化
}
```

### サンプラー設定

| サンプラー | 説明 |
|------------|------|
| `TPESampler` | Tree-structured Parzen Estimator (デフォルト, 推奨) |
| `RandomSampler` | ランダム探索 |
| `CmaEsSampler` | CMA-ES (連続値パラメータ向け) |
| `GridSampler` | グリッドサーチ |

```python
optuna["sampler"] = "TPESampler"
```

### プルーナー設定

| プルーナー | 説明 |
|------------|------|
| `MedianPruner` | 中央値ベースの枝刈り (デフォルト) |
| `PercentilePruner` | パーセンタイルベースの枝刈り |
| `SuccessiveHalvingPruner` | Successive Halving |
| `HyperbandPruner` | Hyperband |
| `NopPruner` | 枝刈りなし |
| `None` | プルーニングなし |

```python
optuna["pruner"] = "MedianPruner"
```

### Storage設定 (オプション)

探索結果をDBに保存し, 中断・再開を可能にします.

```python
optuna["storage"] = None  # メモリ内のみ
# optuna["storage"] = "sqlite:///optuna_study.db"  # SQLiteに保存
```

### 探索空間

各パラメータの探索範囲を指定します.

```python
optuna["search_space"] = {
    # 学習率 (対数スケール)
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-1,
        "log": True,
    },
    # バッチサイズ (カテゴリカル)
    "batch_size": {
        "type": "categorical",
        "choices": [16, 32],
    },
    # オプティマイザー (カテゴリカル)
    "optimizer": {
        "type": "categorical",
        "choices": ["SGD", "Adam", "AdamW"],
    },
}
```

#### 探索空間のパラメータ型

| 型 | 必須フィールド | オプション | 説明 |
|----|----------------|------------|------|
| `float` | `low`, `high` | `log` | 浮動小数点数 |
| `int` | `low`, `high` | `log` | 整数 |
| `categorical` | `choices` | - | カテゴリカル値 |

#### スケジューラーパラメータの探索例

```python
optuna["search_space"] = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    "scheduler_gamma": {"type": "float", "low": 0.85, "high": 0.99},
    "scheduler_step_size": {"type": "int", "low": 5, "high": 30},
}
```

### 最適化の実行

```bash
uv run pochi optimize --config configs/pochi_train_config.py
```

### 出力ファイル

最適化完了後, `work_dirs/optuna_results/`に以下のファイルが生成されます:

| ファイル | 説明 |
|----------|------|
| `best_params.json` | 最適パラメータ |
| `trials_history.json` | 全試行履歴 |
| `optimized_config.py` | 最適パラメータを反映した設定ファイル |
| `study_statistics.json` | 統計情報 + パラメータ重要度 |
| `optimization_history.html` | 最適化履歴グラフ (Plotly) |
| `param_importances.html` | パラメータ重要度グラフ (Plotly) |
| `contour.html` | パラメータ間の等高線プロット (Plotly) |
