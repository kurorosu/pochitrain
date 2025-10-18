# pochitrain 設定ファイルガイド

## 概要

`configs/pochi_train_config.py` は、pochitrainの訓練用設定を管理するPythonファイルです。
`importlib.util`を使用してPythonコードとして読み込まれるため、関数やオブジェクトを直接定義できます。

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
| `val_data_root` | str/None | 検証データのパス（Noneで検証なし） |
| `batch_size` | int | バッチサイズ |
| `num_workers` | int | データローダーのワーカー数 |

### 訓練設定

| パラメータ | 型 | 説明 | 選択肢 |
|------------|----|----- |--------|
| `epochs` | int | エポック数 | 任意の正整数 |
| `learning_rate` | float | 学習率 | 0.0 ~ 1.0 |
| `optimizer` | str | 最適化器 | `'Adam'`, `'SGD'` |

### その他設定

| パラメータ | 型 | 説明 |
|------------|----|----- |
| `work_dir` | str | 作業ディレクトリ |
| `device` | str/None | デバイス（Noneで自動選択） |

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

- `T_max`: 最大エポック数（コサイン周期の半分）

#### ExponentialLR
学習率を指数的に減衰

```python
scheduler = "ExponentialLR"
scheduler_params = {"gamma": 0.95}
```

- `gamma`: 毎エポックの減衰率（0 < gamma < 1）
- 例：`gamma=0.95` ならエポックごとに 0.95 倍

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

- `start_factor`: 減衰開始時の係数（通常 1.0）
- `end_factor`: 減衰終了時の係数（終了時の学習率は `learning_rate * end_factor`）
- `total_iters`: 線形減衰を行う総ステップ数（pochitrain ではエポック数想定）

#### スケジューラーなし

```python
scheduler = None
scheduler_params = None
```

## 層別学習率（Layer-wise Learning Rates）

### 概要
層（パラメータグループ）ごとに異なる学習率を設定し、微調整や段階的な学習制御を可能にする機能です。事前学習済みモデルの前段を保守的に更新し、後段を積極的に学習させる **Discriminative Fine-tuning** などに有効です。

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
- `layer_rates`: 層名（正規表現対応）と学習率を保持する辞書。記述順にマッチングされるため、より具体的な層名を先に書く
- `graph_config.use_log_scale`: 層別学習率グラフを対数スケールで表示するか

### 動作
- 層別学習率が有効でも、CSV の `learning_rate` カラムには設定した値が固定出力される
- 実際の各層の学習率は `lr_<layer_name>` カラムとして CSV に記録され、専用グラフにも描画される
- 正答率グラフには学習率を描画せず、層別学習率専用グラフが生成される

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

## 損失関数設定

### 基本設定（自動重み）

```python
class_weights = None  # 自動的にバランスを調整
```

### クラス重みの手動設定

不均衡データセット用にクラス毎の重みを設定できます：

```python
class_weights = [1.0, 2.0, 1.5, 0.8]  # 4クラスの例
```

- インデックスがクラス番号に対応
- 大きい値ほどそのクラスの損失を重視
- クラス数と配列の長さが一致しない場合はエラー

### 使用例

```python
# 少数クラス（クラス1）を重視する場合
class_weights = [1.0, 3.0, 1.0, 1.0]

# 多数クラス（クラス0）の影響を抑える場合
class_weights = [0.5, 1.0, 1.0, 1.0]
```

## データ変換（Transform）設定

### 重要な注意点

⚠️ **transformは必須です。未設定（None）の場合はエラーになります**

transformを設定しない場合、PIL Imageがそのままの状態で返されるため、PyTorchのテンソルに変換されずに訓練時にエラーが発生します。最低限でも `transforms.ToTensor()` を含む変換を設定してください。

### 基本パラメータ

```python
mean = [0.485, 0.456, 0.406]  # ImageNet標準値
std = [0.229, 0.224, 0.225]   # ImageNet標準値
```

### 訓練用変換の例

#### 最小限構成（必須設定）
```python
train_transform = transforms.Compose([
    transforms.ToTensor(),  # 必須：PIL Image → PyTorchテンソルに変換
    transforms.Normalize(mean=mean, std=std),  # 推奨：正規化
])
```

**重要**: `transforms.ToTensor()` は必須です。これがないとPIL ImageがPyTorchテンソルに変換されず、訓練時にエラーが発生します。

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

#### リサイズなし（最小限構成）
```python
val_transform = transforms.Compose([
    transforms.ToTensor(),  # 必須：PIL Image → PyTorchテンソルに変換
    transforms.Normalize(mean=mean, std=std),  # 推奨：正規化
])
```

**注意**: 検証用変換でも `transforms.ToTensor()` は必須です。

## カスタム正規化

### グレースケール画像用
```python
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
```

### 医用画像用
```python
mean = [0.485, 0.485, 0.485]  # グレースケール寄り
std = [0.229, 0.229, 0.229]
```

## 設定例

### 高速訓練用（軽量設定）
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

### 高精度狙い（重訓練設定）
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
   → 対応スケジューラー（StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR, LinearLR）を使用

3. **Transform設定エラー**
   → `torchvision.transforms`のインポートを確認

4. **層別学習率の設定エラー**
   ```
   ValueError: 層別学習率が有効ですが layer_rates が定義されていません
   ```
   → `layer_wise_lr_config["layer_rates"]` に層名と学習率を指定

5. **層別学習率グラフが見づらい**
   → `graph_config["use_log_scale"]` を調整
     - 学習率差が大きい場合: `True`（対数）
     - 学習率差が小さい場合: `False`（線形）

### パフォーマンス最適化

- **num_workers**: CPUコア数の1/2〜1倍に設定
- **batch_size**: GPUメモリに応じて調整
- **pin_memory**: GPU使用時は`True`に設定（現在は自動）