# Pochitrain Tools

訓練後の分析・可視化ツールを提供します。

## visualize_gradient_trace.py

訓練中に記録された勾配トレースCSVから、層ごとの勾配ノルムを可視化します。

### 使用方法

```bash
python tools/visualize_gradient_trace.py work_dirs/YYYYMMDD_XXX/visualization/gradient_trace_YYYYMMDD_HHMMSS.csv
```

### オプション

```bash
python tools/visualize_gradient_trace.py <csv_path> [--output-dir <dir>]
```

- `csv_path`: 勾配トレースCSVファイルのパス（必須）
- `--output-dir`: 出力ディレクトリ（省略時はCSVと同じディレクトリ）

### 生成される可視化（10種類の画像）

1. **時系列プロット（3枚）**
   - `gradient_trace_timeline_all.png` - すべての層の勾配ノルム推移（対数スケール）
   - `gradient_trace_timeline_early.png` - 前半層の詳細
   - `gradient_trace_timeline_late.png` - 後半層の詳細

2. **ヒートマップ（1枚）**
   - `gradient_trace_heatmap.png` - 層×エポックの2次元表示

3. **統計情報（4枚）**
   - `gradient_trace_statistics_initial_vs_final.png` - 初期 vs 最終エポックの比較
   - `gradient_trace_statistics_stability.png` - 勾配の変動（安定性）
   - `gradient_trace_statistics_max.png` - 勾配の最大値（爆発検出）
   - `gradient_trace_statistics_min.png` - 勾配の最小値（消失検出）

4. **スナップショット（2枚）**
   - `gradient_trace_snapshots.png` - 特定エポックでの層プロファイル（対数スケール）
   - `gradient_trace_snapshots_linear.png` - 特定エポックでの層プロファイル（線形スケール）

### 勾配トレースの有効化

`configs/pochi_train_config.py`で設定：

```python
enable_gradient_tracking = True  # 勾配トレース機能を有効化
gradient_tracking_config = {
    "record_frequency": 1,  # 記録頻度（1 = 毎エポック）
    "exclude_patterns": ["fc\\.", "\\.bias"],  # 除外する層名パターン（正規表現）
    "group_by_block": True,  # ResNetブロック単位で集約
    "aggregation_method": "median",  # 集約方法: "median", "mean", "max", "rms"
}
```

#### 設定の詳細

- **exclude_patterns**: 除外する層名パターン（正規表現）
  - デフォルト: `["fc\\.", "\\.bias"]`（全結合層とbias項を除外）
  - 例: `["fc\\.", "\\.bias", "bn"]` でBatchNorm層も除外

- **group_by_block**: ResNetブロック単位で集約
  - `True`: `layer1.*`, `layer2.*`などをブロック単位で集約（6層に削減）
  - `False`: すべてのパラメータを個別に記録

- **aggregation_method**: 集約方法
  - `"median"`: 中央値（外れ値に強い、推奨）
  - `"mean"`: 平均値（直感的）
  - `"max"`: 最大値（勾配爆発の検出に特化）
  - `"rms"`: RMS（全体のエネルギー）

### 使用例

```bash
# 1. 勾配トレースを有効にして訓練
python pochi.py

# 2. 生成されたCSVを可視化
python tools/visualize_gradient_trace.py work_dirs/20251002_001/visualization/gradient_trace_20251002_192756.csv

# 3. 出力先を指定
python tools/visualize_gradient_trace.py \
    work_dirs/20251002_001/visualization/gradient_trace_20251002_192756.csv \
    --output-dir analysis/
```

### 注意事項

- 勾配トレースは訓練時の計算コストが若干増加します
- デフォルトでは無効化されています（`enable_gradient_tracking = False`）
- 大規模なモデルでは層数が多いため、可視化に時間がかかる場合があります

