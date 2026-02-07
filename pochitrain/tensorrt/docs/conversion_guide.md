# TensorRT 変換ガイド

ONNX モデルから TensorRT エンジンへの変換に関する詳細ドキュメント.

## パイプライン全体像

```
PyTorch (.pth)
    ↓  export-onnx --input-size 224 224
ONNX (.onnx)
    ↓  pochi convert [--fp16 | --int8]
TensorRT Engine (.engine)
    ↓  infer-trt
推論結果 (CSV, サマリー)
```

## ONNX エクスポート時のシェイプ

`export-onnx` は以下の `dynamic_axes` でエクスポートする:

```python
dynamic_axes = {
    "input":  {0: "batch_size"},
    "output": {0: "batch_size"},
}
```

| 次元 | 動的? | 備考 |
|------|-------|------|
| batch (dim 0) | 動的 | `batch_size` シンボル |
| channels (dim 1) | 固定 | 常に 3 |
| height (dim 2) | 固定 | `--input-size` で指定した値 |
| width (dim 3) | 固定 | `--input-size` で指定した値 |

**結果**: ONNX モデルの shape は `(-1, 3, 224, 224)` のようになる (バッチのみ動的).

## pochi convert の変換パターン

### パターン 1: 静的 ONNX (通常ケース)

バッチ次元のみ動的, H/W 固定の ONNX モデル.
**これが `export-onnx` のデフォルト出力.**

```bash
# FP32 変換
pochi convert model.onnx

# FP16 変換
pochi convert model.onnx --fp16

# INT8 変換 (キャリブレーションデータ必要)
pochi convert model.onnx --int8 --calib-data data/val
```

| 項目 | 値 |
|------|-----|
| `--input-size` | 不要 |
| Optimization Profile | batch=1 に固定, H/W は ONNX から取得 |
| 制限事項 | なし |

### パターン 2: 動的シェイプ ONNX (空間次元も動的)

H/W も動的な ONNX モデル. pochitrain の `export-onnx` では生成されないが,
外部ツールでエクスポートした場合にこのパターンになり得る.

```bash
# --input-size が必須
pochi convert dynamic_model.onnx --fp16 --input-size 224 224

# INT8 も同様
pochi convert dynamic_model.onnx --int8 --calib-data data/val --input-size 224 224
```

| 項目 | 値 |
|------|-----|
| `--input-size` | **必須** (未指定時はエラー) |
| Optimization Profile | batch=1, H/W は `--input-size` の値 |
| 制限事項 | 推論時は指定サイズのみ対応 |

### パターン 3: 完全静的 ONNX (バッチも固定)

全次元が固定の ONNX モデル. Optimization Profile の設定不要.

```bash
pochi convert static_model.onnx --fp16
```

| 項目 | 値 |
|------|-----|
| `--input-size` | 不要 |
| Optimization Profile | 不要 (動的次元なし) |
| 制限事項 | バッチサイズ変更不可 |

## 精度モード別の要件

| 精度 | 追加引数 | 備考 |
|------|----------|------|
| FP32 | なし | デフォルト. 最も互換性が高い |
| FP16 | `--fp16` | GPU の FP16 対応が必要 |
| INT8 | `--int8` + キャリブレーション | `--calib-data` または config の `val_data_root` が必要 |

### INT8 キャリブレーションの要件

```bash
pochi convert model.onnx --int8 \
  --calib-data data/val \          # キャリブレーション画像
  --calib-samples 500 \            # サンプル数 (デフォルト: 500)
  --calib-batch-size 1 \           # バッチサイズ (デフォルト: 1)
  -c work_dirs/XXXXXXXX_XXX/config.py  # val_transform の取得元
```

- `val_transform` が config に必須 (キャリブレーション画像の前処理)
- config 未指定時は ONNX パスから自動検出

## エラーと対処法

### "ONNXモデルに動的シェイプが含まれています"

**原因**: 空間次元 (H/W) が動的な ONNX モデルを `--input-size` なしで変換しようとした.

**対処**: `--input-size` で実際に使う解像度を指定する.

```bash
pochi convert model.onnx --fp16 --input-size 224 224
```

### "calibratorが必須です"

**原因**: `--int8` を指定したがキャリブレータが作成できなかった.

**対処**: `--calib-data` でキャリブレーション画像ディレクトリを指定する.

### "configにval_transformが設定されていません"

**原因**: INT8 変換時に config から `val_transform` が見つからない.

**対処**: `--config-path` で `val_transform` を含む config を指定する.

## Optimization Profile の仕組み

TensorRT の Optimization Profile は, 動的次元の min/opt/max 値を定義する.

```
profile.set_shape(name, min_shape, opt_shape, max_shape)
```

pochitrain では min = opt = max に統一している (固定サイズ推論前提):

| 次元 | 解決方法 |
|------|----------|
| batch (-1) | 常に 1 に固定 |
| channels | ONNX から取得 (通常 3) |
| height (-1) | `--input-size` の HEIGHT 値 |
| width (-1) | `--input-size` の WIDTH 値 |

静的次元 (H/W が固定) の場合は ONNX の値をそのまま使用する.
