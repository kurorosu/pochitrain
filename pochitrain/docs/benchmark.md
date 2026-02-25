# 推論ベンチマーク

## 計測メタ情報

- 最終更新日: 2026-02-24
- 時刻表記: `YYYY-MM-DD HH:MM:SS` (JST)

## 指標定義

- 推論: 純粋推論時間 (ms/image). 転送・I/O を除外.
- E2E: 全処理時間 (ms/image). I/O・前処理・転送・推論・後処理を含む.

## 共通計測条件

- 計測対象: `infer-trt`, `infer-onnx`
- パイプライン: `pipeline=gpu`
- 入力解像度: `512x512`
- バッチサイズ: `1`
- モデル: `resnet18`, `resnet50`
- 各条件 10 回計測し `Ave` を採用

## 運用手順

### 1. suites 設定

- `tools/benchmark/suites.yaml` の `suites.<suite_name>` を編集する.
- 主要項目:
  - `repeats`: 反復回数
  - `defaults.pipelines`: `gpu`, `fast`, `current` など
  - `defaults.model_paths`: `trt`, `onnx` のモデルパス
  - `cases`: 実行ケース名と runtime

### 2. ベンチ実行

```bash
uv run python tools/benchmark/run_benchmark.py --suite <suite_name>
```

### 3. 再集計のみ実行

```bash
uv run python tools/benchmark/run_benchmark.py --aggregate-only --input-dir benchmark_runs/<suite_name>_<timestamp>
```

### 4. 出力物

- `benchmark_runs/<suite_name>_<timestamp>/raw/`
  - ケースごとの実行結果 (`benchmark_result.json`)
- `benchmark_runs/<suite_name>_<timestamp>/summary/`
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
- `benchmark_runs/<suite_name>_<timestamp>/configs/`
  - 実行時に参照した `config.py` のコピー

## 1. gpu_non_blocking A/B

### 1.1 resnet18

#### Windows 11 + RTX4070Ti

- 計測日: 2026-02-22

| Runtime | gpu_non_blocking | 推論 avg ms/image | E2E avg ms/image |
| --- | --- | ---: | ---: |
| TensorRT INT8 | True | 0.29 | 1.70 |
| TensorRT INT8 | False | 0.28 | 1.95 |
| ONNX FP32 | True | 1.48 | 2.90 |
| ONNX FP32 | False | 1.47 | 3.17 |

所見.
- TensorRT は `gpu_non_blocking=True` で E2E が改善.
- ONNX も `gpu_non_blocking=True` で E2E が改善.
- 推論時間は差が小さく, 主に E2E 側で効果が出る.

#### Jetson Orin Nano

- 計測日: 2026-02-22

| Runtime | gpu_non_blocking | 推論 avg ms/image | E2E avg ms/image |
| --- | --- | ---: | ---: |
| TensorRT INT8 | True | 3.36 | 10.32 |
| TensorRT INT8 | False | 3.36 | 10.99 |
| ONNX FP32 | True | 20.88 | 28.08 |
| ONNX FP32 | False | 23.05 | 31.34 |

所見.
- TensorRT は `gpu_non_blocking=True` で E2E が改善.
- ONNX は `gpu_non_blocking=True` で推論/E2E とも改善.

### 1.2 resnet50

#### Windows 11 + RTX4070Ti

- 計測日: 2026-02-22

| Runtime | gpu_non_blocking | 推論 avg ms/image | E2E avg ms/image |
| --- | --- | ---: | ---: |
| TensorRT INT8 | True | 0.28 | 1.72 |
| TensorRT INT8 | False | 0.29 | 1.97 |
| ONNX FP32 | True | 3.47 | 5.04 |
| ONNX FP32 | False | 3.47 | 5.06 |

所見.
- `resnet50` でも `gpu_non_blocking=True` の E2E 改善傾向を確認.

## 2. pin_memory A/B

### 2.1 resnet18

#### Windows 11 + RTX4070Ti

- 計測日: 2026-02-24
- 条件: `image_size=512x512`, `batch_size=1`, `runs=3`, `device=cuda`

| Runtime | Pipeline | pin_memory | 推論 avg ms/image | E2E avg ms/image | Accuracy avg % |
| --- | --- | --- | ---: | ---: | ---: |
| TensorRT INT8 | current | True | 1.5598 | 8.7788 | 100.00 |
| TensorRT INT8 | current | False | 1.5717 | 9.2366 | 100.00 |
| TensorRT INT8 | fast | True | 1.5559 | 7.7838 | 100.00 |
| TensorRT INT8 | fast | False | 1.5702 | 9.5739 | 100.00 |
| TensorRT INT8 | gpu | True | 1.0911 | 4.5879 | 100.00 |
| TensorRT INT8 | gpu | False | 1.0977 | 4.4112 | 100.00 |
| ONNX FP32 | current | True | 13.5728 | 21.2122 | 100.00 |
| ONNX FP32 | current | False | 13.5679 | 20.7834 | 100.00 |
| ONNX FP32 | fast | True | 13.5734 | 20.6670 | 100.00 |
| ONNX FP32 | fast | False | 13.5809 | 20.2642 | 100.00 |
| ONNX FP32 | gpu | True | 12.1551 | 16.5040 | 92.98 |
| ONNX FP32 | gpu | False | 12.2021 | 16.4108 | 92.98 |
| PyTorch FP32 | current | True | 14.2258 | 22.7227 | 100.00 |
| PyTorch FP32 | current | False | 14.2264 | 21.1343 | 100.00 |
| PyTorch FP32 | fast | True | 14.1536 | 20.5654 | 100.00 |
| PyTorch FP32 | fast | False | 14.1993 | 20.6883 | 100.00 |
| PyTorch FP32 | gpu | True | 13.3934 | 16.9591 | 100.00 |
| PyTorch FP32 | gpu | False | 13.3203 | 16.8297 | 100.00 |

所見.
- `pin_memory` は本計測条件では大差なしで, runtime と pipeline により揺らぐ.
- TensorRT は `current/fast` で `True` 優位, `gpu` は `False` 優位で一貫しない.
- ONNX と PyTorch は `True/False` で概ね同等.
- ONNX `pipeline=gpu` の `accuracy=92.98%` は `pin_memory` の値に依存せず再現しており, 別要因として切り分けが必要.

推奨設定.
- デフォルトは `train_pin_memory=True`, `infer_pin_memory=True` を維持する.
- Jetson を含む実機運用では, `benchmark` の実測を基準に個別に切り替える.
