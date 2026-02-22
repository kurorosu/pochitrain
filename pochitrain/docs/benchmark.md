# 推論ベンチマーク

## 計測メタ情報

- 最終更新日: 2026-02-22
- 時刻表記: ローカル時刻

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

計測予定.

### 2.1 計測テンプレート

| Environment | Runtime | pin_memory=True 推論 | pin_memory=True E2E | pin_memory=False 推論 | pin_memory=False E2E |
| --- | --- | ---: | ---: | ---: | ---: |
| Windows 11 + RTX4070Ti | TensorRT INT8 | 未計測 | 未計測 | 未計測 | 未計測 |
| Windows 11 + RTX4070Ti | ONNX FP32 | 未計測 | 未計測 | 未計測 | 未計測 |
| Jetson Orin Nano | TensorRT INT8 | 未計測 | 未計測 | 未計測 | 未計測 |
| Jetson Orin Nano | ONNX FP32 | 未計測 | 未計測 | 未計測 | 未計測 |
