"""推論実行サービスで共有するデータ型."""

from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor


@dataclass(frozen=True)
class ExecutionRequest:
    """推論実行サービスへの入力パラメータ."""

    use_gpu_pipeline: bool
    mean_255: Optional[Tensor] = None
    std_255: Optional[Tensor] = None
    warmup_repeats: int = 10
    skip_measurement_batches: int = 1
    use_cuda_timing: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    """推論実行サービスの集計結果."""

    predictions: List[int]
    confidences: List[float]
    true_labels: List[int]
    total_inference_time_ms: float
    total_samples: int
    warmup_samples: int
    e2e_total_time_ms: float
