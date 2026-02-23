"""推論CLIオーケストレーションで共有する型定義."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from torch.utils.data import DataLoader

from .execution_types import ExecutionRequest, ExecutionResult

if TYPE_CHECKING:
    from pochitrain.inference.interfaces import IRuntimeAdapter


@dataclass(frozen=True)
class InferenceCliRequest:
    """推論CLIからオーケストレーション層へ渡す共通入力.

    Args:
        model_path: モデルファイルパス.
        data_path: 推論データディレクトリ. 未指定時は設定ファイルから解決する.
        output_dir: 結果出力ディレクトリ. 未指定時はモデル位置から自動解決する.
        requested_pipeline: ユーザー指定のパイプライン名.
    """

    model_path: Path
    data_path: Optional[Path]
    output_dir: Optional[Path]
    requested_pipeline: str


@dataclass(frozen=True)
class InferenceResolvedPaths:
    """推論前に解決済みのパス群.

    Args:
        model_path: モデルファイルパス.
        data_path: 推論データディレクトリ.
        output_dir: 結果出力ディレクトリ.
    """

    model_path: Path
    data_path: Path
    output_dir: Path


@dataclass(frozen=True)
class InferenceRuntimeOptions:
    """推論実行時の共通オプション.

    Args:
        pipeline: 解決後のパイプライン名.
        batch_size: DataLoaderのバッチサイズ.
        num_workers: DataLoaderのワーカー数.
        pin_memory: DataLoaderのpin_memory設定.
        use_gpu: 推論ランタイムがGPUを使うかどうか.
        use_gpu_pipeline: 前処理パイプラインがGPUを使うかどうか.
    """

    pipeline: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_gpu: bool
    use_gpu_pipeline: bool


@dataclass(frozen=True)
class RuntimeExecutionRequest:
    """ExecutionService に渡す実行コンテキスト."""

    data_loader: DataLoader[Any]
    runtime_adapter: "IRuntimeAdapter"
    execution_request: ExecutionRequest


@dataclass(frozen=True)
class PyTorchRunRequest:
    """PyTorch 推論実行のリクエスト."""

    predictor: Any
    val_loader: DataLoader[Any]


@dataclass(frozen=True)
class InferenceRunResult:
    """ランタイム実行結果と集計済みメトリクス."""

    predictions: list[int]
    confidences: list[float]
    true_labels: list[int]
    num_samples: int
    correct: int
    avg_time_per_image: float
    total_samples: int
    warmup_samples: int
    avg_total_time_per_image: float

    @property
    def accuracy_percent(self) -> float:
        """精度(%)."""
        if self.num_samples <= 0:
            return 0.0
        return (self.correct / self.num_samples) * 100.0

    @classmethod
    def from_execution_result(
        cls,
        execution_result: ExecutionResult,
    ) -> "InferenceRunResult":
        """実行結果から共通結果型を生成する."""
        return cls.from_components(
            predictions=execution_result.predictions,
            confidences=execution_result.confidences,
            true_labels=execution_result.true_labels,
            total_inference_time_ms=execution_result.total_inference_time_ms,
            total_samples=execution_result.total_samples,
            warmup_samples=execution_result.warmup_samples,
            e2e_total_time_ms=execution_result.e2e_total_time_ms,
        )

    @classmethod
    def from_components(
        cls,
        *,
        predictions: list[int],
        confidences: list[float],
        true_labels: list[int],
        total_inference_time_ms: float,
        total_samples: int,
        warmup_samples: int,
        e2e_total_time_ms: float,
    ) -> "InferenceRunResult":
        """各ランタイムの生データから共通結果型を生成する."""
        num_samples = len(true_labels)
        correct = sum(pred == label for pred, label in zip(predictions, true_labels))
        avg_time_per_image = (
            total_inference_time_ms / total_samples if total_samples > 0 else 0.0
        )
        avg_total_time_per_image = (
            e2e_total_time_ms / num_samples if num_samples > 0 else 0.0
        )
        return cls(
            predictions=predictions,
            confidences=confidences,
            true_labels=true_labels,
            num_samples=num_samples,
            correct=correct,
            avg_time_per_image=avg_time_per_image,
            total_samples=total_samples,
            warmup_samples=warmup_samples,
            avg_total_time_per_image=avg_total_time_per_image,
        )
