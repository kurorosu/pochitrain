"""TensorRT推論CLI向けのオーケストレーション補助サービス."""

from typing import Any, Dict, Optional

from pochitrain.inference.services.interfaces import IInferenceOrchestrationService

from .execution_service import ExecutionService


class TensorRTInferenceService(IInferenceOrchestrationService):
    """TensorRT推論CLIで必要な解決処理を提供するサービス."""

    execution_service_factory = ExecutionService

    def resolve_pipeline(self, requested: str, use_gpu: bool) -> str:
        """TensorRT推論で実際に使うパイプライン名を解決する.

        Args:
            requested: ユーザー指定パイプライン.
            use_gpu: GPU推論を使うかどうか（TensorRTでは未使用）.

        Returns:
            解決後パイプライン名.
        """
        _ = use_gpu
        if requested == "auto":
            return "gpu"
        return requested

    def _resolve_batch_size(self, config: Dict[str, Any]) -> int:
        """TensorRT推論時のバッチサイズを解決する."""
        return 1

    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        """TensorRT入力形状から入力サイズを解決する.

        Args:
            shape: TensorRT入力shape.

        Returns:
            入力サイズ (C, H, W). 解決できない場合はNone.
        """
        if len(shape) != 4:
            return None
        return (shape[1], shape[2], shape[3])
