"""ONNX推論CLI向けのオーケストレーション補助サービス."""

from typing import Any, Optional

from pochitrain.inference.interfaces import IInferenceOrchestrationService

from .execution_service import ExecutionService


class OnnxInferenceService(IInferenceOrchestrationService):
    """ONNX推論CLIで必要な解決処理を提供するサービス."""

    execution_service_factory = ExecutionService

    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        """ONNX入力形状から入力サイズを解決する.

        Args:
            shape: ONNXの入力shape.

        Returns:
            入力サイズ (C, H, W). 解決できない場合はNone.
        """
        if len(shape) != 4:
            return None
        c = shape[1] if isinstance(shape[1], int) else 3
        h = shape[2] if isinstance(shape[2], int) else None
        w = shape[3] if isinstance(shape[3], int) else None
        if h and w:
            return (c, h, w)
        return None
