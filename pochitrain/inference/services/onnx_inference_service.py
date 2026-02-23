"""ONNX推論CLI向けのオーケストレーション補助サービス."""

from typing import Any, Dict, Optional

from pochitrain.inference.interfaces import IOnnxTrtInferenceService

from .execution_service import ExecutionService


class OnnxInferenceService(IOnnxTrtInferenceService):
    """ONNX推論CLIで必要な解決処理を提供するサービス."""

    execution_service_factory = ExecutionService

    def resolve_pipeline(
        self,
        requested: str,
        use_gpu: bool,
    ) -> str:
        """ONNX推論で実際に使うパイプライン名を解決する.

        Args:
            requested: ユーザー指定パイプライン.
            use_gpu: ONNX推論がGPUを使うかどうか.

        Returns:
            解決後パイプライン名.
        """
        if requested != "auto":
            if requested == "gpu" and not use_gpu:
                return "fast"
            return requested

        return "gpu" if use_gpu else "fast"

    def _resolve_batch_size(self, config: Dict[str, Any]) -> int:
        """ONNX推論時のバッチサイズを解決する."""
        return int(config.get("batch_size", 1))

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
