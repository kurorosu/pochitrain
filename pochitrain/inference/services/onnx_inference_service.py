"""ONNX推論CLI向けのオーケストレーション補助サービス."""

import logging
from pathlib import Path
from typing import Any, Optional

from pochitrain.inference.adapters.onnx_runtime_adapter import OnnxRuntimeAdapter
from pochitrain.inference.services.interfaces import IInferenceService
from pochitrain.logging import LoggerManager

from .execution_service import ExecutionService


class OnnxInferenceService(IInferenceService):
    """ONNX推論CLIで必要な解決処理を提供するサービス."""

    execution_service_factory = ExecutionService

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        super().__init__(logger=logger or LoggerManager().get_logger(__name__))

    def create_onnx_session(
        self,
        model_path: Path,
        use_gpu: bool,
    ) -> tuple[Any, bool]:
        """ONNX推論セッションを生成し, 実際のGPU利用可否を返す.

        Args:
            model_path: ONNXモデルファイルパス.
            use_gpu: GPU推論を要求するかどうか.

        Returns:
            ONNX推論インスタンスと実際のGPU利用可否.
        """
        from pochitrain.onnx import OnnxInference

        inference = OnnxInference(model_path, use_gpu=use_gpu)
        actual_use_gpu = inference.use_gpu
        if use_gpu and not actual_use_gpu:
            self.logger.warning(
                "CUDA ExecutionProviderが利用できません.CPUに切り替えます."
            )
        return inference, actual_use_gpu

    def create_runtime_adapter(self, inference: Any) -> OnnxRuntimeAdapter:
        """ONNX推論インスタンスからランタイムアダプタを作成する.

        Args:
            inference: ONNX推論インスタンス.

        Returns:
            ONNXランタイムアダプタ.
        """
        return OnnxRuntimeAdapter(inference)

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
