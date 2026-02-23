"""TensorRT推論CLI向けのオーケストレーション補助サービス."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pochitrain.inference.adapters.trt_runtime_adapter import TensorRTRuntimeAdapter
from pochitrain.inference.services.interfaces import IInferenceOrchestrationService
from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import get_basic_transforms

from .execution_service import ExecutionService


class TensorRTInferenceService(IInferenceOrchestrationService):
    """TensorRT推論CLIで必要な解決処理を提供するサービス."""

    execution_service_factory = ExecutionService

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        super().__init__(logger=logger or LoggerManager().get_logger(__name__))

    def create_trt_inference(self, engine_path: Path) -> Any:
        """TensorRT推論インスタンスを生成する.

        Args:
            engine_path: TensorRTエンジンファイルパス.

        Returns:
            TensorRT推論インスタンス.
        """
        from pochitrain.tensorrt import TensorRTInference

        return TensorRTInference(engine_path)

    def resolve_val_transform(self, config: Dict[str, Any], inference: Any) -> Any:
        """Config またはエンジン入力形状から val_transform を解決する.

        Args:
            config: 推論設定辞書.
            inference: TensorRT推論インスタンス.

        Returns:
            解決した val_transform.
        """
        if "val_transform" in config:
            return config["val_transform"]

        engine_input_shape = inference.get_input_shape()
        height = engine_input_shape[2]
        width = engine_input_shape[3]
        self.logger.debug(f"入力サイズをエンジンから取得: {height}x{width}")
        return get_basic_transforms(image_size=height, is_training=False)

    def create_runtime_adapter(self, inference: Any) -> TensorRTRuntimeAdapter:
        """TensorRT推論インスタンスからランタイムアダプタを作成する.

        Args:
            inference: TensorRT推論インスタンス.

        Returns:
            TensorRTランタイムアダプタ.
        """
        return TensorRTRuntimeAdapter(inference)

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
