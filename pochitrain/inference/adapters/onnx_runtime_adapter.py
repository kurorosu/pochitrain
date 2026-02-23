"""ONNX推論をExecutionServiceへ接続するアダプタ."""

import torch

from pochitrain.onnx.inference import OnnxInference

from .engine_runtime_adapter import EngineRuntimeAdapter


class OnnxRuntimeAdapter(EngineRuntimeAdapter):
    """OnnxInferenceをIRuntimeAdapterとして扱うためのラッパー."""

    def __init__(self, inference: OnnxInference) -> None:
        """ONNX推論インスタンスを受け取って初期化する.

        Args:
            inference: ONNX推論インスタンス.
        """
        self.inference = inference

    @property
    def use_cuda_timing(self) -> bool:
        """CUDA Event計測の可否を返す.

        Returns:
            GPU推論が有効な場合True.
        """
        result: bool = self.inference.use_gpu
        return result

    def get_timing_stream(self) -> torch.cuda.Stream | None:
        """CUDA Event 計測に使うストリームを返す.

        Returns:
            ONNX Runtime では専用ストリームを公開していないためNone.
        """
        return None

    def run_inference(self) -> None:
        """純粋推論を1回実行する."""
        self.inference.run_pure()
