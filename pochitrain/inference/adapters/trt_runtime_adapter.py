"""TensorRT推論をExecutionServiceへ接続するアダプタ."""

import torch

from pochitrain.tensorrt.inference import TensorRTInference

from .engine_runtime_adapter import EngineRuntimeAdapter


class TensorRTRuntimeAdapter(EngineRuntimeAdapter):
    """TensorRTInferenceをIRuntimeAdapterとして扱うためのラッパー."""

    def __init__(self, inference: TensorRTInference) -> None:
        """TensorRT推論インスタンスを受け取って初期化する.

        Args:
            inference: TensorRT推論インスタンス.
        """
        self.inference = inference

    @property
    def use_cuda_timing(self) -> bool:
        """CUDA Event計測の可否を返す.

        Returns:
            TensorRTは常にGPU実行のためTrue.
        """
        return True

    def get_timing_stream(self) -> torch.cuda.Stream:
        """CUDA Event 計測に使うストリームを返す.

        Returns:
            TensorRT 実行に使用する CUDA ストリーム.
        """
        stream: torch.cuda.Stream = self.inference.stream
        return stream

    def run_inference(self) -> None:
        """純粋推論を1回実行する."""
        self.inference.execute()
