"""ONNX推論をExecutionServiceへ接続するアダプタ."""

import numpy as np
import torch
from torch import Tensor

from pochitrain.inference.interfaces import IRuntimeAdapter
from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.onnx.inference import OnnxInference
from pochitrain.pochi_dataset import gpu_normalize


class OnnxRuntimeAdapter(IRuntimeAdapter):
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
        return self.inference.use_gpu

    def get_timing_stream(self) -> torch.cuda.Stream | None:
        """CUDA Event 計測に使うストリームを返す.

        Returns:
            ONNX Runtime では専用ストリームを公開していないためNone.
        """
        return None

    def warmup(self, image: Tensor, request: ExecutionRequest) -> None:
        """単一画像でウォームアップを行う.

        Args:
            image: 単一画像テンソル (C,H,W).
            request: 実行パラメータ.
        """
        for _ in range(request.warmup_repeats):
            if request.use_gpu_pipeline:
                assert request.mean_255 is not None and request.std_255 is not None
                gpu_tensor = gpu_normalize(
                    image,
                    request.mean_255,
                    request.std_255,
                    non_blocking=request.gpu_non_blocking,
                )
                self.inference.set_input_gpu(gpu_tensor)
            else:
                image_np = image.numpy()[np.newaxis, ...]
                self.inference.set_input(image_np)

            self.inference.run_pure()
            self.inference.get_output()

    def set_input(self, images: Tensor, request: ExecutionRequest) -> None:
        """推論入力を設定する.

        Args:
            images: バッチ画像テンソル (N,C,H,W).
            request: 実行パラメータ.
        """
        if request.use_gpu_pipeline:
            assert request.mean_255 is not None and request.std_255 is not None
            gpu_tensor = gpu_normalize(
                images,
                request.mean_255,
                request.std_255,
                non_blocking=request.gpu_non_blocking,
            )
            self.inference.set_input_gpu(gpu_tensor)
            return

        self.inference.set_input(images.numpy())

    def run_inference(self) -> None:
        """純粋推論を1回実行する."""
        self.inference.run_pure()

    def get_output(self) -> np.ndarray:
        """推論結果ロジットを取得する.

        Returns:
            モデル出力ロジット.
        """
        return self.inference.get_output()
