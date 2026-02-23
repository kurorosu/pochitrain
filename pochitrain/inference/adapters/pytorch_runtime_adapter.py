"""PyTorch推論をExecutionServiceへ接続するアダプタ."""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from pochitrain.inference.interfaces import IRuntimeAdapter
from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.pochi_predictor import PochiPredictor


class PyTorchRuntimeAdapter(IRuntimeAdapter):
    """PochiPredictorをIRuntimeAdapterとして扱うためのラッパー."""

    def __init__(self, predictor: PochiPredictor) -> None:
        """PyTorch推論インスタンスを受け取って初期化する.

        Args:
            predictor: PyTorch推論インスタンス.
        """
        self.predictor = predictor
        self._input_batch: Tensor | None = None
        self._output: Tensor | None = None

    @property
    def use_cuda_timing(self) -> bool:
        """CUDA Event計測の可否を返す.

        Returns:
            CUDA実行時はTrue.
        """
        device_type = str(getattr(self.predictor.device, "type", "cpu"))
        return device_type == "cuda"

    def get_timing_stream(self) -> torch.cuda.Stream | None:
        """CUDA Event計測に使うストリームを返す.

        Returns:
            PyTorch推論では専用ストリームを使わないためNone.
        """
        return None

    def warmup(self, image: Tensor, request: ExecutionRequest) -> None:
        """単一画像でウォームアップを行う.

        Args:
            image: 単一画像テンソル (C,H,W).
            request: 実行パラメータ.
        """
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        batch = image.to(self.predictor.device, non_blocking=request.gpu_non_blocking)
        self.predictor.model.eval()
        with torch.inference_mode():
            for _ in range(request.warmup_repeats):
                self.predictor.model(batch)
        if self.use_cuda_timing:
            torch.cuda.synchronize()

    def set_input(self, images: Tensor, request: ExecutionRequest) -> None:
        """推論入力を設定する.

        Args:
            images: バッチ画像テンソル (N,C,H,W).
            request: 実行パラメータ.
        """
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        self._input_batch = images.to(
            self.predictor.device,
            non_blocking=request.gpu_non_blocking,
        )

    def run_inference(self) -> None:
        """純粋推論を1回実行する."""
        if self._input_batch is None:
            raise RuntimeError("推論入力が設定されていません")

        self.predictor.model.eval()
        with torch.inference_mode():
            self._output = self.predictor.model(self._input_batch)

    def get_output(self) -> np.ndarray[Any, Any]:
        """推論結果ロジットを取得する.

        Returns:
            モデル出力ロジット.
        """
        if self._output is None:
            raise RuntimeError("推論結果がありません")
        return self._output.detach().cpu().numpy()
