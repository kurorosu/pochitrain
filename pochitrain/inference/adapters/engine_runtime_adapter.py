"""ONNX/TRT 共通のアダプタ基底クラス.

set_input, warmup, get_output は両ランタイムで同一実装のため,
共通基底へ集約する. 各サブクラスは run_inference, use_cuda_timing,
get_timing_stream のみを実装する.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pochitrain.inference.types.execution_types import ExecutionRequest
from pochitrain.pochi_dataset import gpu_normalize


class EngineRuntimeAdapter(ABC):
    """ONNX/TRT 共通のランタイムアダプタ基底クラス.

    サブクラスは ``__init__`` で ``self.inference`` に具象推論インスタンスを
    設定する. 共通メソッドは ``set_input`` / ``set_input_gpu`` / ``get_output``
    のみを使用する.
    """

    inference: Any

    def _synchronize_input_if_needed(self) -> None:
        """入力バッファ同期フックを必要時のみ呼び出す."""
        synchronize = getattr(self.inference, "synchronize_input_if_needed", None)
        if callable(synchronize):
            synchronize()

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
                if request.gpu_non_blocking:
                    self._synchronize_input_if_needed()
            else:
                image_np = image.numpy()[np.newaxis, ...]
                self.inference.set_input(image_np)

            self.run_inference()
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
            if request.gpu_non_blocking:
                self._synchronize_input_if_needed()
            return

        self.inference.set_input(images.numpy())

    @abstractmethod
    def run_inference(self) -> None:
        """純粋推論を1回実行する."""

    @property
    @abstractmethod
    def use_cuda_timing(self) -> bool:
        """CUDA Event 計測を使用可能か返す."""

    @abstractmethod
    def get_timing_stream(self) -> torch.cuda.Stream | None:
        """CUDA Event 計測対象ストリームを返す."""

    def get_output(self) -> np.ndarray:
        """推論結果ロジットを取得する.

        Returns:
            モデル出力ロジット.
        """
        result: np.ndarray = self.inference.get_output()
        return result
