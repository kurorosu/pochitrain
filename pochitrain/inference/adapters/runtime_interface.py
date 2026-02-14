"""推論実行サービスが依存するランタイム抽象."""

from typing import Protocol

import numpy as np
from torch import Tensor

from pochitrain.inference.types.execution_types import ExecutionRequest


class IRuntimeAdapter(Protocol):
    """推論ランタイム差分を吸収する抽象インターフェース."""

    @property
    def use_cuda_timing(self) -> bool:
        """CUDA Event計測を使用するかどうかを返す.

        Returns:
            CUDA Event計測を使用可能ならTrue.
        """
        ...

    def warmup(self, image: Tensor, request: ExecutionRequest) -> None:
        """単一画像でウォームアップを実行する.

        Args:
            image: 単一画像テンソル (C,H,W) を想定.
            request: 実行パラメータ.
        """
        ...

    def set_input(self, images: Tensor, request: ExecutionRequest) -> None:
        """バッチ入力をランタイムへ設定する.

        Args:
            images: バッチ画像テンソル (N,C,H,W).
            request: 実行パラメータ.
        """
        ...

    def run_inference(self) -> None:
        """純粋推論を1回実行する."""
        ...

    def get_output(self) -> np.ndarray:
        """推論結果ロジットを取得する.

        Returns:
            モデル出力ロジット.
        """
        ...
