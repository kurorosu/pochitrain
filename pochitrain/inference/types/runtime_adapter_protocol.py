"""ランタイムアダプタの Protocol 定義.

IRuntimeAdapter は `typing.Protocol` を採用する.
主な理由は次のとおり.

- 構造的型付けにより, 必要メソッドを満たす実装を柔軟に受け入れられる.
- テストで Fake / Stub を差し替える際に, 明示的な継承が不要で記述負荷が低い.
- ランタイム実装の差分を最小化しつつ, mypy で静的検証できる.
"""

from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from pochitrain.inference.types.execution_types import ExecutionRequest


class IRuntimeAdapter(Protocol):
    """推論ランタイム差分を吸収するアダプタインターフェース."""

    @property
    def use_cuda_timing(self) -> bool:
        """CUDA Event 計測を使用可能か返す.

        Returns:
            CUDA Event 計測を使用可能ならTrue.
        """
        ...

    def get_timing_stream(self) -> torch.cuda.Stream | None:
        """CUDA Event 計測対象ストリームを返す.

        Returns:
            計測対象ストリーム. 指定しない場合はNone.
        """
        ...

    def warmup(self, image: Tensor, request: ExecutionRequest) -> None:
        """単一画像でウォームアップを実行する.

        Args:
            image: 単一画像テンソル (C,H,W).
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
