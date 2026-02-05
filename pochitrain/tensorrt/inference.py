"""TensorRT推論機能を提供するモジュール."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from pochitrain.logging import LoggerManager
from pochitrain.utils.inference_utils import post_process_logits

logger: logging.Logger = LoggerManager().get_logger(__name__)


def check_tensorrt_availability() -> bool:
    """TensorRTの利用可否をチェック.

    Returns:
        TensorRTが利用可能な場合True
    """
    try:
        import tensorrt as trt  # noqa: F401

        return True
    except ImportError:
        return False


class TensorRTInference:
    """TensorRTエンジンを使用した高速推論クラス.

    PyTorch CUDAテンソルをバッファとして使用するため、pycudaは不要.

    Attributes:
        engine_path: TensorRTエンジンファイルパス
        engine: TensorRTエンジン
        context: TensorRT実行コンテキスト
    """

    def __init__(self, engine_path: Path) -> None:
        """TensorRTInferenceを初期化.

        Args:
            engine_path: TensorRTエンジンファイルパス (.engine)

        Raises:
            ImportError: TensorRTがインストールされていない場合
            FileNotFoundError: エンジンファイルが見つからない場合
            RuntimeError: エンジンの読み込みに失敗した場合
        """
        if not check_tensorrt_availability():
            raise ImportError(
                "TensorRTがインストールされていません. "
                "TensorRT SDKをインストールしてください."
            )

        import tensorrt as trt

        self.engine_path = Path(engine_path)

        if not self.engine_path.exists():
            raise FileNotFoundError(f"エンジンファイルが見つかりません: {engine_path}")

        # エンジン読み込み
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"エンジンの読み込みに失敗しました: {engine_path}")

        self.context = self.engine.create_execution_context()

        # 入出力shape取得
        self.input_shape = self._get_binding_shape(0)
        self.output_shape = self._get_binding_shape(1)

        # PyTorch CUDAテンソルをバッファとして確保
        self._d_input = torch.empty(
            self.input_shape, dtype=torch.float32, device="cuda"
        )
        self._d_output = torch.empty(
            self.output_shape, dtype=torch.float32, device="cuda"
        )

        logger.debug(f"TensorRTエンジンを読み込み: {engine_path}")
        logger.debug(f"入力shape: {self.input_shape}")
        logger.debug(f"出力shape: {self.output_shape}")

    def _get_binding_shape(self, binding_idx: int) -> Tuple[int, ...]:
        """バインディングのshapeを取得.

        Args:
            binding_idx: バインディングインデックス

        Returns:
            shape のタプル
        """
        # TensorRT 10.x API
        tensor_name = self.engine.get_tensor_name(binding_idx)
        shape = self.engine.get_tensor_shape(tensor_name)
        return tuple(shape)

    def set_input(self, image: np.ndarray) -> None:
        """入力を設定（GPUへの転送を含む）.

        Args:
            image: 入力画像 (1, channels, height, width)
        """
        self._d_input.copy_(torch.from_numpy(image))

    def execute(self) -> None:
        """純粋な推論実行（計測対象）.

        GPU上のバッファに対して計算命令のみを行う.
        """
        self.context.execute_v2([self._d_input.data_ptr(), self._d_output.data_ptr()])

    def get_output(self) -> np.ndarray:
        """推論結果を取得（GPUからの取得転送を含む）.

        Returns:
            出力ロジット (1, num_classes)
        """
        return self._d_output.cpu().numpy()

    def run(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """推論を実行.

        Args:
            images: 入力画像 (batch, channels, height, width)

        Returns:
            (予測クラス, 信頼度) のタプル
        """
        # バッチ処理（現在はバッチサイズ1のみ対応）
        batch_size = images.shape[0]
        all_logits = []

        for i in range(batch_size):
            image = images[i : i + 1].astype(np.float32)
            self.set_input(image)
            self.execute()
            logits = self.get_output()
            all_logits.append(logits)

        logits = np.concatenate(all_logits, axis=0)

        return post_process_logits(logits)

    def get_input_shape(self) -> Tuple[int, ...]:
        """入力shapeを取得.

        Returns:
            入力shapeのタプル (batch, channels, height, width)
        """
        return self.input_shape

    def get_output_shape(self) -> Tuple[int, ...]:
        """出力shapeを取得.

        Returns:
            出力shapeのタプル (batch, num_classes)
        """
        return self.output_shape
