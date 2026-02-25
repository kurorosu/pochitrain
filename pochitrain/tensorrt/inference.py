"""TensorRT推論機能を提供するモジュール.

単一入力・単一出力のCNN分類モデル (ResNet等) のみをサポートする.
マルチ入出力モデルには対応していない.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

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

    単一入力・単一出力のCNN分類モデル専用.
    PyTorch CUDAテンソルをバッファとして使用するため, pycudaは不要.

    Attributes:
        engine_path: TensorRTエンジンファイルパス
        engine: TensorRTエンジン
        context: TensorRT実行コンテキスト
        input_name: 入力テンソル名
        output_name: 出力テンソル名
    """

    def __init__(self, engine_path: Path) -> None:
        """TensorRTInferenceを初期化.

        Args:
            engine_path: TensorRTエンジンファイルパス (.engine)

        Raises:
            ImportError: TensorRTがインストールされていない場合
            FileNotFoundError: エンジンファイルが見つからない場合
            RuntimeError: エンジンの読み込みまたはI/Oバインディング解決に失敗した場合
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

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"エンジンの読み込みに失敗しました: {engine_path}")

        self.context = self.engine.create_execution_context()

        bindings = self._resolve_io_bindings(trt)
        self.input_name: str = bindings["input"]
        self.output_name: str = bindings["output"]

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        self._d_input = torch.empty(
            self.input_shape, dtype=torch.float32, device="cuda"
        )
        self._d_output = torch.empty(
            self.output_shape, dtype=torch.float32, device="cuda"
        )

        self.context.set_tensor_address(self.input_name, self._d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self._d_output.data_ptr())

        # 非デフォルトCUDAストリームを作成・保持
        # デフォルトストリーム(stream 0)を使うと, execute_async_v3内部で
        # 毎回cudaStreamSynchronize()が追加呼び出しされ性能低下するため
        self._stream = torch.cuda.Stream()

        logger.debug(f"TensorRTエンジンを読み込み: {engine_path}")
        logger.debug(f"入力: {self.input_name}, shape: {self.input_shape}")
        logger.debug(f"出力: {self.output_name}, shape: {self.output_shape}")

    @property
    def stream(self) -> torch.cuda.Stream:
        """CUDAストリームを取得.

        Returns:
            推論に使用するCUDAストリーム
        """
        return self._stream

    def _resolve_io_bindings(self, trt: object) -> Dict[str, str]:
        """名前ベースで入出力バインディングを解決する.

        インデックス順序に依存せず, TensorIOMode で入力・出力を判定する.
        単一入力・単一出力のみサポートし, それ以外はエラーとする.

        Args:
            trt: tensorrt モジュール

        Returns:
            {"input": 入力テンソル名, "output": 出力テンソル名}

        Raises:
            RuntimeError: 入力または出力が1つでない場合
        """
        input_names = []
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:  # type: ignore[attr-defined]
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:  # type: ignore[attr-defined]
                output_names.append(name)

        if len(input_names) != 1:
            raise RuntimeError(
                f"単一入力のみサポートしていますが, "
                f"{len(input_names)}個の入力が検出されました: {input_names}"
            )
        if len(output_names) != 1:
            raise RuntimeError(
                f"単一出力のみサポートしていますが, "
                f"{len(output_names)}個の出力が検出されました: {output_names}"
            )

        return {"input": input_names[0], "output": output_names[0]}

    def set_input(self, image: np.ndarray) -> None:
        """入力を設定（GPUへの転送を含む）.

        Args:
            image: 入力画像 (1, channels, height, width)
        """
        with torch.cuda.stream(self._stream):
            self._d_input.copy_(torch.from_numpy(image))

    def set_input_gpu(self, tensor: torch.Tensor) -> None:
        """GPU上のテンソルを直接入力として設定.

        GPU-to-GPUコピーのみ行い, CPU経由のH2D転送をスキップする.
        事前にデフォルトストリームの完了を待機して, ストリーム間競合を防ぐ.

        Args:
            tensor: GPU上のfloat32テンソル (1, channels, height, width)
        """
        self._stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(self._stream):
            self._d_input.copy_(tensor)

    def execute(self) -> None:
        """純粋な推論実行（計測対象）.

        GPU上のバッファに対して計算命令のみを行う.
        テンソルアドレスは__init__で事前設定済みのため, ストリーム指定のみで実行する.
        """
        self.context.execute_async_v3(self._stream.cuda_stream)

    def get_output(self) -> np.ndarray:
        """推論結果を取得（GPUからの取得転送を含む）.

        Returns:
            出力ロジット (1, num_classes)
        """
        self._stream.synchronize()
        return self._d_output.cpu().numpy()

    def run(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """推論を実行.

        Args:
            images: 入力画像 (batch, channels, height, width)

        Returns:
            (予測クラス, 信頼度) のタプル
        """
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
