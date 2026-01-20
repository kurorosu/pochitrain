"""ONNX推論機能を提供するモジュール."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

from pochitrain.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)


def check_gpu_availability() -> bool:
    """GPU(CUDAExecutionProvider)の利用可否をチェック.

    Returns:
        CUDAExecutionProviderが利用可能な場合True
    """
    available_providers = ort.get_available_providers()
    return "CUDAExecutionProvider" in available_providers


class OnnxInference:
    """ONNXモデルを使用した推論クラス.

    Attributes:
        session: ONNXランタイムセッション
        use_gpu: GPU使用フラグ
    """

    def __init__(self, model_path: Path, use_gpu: bool = False) -> None:
        """OnnxInferenceを初期化.

        Args:
            model_path: ONNXモデルファイルパス
            use_gpu: GPUを使用するか
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session: ort.InferenceSession = self._create_session()

    def _create_session(self) -> ort.InferenceSession:
        """ONNXセッションを作成.

        Returns:
            ONNXランタイムセッション
        """
        providers: List[str] = []
        if self.use_gpu:
            if check_gpu_availability():
                providers.append("CUDAExecutionProvider")
            else:
                logger.warning(
                    "CUDAExecutionProviderが利用できません. CPUで実行します."
                )
                logger.warning(
                    "GPUを使用するには onnxruntime-gpu をインストールしてください: "
                    "pip install onnxruntime-gpu"
                )
                self.use_gpu = False
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(str(self.model_path), providers=providers)
        logger.info(f"実行プロバイダー: {session.get_providers()}")
        return session

    def run(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """推論を実行.

        Args:
            images: 入力画像 (batch, channels, height, width)

        Returns:
            (予測クラス, 信頼度) のタプル
        """
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        outputs = self.session.run([output_name], {input_name: images})
        logits = outputs[0]

        # softmaxで確率に変換
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        predicted = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)

        return predicted, confidence

    def get_providers(self) -> List[str]:
        """使用中のプロバイダーを取得.

        Returns:
            プロバイダー名のリスト
        """
        providers: List[str] = self.session.get_providers()
        return providers
