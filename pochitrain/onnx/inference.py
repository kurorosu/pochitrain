"""ONNX推論機能を提供するモジュール."""

import logging
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import onnxruntime as ort

from pochitrain.logging import LoggerManager
from pochitrain.utils.inference_utils import post_process_logits

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
        io_binding: IO Bindingオブジェクト（GPU使用時のみ）
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

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.io_binding = None

        if self.use_gpu:
            self.io_binding = self.session.io_binding()

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
        logger.debug(f"実行プロバイダー: {session.get_providers()}")
        return session

    def set_input(self, images: np.ndarray) -> None:
        """入力を設定（GPUなら転送を含む）.

        Args:
            images: 入力画像 (batch, channels, height, width)
        """
        if self.io_binding:
            # GPU上のメモリにデータをバインド（H2D転送が発生）
            self.io_binding.bind_cpu_input(self.input_name, images)
            # 出力バッファもバインド
            self.io_binding.bind_output(self.output_name, "cuda")
        else:
            # CPUの場合は単純に保持
            self._temp_cpu_images = images

    def run_pure(self) -> None:
        """純粋な推論実行（計測対象）.

        IO Bindingを使用している場合、転送を含まない純粋な計算のみを行う.
        """
        if self.io_binding:
            self.session.run_with_iobinding(self.io_binding)
        else:
            # CPUの場合は通常のrunを実行（転送コストがないため）
            self._temp_cpu_outputs = self.session.run(
                [self.output_name], {self.input_name: self._temp_cpu_images}
            )

    def get_output(self) -> np.ndarray:
        """推論結果を取得（GPUなら取得転送を含む）.

        Returns:
            出力ロジット (batch, num_classes)
        """
        if self.io_binding:
            # GPUからCPUへ結果を取得（D2H転送が発生）
            outputs = self.io_binding.copy_outputs_to_cpu()
            return cast(np.ndarray, outputs[0])
        else:
            return cast(np.ndarray, self._temp_cpu_outputs[0])

    def run(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """推論を一括実行（互換性および簡易用）.

        Args:
            images: 入力画像 (batch, channels, height, width)

        Returns:
            (予測クラス, 信頼度) のタプル
        """
        self.set_input(images)
        self.run_pure()
        logits = self.get_output()

        return post_process_logits(logits)

    def get_providers(self) -> List[str]:
        """使用中のプロバイダーを取得.

        Returns:
            プロバイダー名のリスト
        """
        providers: List[str] = self.session.get_providers()
        return providers
