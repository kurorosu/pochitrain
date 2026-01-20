"""ONNXエクスポート機能を提供するモジュール."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

from pochitrain.logging import LoggerManager
from pochitrain.models.pochi_models import create_model

logger: logging.Logger = LoggerManager().get_logger(__name__)


class OnnxExporter:
    """PyTorchモデルをONNX形式にエクスポートするクラス.

    Attributes:
        model: PyTorchモデル
        device: 使用デバイス
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """OnnxExporterを初期化.

        Args:
            model: PyTorchモデル（後からload_modelで設定も可）
            device: 使用デバイス
        """
        self.model = model
        self.device = device or torch.device("cpu")

    def load_model(
        self,
        model_path: Path,
        model_name: str,
        num_classes: int,
    ) -> None:
        """チェックポイントからモデルを読み込む.

        Args:
            model_path: モデルファイルパス
            model_name: モデル名（resnet18, resnet34, resnet50）
            num_classes: 分類クラス数
        """
        self.model = create_model(model_name, num_classes, pretrained=False)
        self.model.to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "best_accuracy" in checkpoint:
                logger.info(f"モデル精度: {checkpoint['best_accuracy']:.2f}%")
            if "epoch" in checkpoint:
                logger.info(f"エポック: {checkpoint['epoch']}")
        else:
            self.model.load_state_dict(checkpoint)

        logger.info("モデルの読み込み完了")

    def export(
        self,
        output_path: Path,
        input_size: Tuple[int, int],
        opset_version: int = 17,
    ) -> Path:
        """モデルをONNX形式でエクスポート.

        Args:
            output_path: 出力ファイルパス
            input_size: 入力サイズ (height, width)
            opset_version: ONNXオペセットバージョン

        Returns:
            出力ファイルパス

        Raises:
            ValueError: モデルが設定されていない場合
        """
        if self.model is None:
            raise ValueError(
                "モデルが設定されていません. load_model()を先に呼び出してください."
            )

        self.model.eval()

        dummy_input = torch.randn(
            1, 3, input_size[0], input_size[1], device=self.device
        )

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            dynamo=False,
        )

        logger.info(f"ONNX変換完了: {output_path}")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ファイルサイズ: {file_size_mb:.2f} MB")

        return output_path

    def verify(
        self,
        onnx_path: Path,
        input_size: Tuple[int, int],
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """エクスポートしたONNXモデルを検証.

        Args:
            onnx_path: ONNXモデルのパス
            input_size: 入力サイズ (height, width)
            rtol: 相対許容誤差
            atol: 絶対許容誤差

        Returns:
            検証成功の場合True

        Raises:
            ValueError: モデルが設定されていない場合
        """
        if self.model is None:
            raise ValueError(
                "モデルが設定されていません. load_model()を先に呼び出してください."
            )

        # 1. ONNXモデルの構造検証
        logger.info("ONNXモデルの構造を検証中...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("構造検証: OK")

        # 2. PyTorchとONNXの出力比較
        logger.info("PyTorchとONNXの出力を比較中...")
        dummy_input = torch.randn(
            1, 3, input_size[0], input_size[1], device=self.device
        )

        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(dummy_input).cpu().numpy()

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run(None, {"input": dummy_input.cpu().numpy()})[0]

        is_close: bool = bool(
            np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
        )

        if is_close:
            logger.info("出力比較: OK")
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            logger.info(f"最大差分: {max_diff:.2e}")
        else:
            logger.warning("出力比較: NG")
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            logger.warning(f"最大差分: {max_diff:.2e}, 平均差分: {mean_diff:.2e}")

        return is_close
