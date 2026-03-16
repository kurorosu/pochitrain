"""
pochitrain.pochi_predictor: 推論機能のメインモジュール.

学習済みモデルを読み込んで推論を実行する機能を提供します.
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import PochiConfig
from .logging import LoggerManager
from .models.pochi_models import create_model
from .utils.inference_utils import post_process_logits
from .utils.model_loading import load_model_from_checkpoint


class PochiPredictor:
    """
    推論専用クラス.

    学習済みモデルを読み込み, 推論に特化した機能を提供します.
    PochiTrainer とは独立したクラスで, 必要な機能のみを保持します.

    Args:
        model_name (str): モデル名 ('resnet18', 'resnet34', 'resnet50')
        num_classes (int): 分類クラス数
        device (str): デバイス ('cuda' or 'cpu')
        model_path (str): 学習済みモデルのパス
    """

    @classmethod
    def from_config(cls, config: PochiConfig, model_path: str) -> "PochiPredictor":
        """PochiConfigから推論器を作成."""
        return cls(
            model_name=config.model_name,
            num_classes=config.num_classes,
            device=config.device,
            model_path=model_path,
        )

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str,
        model_path: str,
    ):
        """PochiPredictorを初期化."""
        self.model_name = model_name
        self.num_classes = num_classes

        self.device = torch.device(device)

        logger_manager = LoggerManager()
        self.logger: logging.Logger = logger_manager.get_logger("pochitrain")

        # 推論用のため事前学習済み重みは不要.
        self.model = create_model(model_name, num_classes, pretrained=False)
        self.model.to(self.device)

        # 訓練メタ情報はチェックポイントから復元される.
        self.best_accuracy = 0.0
        self.epoch = 0

        self.model_path = Path(model_path)
        self._load_trained_model()

    def _load_trained_model(self) -> None:
        """学習済みモデルを読み込み."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {self.model_path}"
            )

        try:
            metadata = load_model_from_checkpoint(
                model=self.model,
                checkpoint_path=self.model_path,
                device=self.device,
            )
            self.model.eval()  # 推論モードに設定

            if "best_accuracy" in metadata:
                self.best_accuracy = metadata["best_accuracy"]
                self.logger.debug(f"学習時の最高精度: {self.best_accuracy:.2f}%")

            if "epoch" in metadata:
                self.epoch = metadata["epoch"]
                self.logger.debug(f"学習エポック数: {self.epoch}")

            self.logger.debug(f"学習済みモデルを読み込み: {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"モデルの読み込みに失敗しました: {e}") from e

    def predict(
        self,
        data_loader: DataLoader[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        予測の実行.

        Args:
            data_loader (DataLoader): 予測データローダー

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                (予測値, 確信度, メトリクス)
                メトリクスには以下が含まれる:
                - avg_time_per_image: 1枚あたりの平均推論時間 (ms)
                - total_samples: 計測サンプル数
                - warmup_samples: ウォームアップ除外サンプル数
        """
        self.model.eval()
        use_cuda = self.device.type == "cuda"

        self._run_warmup(data_loader, use_cuda)

        predictions: list[Any] = []
        confidences: list[Any] = []
        total_samples = 0
        warmup_samples = 0
        inference_time_ms = 0.0

        # Note: ONNX/TRTと異なり, PyTorchは事前確保バッファを持たないため,
        # to(device)で毎回CUDAメモリアロケーションが発生する.
        # この転送コストと .cpu().numpy() のD2H転送は計測対象外だが,
        # E2E時間には含まれるため, E2E - 純粋推論 の差が大きくなる.
        with torch.inference_mode():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                batch_size = data.size(0)

                if batch_idx == 0:
                    output, warmup_samples = self._run_first_batch(
                        data, batch_size, use_cuda
                    )
                else:
                    output, elapsed = self._run_timed_batch(data, use_cuda)
                    inference_time_ms += elapsed
                    total_samples += batch_size

                logits = output.cpu().numpy()
                predicted, confidence = post_process_logits(logits)
                predictions.extend(predicted)
                confidences.extend(confidence)

        avg_time = inference_time_ms / total_samples if total_samples > 0 else 0.0
        metrics = {
            "avg_time_per_image": avg_time,
            "total_samples": total_samples,
            "warmup_samples": warmup_samples,
        }
        return torch.tensor(predictions), torch.tensor(confidences), metrics

    def _run_warmup(self, data_loader: DataLoader[Any], use_cuda: bool) -> None:
        """GPU/CPUキャッシュのウォームアップを実行."""
        warmup_data, _ = data_loader.dataset[0]
        if not isinstance(warmup_data, torch.Tensor):
            warmup_data = torch.tensor(warmup_data)
        warmup_data = warmup_data.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            for _ in range(10):
                self.model(warmup_data)
            if use_cuda:
                torch.cuda.synchronize()

    def _run_first_batch(
        self, data: torch.Tensor, batch_size: int, use_cuda: bool
    ) -> tuple[torch.Tensor, int]:
        """最初のバッチを計測対象外として実行."""
        output = self.model(data)
        if use_cuda:
            torch.cuda.synchronize()
        return output, batch_size

    def _run_timed_batch(
        self, data: torch.Tensor, use_cuda: bool
    ) -> tuple[torch.Tensor, float]:
        """1バッチの推論を実行し, 経過時間 (ms) を返す."""
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = self.model(data)
            end_event.record()
            torch.cuda.synchronize()
            return output, start_event.elapsed_time(end_event)

        start_time = time.perf_counter()
        output = self.model(data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return output, elapsed_ms

    def get_model_info(self) -> dict[str, Any]:
        """
        モデル情報を取得.

        Returns:
            dict[str, Any]: モデル情報
        """
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "model_path": str(self.model_path),
            "best_accuracy": self.best_accuracy,
            "epoch": self.epoch,
        }
