"""
pochitrain.pochi_predictor: 推論機能のメインモジュール.

学習済みモデルを読み込んで推論を実行する機能を提供します.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .config import PochiConfig
from .logging import LoggerManager
from .models.pochi_models import create_model
from .utils.inference_utils import post_process_logits


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
        # モデル設定の保存
        self.model_name = model_name
        self.num_classes = num_classes

        # デバイスの設定
        self.device = torch.device(device)

        # ロガーの設定
        logger_manager = LoggerManager()
        self.logger: logging.Logger = logger_manager.get_logger("pochitrain")

        # モデルの作成（推論用のため事前学習済み重みは不要）
        self.model = create_model(model_name, num_classes, pretrained=False)
        self.model.to(self.device)

        # 訓練メタ情報（チェックポイントから復元される）
        self.best_accuracy = 0.0
        self.epoch = 0

        # 学習済みモデルの読み込み
        self.model_path = Path(model_path)
        self._load_trained_model()

    def _load_trained_model(self) -> None:
        """学習済みモデルを読み込み."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {self.model_path}"
            )

        try:
            # チェックポイントの読み込み
            try:
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                # 古いPyTorchとの互換のため, weights_only未対応時のみフォールバック
                checkpoint = torch.load(self.model_path, map_location=self.device)

            # モデルの状態辞書を読み込み
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()  # 推論モードに設定

            # メタ情報の取得
            if "best_accuracy" in checkpoint:
                self.best_accuracy = checkpoint["best_accuracy"]
                self.logger.debug(f"学習時の最高精度: {self.best_accuracy:.2f}%")

            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
                self.logger.debug(f"学習エポック数: {self.epoch}")

            self.logger.debug(f"学習済みモデルを読み込み: {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"モデルの読み込みに失敗しました: {e}")

    def predict(
        self,
        data_loader: DataLoader[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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
        predictions: List[Any] = []
        confidences: List[Any] = []
        total_samples = 0
        warmup_samples = 0
        inference_time_ms = 0.0

        use_cuda = self.device.type == "cuda"

        # ウォームアップ: 最初のバッチで10回事前実行
        # dataset[0]を直接アクセスし, DataLoaderのイテレーションに影響させない
        warmup_data, _ = data_loader.dataset[0]
        if not isinstance(warmup_data, torch.Tensor):
            warmup_data = torch.tensor(warmup_data)
        warmup_data = warmup_data.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            for _ in range(10):
                self.model(warmup_data)
            if use_cuda:
                torch.cuda.synchronize()

        # CUDA Eventをループ外で1回だけ生成
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # Note: ONNX/TRTと異なり, PyTorchは事前確保バッファを持たないため,
        # to(device)で毎回CUDAメモリアロケーションが発生する.
        # この転送コストと .cpu().numpy() のD2H転送は計測対象外だが,
        # E2E時間には含まれるため, E2E - 純粋推論 の差が大きくなる.
        # inference_mode: no_gradに加えテンソルのメタデータ追跡も無効化.
        # 推論専用のため採用. 訓練側の検証ループ(evaluator.py)は
        # 現在no_gradだが, 同様にinference_modeへ移行可能.
        with torch.inference_mode():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)  # H2D転送（計測外）
                batch_size = data.size(0)

                if batch_idx == 0:
                    # 最初のバッチは計測対象外（ウォームアップ）
                    output = self.model(data)
                    if use_cuda:
                        torch.cuda.synchronize()
                    warmup_samples = batch_size
                else:
                    # 推論時間計測（モデル推論部分のみ）
                    if use_cuda:
                        start_event.record()
                        output = self.model(data)
                        end_event.record()
                        torch.cuda.synchronize()
                        inference_time_ms += start_event.elapsed_time(end_event)
                    else:
                        start_time = time.perf_counter()
                        output = self.model(data)
                        inference_time_ms += (time.perf_counter() - start_time) * 1000

                    total_samples += batch_size

                # 後処理（計測対象外）
                logits = output.cpu().numpy()  # D2H転送 + 暗黙同期
                predicted, confidence = post_process_logits(logits)

                predictions.extend(predicted)
                confidences.extend(confidence)

        avg_time_per_image = 0.0
        if total_samples > 0:
            avg_time_per_image = inference_time_ms / total_samples

        metrics = {
            "avg_time_per_image": avg_time_per_image,
            "total_samples": total_samples,
            "warmup_samples": warmup_samples,
        }

        return torch.tensor(predictions), torch.tensor(confidences), metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得.

        Returns:
            Dict[str, any]: モデル情報
        """
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "model_path": str(self.model_path),
            "best_accuracy": self.best_accuracy,
            "epoch": self.epoch,
        }
