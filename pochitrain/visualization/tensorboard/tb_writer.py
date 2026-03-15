"""pochitrain.visualization.tensorboard.tb_writer: TensorBoard メトリクス記録."""

import logging
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """TensorBoard へのメトリクス記録を管理するクラス.

    訓練中の loss, accuracy, 学習率を TensorBoard に記録する.

    Args:
        log_dir: TensorBoard ログ出力ディレクトリ.
        logger: ロガーインスタンス.
    """

    def __init__(self, log_dir: Path, logger: logging.Logger) -> None:
        """初期化."""
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._logger = logger
        self._logger.debug(f"TensorBoard ログディレクトリ: {log_dir}")

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        learning_rate: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        layer_wise_rates: Optional[dict[str, float]] = None,
    ) -> None:
        """1エポック分のメトリクスを TensorBoard に記録.

        Args:
            epoch: エポック番号.
            train_loss: 訓練損失.
            train_accuracy: 訓練精度.
            learning_rate: 現在の学習率.
            val_loss: 検証損失.
            val_accuracy: 検証精度.
            layer_wise_rates: 層別学習率の辞書.
        """
        self._writer.add_scalar("Loss/train", train_loss, epoch)
        self._writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        self._writer.add_scalar("LearningRate/base", learning_rate, epoch)

        if val_loss is not None:
            self._writer.add_scalar("Loss/val", val_loss, epoch)
        if val_accuracy is not None:
            self._writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        if layer_wise_rates:
            for layer_name, lr in layer_wise_rates.items():
                tag = f"LearningRate/{layer_name}"
                self._writer.add_scalar(tag, lr, epoch)

    def close(self) -> None:
        """ライターをフラッシュして閉じる."""
        self._writer.flush()
        self._writer.close()
        self._logger.debug("TensorBoard ライターを閉じました")
