"""pochitrain.training.epoch_runner: 1エポック訓練実行モジュール."""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class EpochRunner:
    """1エポック分の訓練処理を実行する.

    Args:
        device: 訓練デバイス.
        logger: ロガーインスタンス.
    """

    def __init__(self, device: torch.device, logger: logging.Logger) -> None:
        """EpochRunnerを初期化."""
        self.device = device
        self.logger = logger

    def run(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader[Any],
        epoch: int,
    ) -> Dict[str, float]:
        """1エポック訓練を実行.

        Args:
            model: 訓練対象モデル.
            optimizer: オプティマイザ.
            criterion: 損失関数.
            train_loader: 訓練データローダー.
            epoch: 現在のエポック番号.

        Returns:
            Dict[str, float]: {"loss": 平均損失, "accuracy": 精度}.
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = output.max(1)
            total += batch_size
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                self.logger.debug(
                    f"エポック {epoch}, バッチ {batch_idx}/{len(train_loader)}, "
                    f"損失: {loss.item():.4f}, 精度: {100.0 * correct / total:.2f}%"
                )

        # 例外回避のための防御的ガード. 本来はバリデーションで止めるのが望ましい.
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}
