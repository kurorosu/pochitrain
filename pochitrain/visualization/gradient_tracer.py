"""層ごとの勾配ノルムを記録するモジュール."""

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from pochitrain.logging import LoggerManager


class GradientTracer:
    """
    訓練中の各層の勾配ノルムを記録するクラス.

    Args:
        logger (logging.Logger, optional): ロガーインスタンス
        exclude_patterns (List[str], optional): 除外する層名パターン（正規表現）
        group_by_block (bool, optional): Trueの場合、ResNetブロック単位で集約
        aggregation_method (str, optional): 集約方法 ("median", "mean", "max", "rms")
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        exclude_patterns: Optional[List[str]] = None,
        group_by_block: bool = True,
        aggregation_method: str = "median",
    ):
        """GradientTracerを初期化."""
        self.gradient_history: Dict[str, List[float]] = {}
        self.epochs: List[int] = []
        self.layer_names: List[str] = []

        if logger is None:
            self.logger = LoggerManager().get_logger(__name__)
        else:
            self.logger = logger

        self.exclude_patterns = exclude_patterns if exclude_patterns else []
        self.group_by_block = group_by_block
        self.aggregation_method = aggregation_method.lower()

        valid_methods = ["median", "mean", "max", "rms"]
        if self.aggregation_method not in valid_methods:
            self.logger.warning(
                f"不明な集約方法: {self.aggregation_method}. 'median'を使用します。"
            )
            self.aggregation_method = "median"

    def _should_exclude(self, layer_name: str) -> bool:
        """
        層名が除外パターンに一致するかチェック.

        Args:
            layer_name (str): 層名

        Returns:
            bool: 除外する場合True
        """
        for pattern in self.exclude_patterns:
            if re.search(pattern, layer_name):
                return True
        return False

    def _get_group_name(self, layer_name: str) -> str:
        """
        層名をグループ名に変換（グループ化が有効な場合）.

        Args:
            layer_name (str): 元の層名

        Returns:
            str: グループ名
        """
        if not self.group_by_block:
            return layer_name

        # ResNetのlayer1〜layer4をグループ化
        # 例: "model.layer1.0.conv1.weight" -> "layer1"
        match = re.match(r"^(?:model\.)?([^.]+)", layer_name)
        if match:
            prefix = match.group(1)
            # layer1, layer2, layer3, layer4の場合はそのまま
            if re.match(r"^layer[1-4]$", prefix):
                return prefix
        return layer_name

    def _aggregate_gradients(self, grad_norms: List[float]) -> float:
        """
        勾配ノルムのリストを集約.

        Args:
            grad_norms (List[float]): 勾配ノルムのリスト

        Returns:
            float: 集約された勾配ノルム
        """
        if not grad_norms:
            return 0.0

        if self.aggregation_method == "mean":
            return sum(grad_norms) / len(grad_norms)
        elif self.aggregation_method == "median":
            sorted_norms = sorted(grad_norms)
            n = len(sorted_norms)
            if n % 2 == 0:
                return (sorted_norms[n // 2 - 1] + sorted_norms[n // 2]) / 2
            else:
                return sorted_norms[n // 2]
        elif self.aggregation_method == "max":
            return float(max(grad_norms))
        elif self.aggregation_method == "rms":
            return float((sum(x**2 for x in grad_norms) / len(grad_norms)) ** 0.5)
        else:
            return float(sum(grad_norms) / len(grad_norms))  # fallback to mean

    def record_gradients(self, model: nn.Module, epoch: int) -> None:
        """
        現在のモデルから各層の勾配ノルムを記録.

        Args:
            model (nn.Module): 訓練されたモデル
            epoch (int): 現在のエポック番号
        """
        self.epochs.append(epoch)

        grouped_gradients: Dict[str, List[float]] = defaultdict(list)

        for name, param in model.named_parameters():
            if self._should_exclude(name):
                continue

            if param.grad is not None:
                grad_norm = param.grad.norm().item()
            else:
                grad_norm = 0.0

            group_name = self._get_group_name(name)
            grouped_gradients[group_name].append(grad_norm)

        for group_name, grad_norms in grouped_gradients.items():
            aggregated_norm = self._aggregate_gradients(grad_norms)

            if group_name not in self.gradient_history:
                self.gradient_history[group_name] = []
                self.layer_names.append(group_name)

            self.gradient_history[group_name].append(aggregated_norm)

        self.logger.debug(
            f"Epoch {epoch}: 勾配ノルムを記録しました（{len(self.layer_names)}層）"
        )

    def save_csv(self, output_path: Path) -> None:
        """
        記録した勾配データをCSVファイルに保存.

        Args:
            output_path (Path): 出力CSVファイルパス
        """
        if not self.epochs:
            self.logger.warning("記録されたデータがありません")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            header = ["epoch"] + self.layer_names
            writer.writerow(header)

            for i, epoch in enumerate(self.epochs):
                row: List[Union[int, float]] = [epoch]
                for layer_name in self.layer_names:
                    row.append(self.gradient_history[layer_name][i])
                writer.writerow(row)

        self.logger.info(f"勾配トレースCSVを保存: {output_path}")
        self.logger.info(f"  - エポック数: {len(self.epochs)}")
        self.logger.info(f"  - 記録層数: {len(self.layer_names)}")

    def get_summary(self) -> Dict[str, Any]:
        """
        記録データのサマリー情報を取得.

        Returns:
            Dict[str, Any]: サマリー情報
        """
        if not self.epochs:
            return {}

        summary = {
            "total_epochs": len(self.epochs),
            "total_layers": len(self.layer_names),
            "layer_names": self.layer_names.copy(),
        }

        layer_stats = {}
        for layer_name in self.layer_names:
            grad_norms = self.gradient_history[layer_name]
            layer_stats[layer_name] = {
                "mean": sum(grad_norms) / len(grad_norms),
                "max": max(grad_norms),
                "min": min(grad_norms),
                "initial": grad_norms[0],
                "final": grad_norms[-1],
            }
        summary["layer_stats"] = layer_stats

        return summary
