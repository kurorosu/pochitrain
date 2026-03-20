"""pochitrain.training.layer_wise_lr.param_group_builder: パラメータグループの構築."""

import logging
from typing import Any

import torch.nn as nn

from .interfaces import ILayerGrouper


class ParamGroupBuilder:
    """層別学習率のパラメータグループを構築する.

    ILayerGrouper を使用してパラメータを層グループに分類し,
    各グループに適切な学習率を割り当てる.
    """

    def __init__(self, layer_grouper: ILayerGrouper, logger: logging.Logger):
        """初期化.

        Args:
            layer_grouper: 層グルーパー.
            logger: ロガー.
        """
        self._layer_grouper = layer_grouper
        self._logger = logger

    def build(
        self,
        model: nn.Module,
        base_lr: float,
        layer_wise_lr_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """層別学習率のパラメータグループを構築.

        Args:
            model: 訓練対象のモデル.
            base_lr: 基本学習率.
            layer_wise_lr_config: 層別学習率の設定.

        Returns:
            list[dict[str, Any]]: パラメータグループのリスト.
        """
        layer_rates = layer_wise_lr_config.get("layer_rates", {})

        layer_params: dict[str, list] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_group = self._layer_grouper.get_group(name)
                if layer_group not in layer_params:
                    layer_params[layer_group] = []
                layer_params[layer_group].append(param)

        param_groups = []
        for layer_name, params in layer_params.items():
            lr = layer_rates.get(layer_name, base_lr)
            param_groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "layer_name": layer_name,
                }
            )

        return param_groups

    def log_param_groups(self, param_groups: list[dict[str, Any]]) -> None:
        """層別学習率の設定をログ出力.

        Args:
            param_groups: パラメータグループのリスト.
        """
        self._logger.debug("=== 層別学習率設定 ===")
        for group in param_groups:
            layer_name = group.get("layer_name", "unknown")
            lr = group["lr"]
            param_count = sum(p.numel() for p in group["params"])
            self._logger.debug(f"  {layer_name}: lr={lr:.6f}, params={param_count:,}")
        self._logger.debug("=====================")
