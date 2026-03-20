"""pochitrain.training.layer_wise_lr.interfaces: 層別学習率のインターフェース."""

from abc import ABC, abstractmethod


class ILayerGrouper(ABC):
    """パラメータ名から層グループ名を取得するインターフェース."""

    @abstractmethod
    def get_group(self, param_name: str) -> str:
        """パラメータ名から層グループ名を取得.

        Args:
            param_name: パラメータ名.

        Returns:
            str: 層グループ名.
        """
