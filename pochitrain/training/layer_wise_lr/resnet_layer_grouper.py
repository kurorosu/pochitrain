"""pochitrain.training.layer_wise_lr.resnet_layer_grouper: ResNet 用の層グルーパー."""

from .interfaces import ILayerGrouper


class ResNetLayerGrouper(ILayerGrouper):
    """ResNet アーキテクチャに基づく層グルーパー.

    パラメータ名を ResNet の構造 (conv1, bn1, layer1-4, fc) に基づいて
    グループ名にマッピングする.
    """

    _GROUP_KEYWORDS: tuple[str, ...] = (
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "conv1",
        "bn1",
        "fc",
    )

    def get_group(self, param_name: str) -> str:
        """パラメータ名から ResNet 層グループ名を取得.

        Args:
            param_name: パラメータ名.

        Returns:
            str: 層グループ名. 該当しない場合は "other".
        """
        for keyword in self._GROUP_KEYWORDS:
            if keyword in param_name:
                return keyword
        return "other"
