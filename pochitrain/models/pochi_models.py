"""
pochitrain.models.pochi_models: Pochiモデル実装.

torchvisionのモデルを直接使用するシンプルなラッパー.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PochiModel(nn.Module):
    """
    torchvisionモデルのPochiラッパー.

    Args:
        model_name (str): モデル名 ('resnet18', 'resnet34', 'resnet50')
        num_classes (int): 分類クラス数
        pretrained (bool): 事前学習済みモデルを使用するか
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        """PochiModelを初期化."""
        super().__init__()

        supported_models = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }

        if model_name not in supported_models:
            raise ValueError(
                f"サポートされていないモデル: {model_name}. "
                f"サポートされているモデル: {list(supported_models.keys())}"
            )

        weights = "DEFAULT" if pretrained else None
        self.model = supported_models[model_name](weights=weights)

        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "classifier"):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

        self.model_name = model_name
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        output: torch.Tensor = self.model(x)
        return output

    def get_model_info(self) -> dict:
        """モデル情報を取得."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


def create_model(
    model_name: str, num_classes: int, pretrained: bool = True
) -> PochiModel:
    """
    モデルを作成する便利関数.

    Args:
        model_name (str): モデル名
        num_classes (int): 分類クラス数
        pretrained (bool): 事前学習済みモデルを使用するか

    Returns:
        PochiModel: 作成されたモデル
    """
    return PochiModel(model_name, num_classes, pretrained)
