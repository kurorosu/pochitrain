"""test_cli 共通フィクスチャ."""

from typing import Any

import torchvision.transforms as transforms


def build_cli_config(**overrides: Any) -> dict[str, Any]:
    """CLI テスト用の最小設定辞書を生成する.

    Args:
        **overrides: 上書きするキーと値.

    Returns:
        設定辞書.
    """
    config: dict[str, Any] = {
        "model_name": "resnet18",
        "num_classes": 2,
        "device": "cpu",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "train_data_root": "data/train",
        "val_data_root": "data/val",
        "num_workers": 0,
        "train_transform": transforms.Compose([transforms.ToTensor()]),
        "val_transform": transforms.Compose([transforms.ToTensor()]),
        "enable_layer_wise_lr": False,
    }
    config.update(overrides)
    return config
