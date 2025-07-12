"""
pochitrain.data.transforms: データ変換処理

画像データの前処理・データ拡張を行うモジュール
"""

from typing import List, Dict, Any, Optional, Callable
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import torch

from ..core.registry import TRANSFORMS


# 基本的な変換処理をレジストリに登録
@TRANSFORMS.register_module()
class ToTensor:
    """PIL ImageをTensorに変換"""

    def __init__(self):
        self.transform = T.ToTensor()

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class Normalize:
    """正規化"""

    def __init__(self, mean: List[float], std: List[float]):
        self.transform = T.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        return self.transform(tensor)


@TRANSFORMS.register_module()
class Resize:
    """サイズ変更"""

    def __init__(self, size: int):
        self.transform = T.Resize(size)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class CenterCrop:
    """中央クロップ"""

    def __init__(self, size: int):
        self.transform = T.CenterCrop(size)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class RandomCrop:
    """ランダムクロップ"""

    def __init__(self, size: int, padding: int = 0):
        self.transform = T.RandomCrop(size, padding=padding)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class RandomHorizontalFlip:
    """ランダム水平反転"""

    def __init__(self, p: float = 0.5):
        self.transform = T.RandomHorizontalFlip(p=p)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class RandomVerticalFlip:
    """ランダム垂直反転"""

    def __init__(self, p: float = 0.5):
        self.transform = T.RandomVerticalFlip(p=p)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class RandomRotation:
    """ランダム回転"""

    def __init__(self, degrees: int):
        self.transform = T.RandomRotation(degrees)

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class ColorJitter:
    """色彩変更"""

    def __init__(self, brightness: float = 0, contrast: float = 0,
                 saturation: float = 0, hue: float = 0):
        self.transform = T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue
        )

    def __call__(self, img):
        return self.transform(img)


@TRANSFORMS.register_module()
class GaussianBlur:
    """ガウシアンブラー"""

    def __init__(self, kernel_size: int, sigma: float = 1.0):
        self.transform = T.GaussianBlur(kernel_size, sigma)

    def __call__(self, img):
        return self.transform(img)


def build_transforms(transform_configs: List[Dict[str, Any]]) -> T.Compose:
    """
    変換リストの構築

    Args:
        transform_configs (List[Dict[str, Any]]): 変換設定のリスト

    Returns:
        T.Compose: 変換処理のコンポーズ

    Examples:
        >>> transforms = [
        ...     {'type': 'Resize', 'size': 224},
        ...     {'type': 'ToTensor'},
        ...     {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        ... ]
        >>> transform = build_transforms(transforms)
    """
    transform_list = []

    for config in transform_configs:
        if isinstance(config, str):
            # 文字列の場合は設定なしで追加
            transform_list.append(TRANSFORMS.build(config))
        elif isinstance(config, dict):
            # 辞書の場合は設定を渡して構築
            transform_list.append(TRANSFORMS.build(config))
        else:
            raise ValueError(f"サポートされていない変換設定: {config}")

    return T.Compose(transform_list)


def get_preset_transforms(preset: str, image_size: int = 224) -> T.Compose:
    """
    プリセット変換の取得

    Args:
        preset (str): プリセット名
        image_size (int): 画像サイズ

    Returns:
        T.Compose: 変換処理

    Available presets:
        - 'imagenet_train': ImageNet訓練用の標準変換
        - 'imagenet_val': ImageNet検証用の標準変換
        - 'cifar_train': CIFAR訓練用の標準変換
        - 'cifar_val': CIFAR検証用の標準変換
        - 'simple': 最小限の変換
    """
    if preset == 'imagenet_train':
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif preset == 'imagenet_val':
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif preset == 'cifar_train':
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    elif preset == 'cifar_val':
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    elif preset == 'simple':
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    else:
        raise ValueError(f"サポートされていないプリセット: {preset}")


# 便利な変換関数を定義
TRANSFORM_REGISTRY = {
    'ToTensor': ToTensor,
    'Normalize': Normalize,
    'Resize': Resize,
    'CenterCrop': CenterCrop,
    'RandomCrop': RandomCrop,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomRotation': RandomRotation,
    'ColorJitter': ColorJitter,
    'GaussianBlur': GaussianBlur,
}
