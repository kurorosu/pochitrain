"""
pochitrain.data.datasets: データセット実装

基本的なデータセットクラスの実装
"""

import os
from typing import List, Tuple, Optional, Any, Callable
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as torchvision_datasets

from ..core.registry import DATASETS


class BaseDataset(Dataset):
    """
    基本データセットクラス

    Args:
        root (str): データのルートディレクトリ
        transform (callable, optional): データ変換関数
        target_transform (callable, optional): ターゲット変換関数
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        # データの読み込み
        self.data = []
        self.targets = []
        self.classes = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

        return data, target

    def get_classes(self) -> List[str]:
        """クラス名のリストを取得"""
        return self.classes


@DATASETS.register_module()
class ImageClassificationDataset(BaseDataset):
    """
    画像分類データセット

    フォルダ構造:
    root/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg

    Args:
        root (str): データのルートディレクトリ
        transform (callable, optional): データ変換関数
        target_transform (callable, optional): ターゲット変換関数
        extensions (tuple, optional): 許可する拡張子
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        super().__init__(root, transform, target_transform)
        self.extensions = extensions

        # データの読み込み
        self._load_data()

    def _load_data(self) -> None:
        """データの読み込み"""
        if not self.root.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {self.root}")

        # クラスフォルダの取得
        class_folders = [d for d in self.root.iterdir() if d.is_dir()]
        class_folders.sort()

        if not class_folders:
            raise ValueError(f"クラスフォルダが見つかりません: {self.root}")

        self.classes = [folder.name for folder in class_folders]
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 画像ファイルの読み込み
        for class_folder in class_folders:
            class_idx = class_to_idx[class_folder.name]

            for image_path in class_folder.iterdir():
                if image_path.suffix.lower() in self.extensions:
                    self.data.append(image_path)
                    self.targets.append(class_idx)

        if not self.data:
            raise ValueError(f"画像ファイルが見つかりません: {self.root}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path, target = self.data[index], self.targets[index]

        # 画像の読み込み
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target


@DATASETS.register_module()
class CIFAR10(torchvision_datasets.CIFAR10):
    """
    CIFAR-10データセット

    Args:
        root (str): データのルートディレクトリ
        train (bool): 訓練データかどうか
        transform (callable, optional): データ変換関数
        target_transform (callable, optional): ターゲット変換関数
        download (bool): データをダウンロードするかどうか
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True):
        super().__init__(root, train, transform, target_transform, download)

        # クラス名の設定
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    def get_classes(self) -> List[str]:
        """クラス名のリストを取得"""
        return self.classes


@DATASETS.register_module()
class CIFAR100(torchvision_datasets.CIFAR100):
    """
    CIFAR-100データセット

    Args:
        root (str): データのルートディレクトリ
        train (bool): 訓練データかどうか
        transform (callable, optional): データ変換関数
        target_transform (callable, optional): ターゲット変換関数
        download (bool): データをダウンロードするかどうか
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True):
        super().__init__(root, train, transform, target_transform, download)

    def get_classes(self) -> List[str]:
        """クラス名のリスト（細分類）を取得"""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]


@DATASETS.register_module()
class ImageNet(torchvision_datasets.ImageNet):
    """
    ImageNetデータセット

    Args:
        root (str): データのルートディレクトリ
        split (str): データセットの分割 ('train', 'val')
        transform (callable, optional): データ変換関数
        target_transform (callable, optional): ターゲット変換関数
    """

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, split, transform, target_transform)

    def get_classes(self) -> List[str]:
        """クラス名のリストを取得"""
        return self.classes


def build_dataset(config: dict) -> Dataset:
    """
    設定からデータセットを構築

    Args:
        config (dict): データセット設定

    Returns:
        Dataset: 構築されたデータセット
    """
    return DATASETS.build(config)
