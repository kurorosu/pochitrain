"""
pochitrain.pochi_dataset: Pochiデータセット.

カスタムデータセット用のシンプルなクラスと基本的なtransform
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PochiImageDataset(Dataset):
    """
    Pochi画像分類データセット.

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
        extensions (tuple): 許可する拡張子
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        """PochiImageDatasetを初期化."""
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions

        # データの読み込み
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.classes: List[str] = []

        self._load_data()

    def _load_data(self) -> None:
        """データの読み込み."""
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
                    self.image_paths.append(image_path)
                    self.labels.append(class_idx)

        if not self.image_paths:
            raise ValueError(f"画像ファイルが見つかりません: {self.root}")

    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """指定されたインデックスのデータを返す."""
        image_path = self.image_paths[index]
        label = self.labels[index]

        # 画像の読み込み
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self) -> List[str]:
        """クラス名のリストを取得."""
        return self.classes

    def get_class_counts(self) -> dict:
        """各クラスのサンプル数を取得."""
        counts = {}
        for cls_name in self.classes:
            counts[cls_name] = self.labels.count(self.classes.index(cls_name))
        return counts


def get_basic_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> transforms.Compose:
    """
    基本的なデータ変換を取得.

    Args:
        image_size (int): 画像サイズ
        is_training (bool): 訓練用かどうか
        mean (List[float]): 正規化の平均値
        std (List[float]): 正規化の標準偏差

    Returns:
        transforms.Compose: 変換処理
    """
    if is_training:
        # 訓練用の変換（データ拡張あり）
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # 検証用の変換（データ拡張なし）
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),  # 256 for 224
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def create_data_loaders(
    train_root: str,
    val_root: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], List[str]]:
    """
    データローダーを作成.

    Args:
        train_root (str): 訓練データのルートディレクトリ
        val_root (str, optional): 検証データのルートディレクトリ
        batch_size (int): バッチサイズ
        image_size (int): 画像サイズ
        num_workers (int): ワーカー数
        pin_memory (bool): メモリピニング

    Returns:
        Tuple[DataLoader, Optional[DataLoader], List[str]]: (訓練ローダー, 検証ローダー, クラス名)
    """
    # 訓練データセット
    train_transform = get_basic_transforms(image_size, is_training=True)
    train_dataset = PochiImageDataset(train_root, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 検証データセット
    val_loader = None
    if val_root:
        val_transform = get_basic_transforms(image_size, is_training=False)
        val_dataset = PochiImageDataset(val_root, transform=val_transform)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, train_dataset.get_classes()


def create_simple_transforms(image_size: int = 224) -> dict:
    """
    シンプルな変換セットを作成.

    Args:
        image_size (int): 画像サイズ

    Returns:
        dict: 変換セット
    """
    return {
        "train": get_basic_transforms(image_size, is_training=True),
        "val": get_basic_transforms(image_size, is_training=False),
        "simple": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        ),
    }


# データセットの情報を表示するヘルパー関数
def print_dataset_info(dataset: PochiImageDataset) -> None:
    """データセットの情報を表示."""
    print("データセット情報:")
    print(f"  総サンプル数: {len(dataset)}")
    print(f"  クラス数: {len(dataset.classes)}")
    print(f"  クラス名: {dataset.classes}")

    class_counts = dataset.get_class_counts()
    print("  各クラスのサンプル数:")
    for cls_name, count in class_counts.items():
        print(f"    {cls_name}: {count}")


def split_dataset(
    dataset: PochiImageDataset, train_ratio: float = 0.8
) -> Tuple[Dataset, Dataset]:
    """
    データセットを訓練用と検証用に分割.

    Args:
        dataset (PochiImageDataset): 分割するデータセット
        train_ratio (float): 訓練用の割合

    Returns:
        Tuple[Dataset, Dataset]: (訓練用データセット, 検証用データセット)
    """
    from torch.utils.data import random_split

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    return random_split(dataset, [train_size, val_size])
