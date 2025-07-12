"""
pochitrain.data.loaders: データローダー

データローダーの構築と管理を行うモジュール
"""

from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np


def build_dataloader(dataset: Dataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = None,
                     drop_last: bool = False,
                     **kwargs) -> DataLoader:
    """
    データローダーの構築

    Args:
        dataset (Dataset): データセット
        batch_size (int): バッチサイズ
        shuffle (bool): データをシャッフルするかどうか
        num_workers (int): ワーカープロセス数
        pin_memory (bool): メモリピン留めを使用するかどうか
        drop_last (bool): 最後の不完全なバッチを削除するかどうか
        **kwargs: その他のDataLoaderパラメータ

    Returns:
        DataLoader: 構築されたデータローダー
    """
    # pin_memoryのデフォルト値を設定
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )


def split_dataset(dataset: Dataset,
                  split_ratio: float = 0.8,
                  random_seed: Optional[int] = None) -> tuple:
    """
    データセットを訓練用と検証用に分割

    Args:
        dataset (Dataset): 分割するデータセット
        split_ratio (float): 訓練データの割合
        random_seed (int, optional): ランダムシード

    Returns:
        tuple: (訓練データセット, 検証データセット)
    """
    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
    else:
        generator = None

    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset


def create_dataloaders(train_dataset: Dataset,
                       val_dataset: Optional[Dataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = None) -> Dict[str, DataLoader]:
    """
    訓練用・検証用データローダーの作成

    Args:
        train_dataset (Dataset): 訓練データセット
        val_dataset (Dataset, optional): 検証データセット
        batch_size (int): バッチサイズ
        num_workers (int): ワーカープロセス数
        pin_memory (bool): メモリピン留めを使用するかどうか

    Returns:
        Dict[str, DataLoader]: データローダーの辞書
    """
    # pin_memoryのデフォルト値を設定
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataloaders = {}

    # 訓練データローダー
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    # 検証データローダー
    if val_dataset is not None:
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

    return dataloaders


class DataLoaderManager:
    """
    データローダーの管理クラス

    Args:
        config (Dict[str, Any]): データローダー設定
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataloaders = {}

    def build_train_dataloader(self, dataset: Dataset) -> DataLoader:
        """訓練用データローダーの構築"""
        return build_dataloader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', torch.cuda.is_available()),
            drop_last=True
        )

    def build_val_dataloader(self, dataset: Dataset) -> DataLoader:
        """検証用データローダーの構築"""
        return build_dataloader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', torch.cuda.is_available()),
            drop_last=False
        )

    def build_test_dataloader(self, dataset: Dataset) -> DataLoader:
        """テスト用データローダーの構築"""
        return build_dataloader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', torch.cuda.is_available()),
            drop_last=False
        )

    def get_dataloader_info(self, dataloader: DataLoader) -> Dict[str, Any]:
        """データローダーの情報を取得"""
        return {
            'batch_size': dataloader.batch_size,
            'dataset_size': len(dataloader.dataset),
            'num_batches': len(dataloader),
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'drop_last': dataloader.drop_last
        }
