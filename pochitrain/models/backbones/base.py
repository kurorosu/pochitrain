"""
pochitrain.models.backbones.base: 基本バックボーンクラス

すべてのバックボーンモデルの基底クラス
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """
    基本バックボーンクラス

    すべてのバックボーンモデルが継承すべき基底クラス
    """

    def __init__(self):
        super().__init__()
        self.feat_dim = None  # 特徴量の次元数
        self.num_stages = None  # ステージ数
        self.stage_dims = None  # 各ステージの次元数

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: 出力特徴量
        """
        pass

    def get_feat_dim(self) -> int:
        """
        特徴量の次元数を取得

        Returns:
            int: 特徴量の次元数
        """
        return self.feat_dim

    def get_num_stages(self) -> int:
        """
        ステージ数を取得

        Returns:
            int: ステージ数
        """
        return self.num_stages

    def get_stage_dims(self) -> List[int]:
        """
        各ステージの次元数を取得

        Returns:
            List[int]: 各ステージの次元数
        """
        return self.stage_dims

    def init_weights(self) -> None:
        """
        重みの初期化

        デフォルトではKaimingの初期化を使用
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_stages(self, num_stages: int) -> None:
        """
        指定したステージまでのパラメータを凍結

        Args:
            num_stages (int): 凍結するステージ数
        """
        # サブクラスで実装
        pass

    def train(self, mode: bool = True) -> 'BaseBackbone':
        """
        訓練モードの設定

        Args:
            mode (bool): 訓練モードかどうか

        Returns:
            BaseBackbone: 自身のインスタンス
        """
        super().train(mode)
        return self

    def eval(self) -> 'BaseBackbone':
        """
        評価モードの設定

        Returns:
            BaseBackbone: 自身のインスタンス
        """
        super().eval()
        return self

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデルの情報を取得

        Returns:
            Dict[str, Any]: モデル情報
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feat_dim': self.feat_dim,
            'num_stages': self.num_stages,
            'stage_dims': self.stage_dims
        }
