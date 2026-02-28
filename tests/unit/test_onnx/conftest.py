"""test_onnx パッケージ共通フィクスチャ."""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """テスト用のシンプルなモデル."""

    def __init__(self, num_classes: int = 10) -> None:
        """モデルを初期化する.

        Args:
            num_classes: 出力クラス数.
        """
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を行う.

        Args:
            x: 入力テンソル.

        Returns:
            ログイット.
        """
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
