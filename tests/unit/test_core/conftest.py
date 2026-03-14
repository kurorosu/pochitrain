"""test_core パッケージ共通フィクスチャ."""

import logging

import pytest

from pochitrain import PochiTrainer


@pytest.fixture
def logger() -> logging.Logger:
    """テスト用ロガー."""
    return logging.getLogger("test")


@pytest.fixture
def trainer() -> PochiTrainer:
    """テスト用のPochiTrainerインスタンスを作成."""
    return PochiTrainer(
        model_name="resnet18",
        num_classes=4,
        device="cpu",
        pretrained=False,
        create_workspace=False,
    )
