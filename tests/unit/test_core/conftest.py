"""test_core パッケージ共通フィクスチャ."""

import logging

import pytest


@pytest.fixture
def logger() -> logging.Logger:
    """テスト用ロガー."""
    return logging.getLogger("test")
