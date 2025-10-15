"""
層別学習率バリデーターのテスト.

LayerWiseLRValidatorの動作をテストします。
"""

import logging
from unittest.mock import Mock

import pytest

from pochitrain.validation.validators.layer_wise_lr_validator import (
    LayerWiseLRValidator,
)


class TestLayerWiseLRValidator:
    """LayerWiseLRValidatorのテストクラス."""

    @pytest.fixture
    def validator(self):
        """テスト用のバリデーターインスタンス."""
        return LayerWiseLRValidator()

    @pytest.fixture
    def mock_logger(self):
        """テスト用のモックロガー."""
        return Mock(spec=logging.Logger)

    def test_layer_wise_lr_disabled(self, validator, mock_logger):
        """層別学習率が無効の場合のテスト."""
        config = {"enable_layer_wise_lr": False}

        result = validator.validate(config, mock_logger)
        assert result is True
        mock_logger.debug.assert_called_once()

    def test_layer_wise_lr_enabled_valid_config(self, validator, mock_logger):
        """層別学習率が有効で正しい設定の場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {
                "layer_rates": {
                    "conv1": 0.0001,
                    "bn1": 0.0001,
                    "layer1": 0.0002,
                    "layer2": 0.0005,
                    "layer3": 0.001,
                    "layer4": 0.002,
                    "fc": 0.01,
                }
            },
        }

        result = validator.validate(config, mock_logger)
        assert result is True
        mock_logger.info.assert_called()

    def test_missing_enable_layer_wise_lr(self, validator, mock_logger):
        """enable_layer_wise_lrが存在しない場合のテスト."""
        config = {}

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "設定に 'enable_layer_wise_lr' が見つかりません"
        )

    def test_invalid_enable_layer_wise_lr_type(self, validator, mock_logger):
        """enable_layer_wise_lrの型が不正な場合のテスト."""
        config = {"enable_layer_wise_lr": "true"}  # 文字列（不正）

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_missing_layer_wise_lr_config(self, validator, mock_logger):
        """layer_wise_lr_configが存在しない場合のテスト."""
        config = {"enable_layer_wise_lr": True}

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "enable_layer_wise_lr=True の場合、'layer_wise_lr_config' が必要です"
        )

    def test_invalid_layer_wise_lr_config_type(self, validator, mock_logger):
        """layer_wise_lr_configの型が不正な場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": "invalid",  # 文字列（不正）
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_missing_layer_rates(self, validator, mock_logger):
        """layer_ratesが存在しない場合のテスト."""
        config = {"enable_layer_wise_lr": True, "layer_wise_lr_config": {}}

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "'layer_wise_lr_config' に 'layer_rates' が見つかりません"
        )

    def test_invalid_layer_rates_type(self, validator, mock_logger):
        """layer_ratesの型が不正な場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {"layer_rates": []},  # リスト（不正）
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_empty_layer_rates(self, validator, mock_logger):
        """layer_ratesが空の場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {"layer_rates": {}},
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called_with(
            "'layer_rates' が空です。少なくとも1つの層の学習率を指定してください"
        )

    def test_invalid_layer_name_type(self, validator, mock_logger):
        """層名の型が不正な場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {"layer_rates": {123: 0.001}},  # 数値（不正）
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_invalid_learning_rate_type(self, validator, mock_logger):
        """学習率の型が不正な場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {
                "layer_rates": {"conv1": "0.001"}  # 文字列（不正）
            },
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_negative_learning_rate(self, validator, mock_logger):
        """学習率が負の値の場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {
                "layer_rates": {"conv1": -0.001}  # 負の値（不正）
            },
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_zero_learning_rate(self, validator, mock_logger):
        """学習率が0の場合のテスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {"layer_rates": {"conv1": 0.0}},  # 0（不正）
        }

        result = validator.validate(config, mock_logger)
        assert result is False
        mock_logger.error.assert_called()

    def test_missing_recommended_layers_warning(self, validator, mock_logger):
        """推奨される層が不足している場合の警告テスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {"layer_rates": {"fc": 0.01}},  # 一部の層のみ
        }

        result = validator.validate(config, mock_logger)
        assert result is True  # 警告だが成功
        mock_logger.warning.assert_called()

    def test_unknown_layers_warning(self, validator, mock_logger):
        """未知の層名がある場合の警告テスト."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {
                "layer_rates": {"conv1": 0.0001, "unknown_layer": 0.001}  # 未知の層
            },
        }

        result = validator.validate(config, mock_logger)
        assert result is True  # 警告だが成功
        mock_logger.warning.assert_called()

    def test_integer_learning_rate(self, validator, mock_logger):
        """学習率が整数の場合のテスト（有効）."""
        config = {
            "enable_layer_wise_lr": True,
            "layer_wise_lr_config": {"layer_rates": {"conv1": 1}},  # 整数（有効）
        }

        result = validator.validate(config, mock_logger)
        assert result is True
