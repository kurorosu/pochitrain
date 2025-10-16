"""
SchedulerValidatorのユニットテスト.

新しいSchedulerValidator実装に対応したテスト.
"""

import logging
import unittest
from unittest.mock import Mock

from pochitrain.validation.validators.scheduler_validator import SchedulerValidator


class TestSchedulerValidator(unittest.TestCase):
    """SchedulerValidatorのテストクラス."""

    def setUp(self):
        """テストセットアップ."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.validator = SchedulerValidator()

    def test_scheduler_none_success(self):
        """scheduler=Noneでバリデーション成功."""
        config = {"scheduler": None}
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_scheduler_missing_success(self):
        """schedulerキーなしでバリデーション成功."""
        config = {}
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_unsupported_scheduler_failure(self):
        """未サポートスケジューラーでバリデーション失敗."""
        config = {"scheduler": "UnsupportedLR", "scheduler_params": {}}
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_scheduler_params_missing_failure(self):
        """scheduler_params未設定でバリデーション失敗."""
        config = {"scheduler": "StepLR"}
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_scheduler_params_invalid_type_failure(self):
        """scheduler_paramsが辞書型でない場合バリデーション失敗."""
        config = {"scheduler": "StepLR", "scheduler_params": "invalid"}
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    # StepLRのテスト
    def test_step_lr_valid_success(self):
        """StepLRの有効なパラメータでバリデーション成功."""
        config = {
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_step_lr_missing_step_size_failure(self):
        """StepLRでstep_size未設定でバリデーション失敗."""
        config = {"scheduler": "StepLR", "scheduler_params": {"gamma": 0.1}}
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_step_lr_invalid_step_size_type_failure(self):
        """StepLRでstep_sizeが整数でない場合バリデーション失敗."""
        config = {
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": "30", "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_step_lr_negative_step_size_failure(self):
        """StepLRでstep_sizeが負の値の場合バリデーション失敗."""
        config = {
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": -5, "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_step_lr_invalid_gamma_type_failure(self):
        """StepLRでgammaが数値でない場合バリデーション失敗."""
        config = {
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": "0.1"},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_step_lr_zero_gamma_failure(self):
        """StepLRでgammaが0の場合バリデーション失敗."""
        config = {
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": 0},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    # MultiStepLRのテスト
    def test_multi_step_lr_valid_success(self):
        """MultiStepLRの有効なパラメータでバリデーション成功."""
        config = {
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": [30, 60, 90], "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_multi_step_lr_missing_milestones_failure(self):
        """MultiStepLRでmilestones未設定でバリデーション失敗."""
        config = {"scheduler": "MultiStepLR", "scheduler_params": {"gamma": 0.1}}
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_multi_step_lr_empty_milestones_failure(self):
        """MultiStepLRでmilestonesが空リストの場合バリデーション失敗."""
        config = {
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": [], "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_multi_step_lr_invalid_milestones_type_failure(self):
        """MultiStepLRでmilestonesがリストでない場合バリデーション失敗."""
        config = {
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": "30,60,90", "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_multi_step_lr_negative_milestone_failure(self):
        """MultiStepLRでmilestonesに負の値が含まれる場合バリデーション失敗."""
        config = {
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": [-10, 30, 60], "gamma": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_multi_step_lr_zero_gamma_failure(self):
        """MultiStepLRでgammaが0の場合バリデーション失敗."""
        config = {
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": [30, 60], "gamma": 0},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    # CosineAnnealingLRのテスト
    def test_cosine_annealing_lr_valid_success(self):
        """CosineAnnealingLRの有効なパラメータでバリデーション成功."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": 100},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_cosine_annealing_lr_missing_t_max_failure(self):
        """CosineAnnealingLRでT_max未設定でバリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_cosine_annealing_lr_invalid_t_max_type_failure(self):
        """CosineAnnealingLRでT_maxが整数でない場合バリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": "100"},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_cosine_annealing_lr_negative_t_max_failure(self):
        """CosineAnnealingLRでT_maxが負の値の場合バリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": -50},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    # ExponentialLRのテスト
    def test_exponential_lr_valid_success(self):
        """ExponentialLRの有効なパラメータでバリデーション成功."""
        config = {
            "scheduler": "ExponentialLR",
            "scheduler_params": {"gamma": 0.95},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_exponential_lr_missing_gamma_failure(self):
        """ExponentialLRでgamma未設定でバリデーション失敗."""
        config = {"scheduler": "ExponentialLR", "scheduler_params": {}}
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_exponential_lr_invalid_gamma_type_failure(self):
        """ExponentialLRでgammaが数値でない場合バリデーション失敗."""
        config = {
            "scheduler": "ExponentialLR",
            "scheduler_params": {"gamma": "0.95"},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_exponential_lr_zero_gamma_failure(self):
        """ExponentialLRでgammaが0の場合バリデーション失敗."""
        config = {
            "scheduler": "ExponentialLR",
            "scheduler_params": {"gamma": 0},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    # LinearLRのテスト
    def test_linear_lr_valid_success(self):
        """LinearLRの有効なパラメータでバリデーション成功."""
        config = {
            "scheduler": "LinearLR",
            "scheduler_params": {
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": 50,
            },
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertTrue(result)

    def test_linear_lr_missing_total_iters_failure(self):
        """LinearLRでtotal_iters未設定でバリデーション失敗."""
        config = {
            "scheduler": "LinearLR",
            "scheduler_params": {"start_factor": 1.0, "end_factor": 0.1},
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_linear_lr_invalid_total_iters_type_failure(self):
        """LinearLRでtotal_itersが整数でない場合バリデーション失敗."""
        config = {
            "scheduler": "LinearLR",
            "scheduler_params": {
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": "50",
            },
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_linear_lr_zero_total_iters_failure(self):
        """LinearLRでtotal_itersが0の場合バリデーション失敗."""
        config = {
            "scheduler": "LinearLR",
            "scheduler_params": {
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": 0,
            },
        }
        result = self.validator.validate(config, self.mock_logger)
        self.assertFalse(result)
        self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
