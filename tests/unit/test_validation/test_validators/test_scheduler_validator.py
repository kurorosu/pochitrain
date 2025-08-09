"""
SchedulerValidatorのユニットテスト.
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
        self.mock_logger.info.assert_called_with("スケジューラー: なし（固定学習率）")

    def test_scheduler_missing_success(self):
        """schedulerキーなしでバリデーション成功."""
        config = {}
        result = self.validator.validate(config, self.mock_logger)

        self.assertTrue(result)
        self.mock_logger.info.assert_called_with("スケジューラー: なし（固定学習率）")

    def test_unsupported_scheduler_failure(self):
        """未サポートスケジューラーでバリデーション失敗."""
        config = {"scheduler": "UnsupportedLR"}
        result = self.validator.validate(config, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_scheduler_params_missing_failure(self):
        """scheduler_params未設定でバリデーション失敗."""
        config = {"scheduler": "StepLR"}
        result = self.validator.validate(config, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_scheduler_params_none_failure(self):
        """scheduler_params=Noneでバリデーション失敗."""
        config = {"scheduler": "StepLR", "scheduler_params": None}
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
        self.mock_logger.info.assert_any_call("スケジューラー: StepLR")

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

    def test_step_lr_invalid_gamma_failure(self):
        """StepLRでgammaが範囲外の場合バリデーション失敗."""
        config = {
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": 1.5},
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
        self.mock_logger.info.assert_any_call("スケジューラー: MultiStepLR")

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

    def test_multi_step_lr_unsorted_milestones_failure(self):
        """MultiStepLRでmilestonesが昇順でない場合バリデーション失敗."""
        config = {
            "scheduler": "MultiStepLR",
            "scheduler_params": {"milestones": [60, 30, 90], "gamma": 0.1},
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

    # CosineAnnealingLRのテスト
    def test_cosine_annealing_lr_valid_success(self):
        """CosineAnnealingLRの有効なパラメータでバリデーション成功."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": 100, "eta_min": 0.001},
        }
        result = self.validator.validate(config, self.mock_logger)

        self.assertTrue(result)
        self.mock_logger.info.assert_any_call("スケジューラー: CosineAnnealingLR")

    def test_cosine_annealing_lr_missing_t_max_failure(self):
        """CosineAnnealingLRでT_max未設定でバリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"eta_min": 0.001},
        }
        result = self.validator.validate(config, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_cosine_annealing_lr_invalid_t_max_type_failure(self):
        """CosineAnnealingLRでT_maxが整数でない場合バリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": "100", "eta_min": 0.001},
        }
        result = self.validator.validate(config, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_cosine_annealing_lr_negative_t_max_failure(self):
        """CosineAnnealingLRでT_maxが負の値の場合バリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": -50, "eta_min": 0.001},
        }
        result = self.validator.validate(config, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_cosine_annealing_lr_negative_eta_min_failure(self):
        """CosineAnnealingLRでeta_minが負の値の場合バリデーション失敗."""
        config = {
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": 100, "eta_min": -0.001},
        }
        result = self.validator.validate(config, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
