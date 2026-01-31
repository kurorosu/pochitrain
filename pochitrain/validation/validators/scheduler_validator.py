"""スケジューラー設定バリデーター."""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class SchedulerValidator(BaseValidator):
    """スケジューラー設定の妥当性をチェック."""

    SUPPORTED_SCHEDULERS = {
        "StepLR": ["step_size", "gamma"],
        "MultiStepLR": ["milestones", "gamma"],
        "CosineAnnealingLR": ["T_max"],
        "ExponentialLR": ["gamma"],
        "LinearLR": ["start_factor", "end_factor", "total_iters"],
    }

    def __init__(self) -> None:
        """初期化."""
        pass

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        スケジューラー設定の妥当性をチェック.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: 検証成功時はTrue、失敗時はFalse
        """
        scheduler_name = config.get("scheduler")

        if scheduler_name is None:
            # スケジューラーが指定されていない場合はOK
            return True

        if not isinstance(scheduler_name, str):
            logger.error(
                f"scheduler は文字列型である必要があります。"
                f"実際の型: {type(scheduler_name)}"
            )
            return False

        # サポートされているスケジューラーかチェック
        if not self._validate_choice(
            scheduler_name,
            "スケジューラー",
            list(self.SUPPORTED_SCHEDULERS.keys()),
            logger,
        ):
            return False

        # scheduler_params の存在確認
        scheduler_params = config.get("scheduler_params")
        if scheduler_params is None:
            logger.error(
                f"スケジューラー '{scheduler_name}' を使用する場合、"
                f"scheduler_params は必須です。"
            )
            return False

        if not isinstance(scheduler_params, dict):
            logger.error(
                f"scheduler_params は辞書型である必要があります。"
                f"実際の型: {type(scheduler_params)}"
            )
            return False

        # 必須パラメータの存在確認
        required_params = self.SUPPORTED_SCHEDULERS[scheduler_name]
        for param in required_params:
            if param not in scheduler_params:
                logger.error(
                    f"スケジューラー '{scheduler_name}' に必須パラメータ "
                    f"'{param}' が指定されていません。"
                )
                return False

        # パラメータ値の妥当性チェック
        if not self._validate_scheduler_params(
            scheduler_name, scheduler_params, logger
        ):
            return False

        return True

    def _validate_scheduler_params(
        self,
        scheduler_name: str,
        scheduler_params: Dict[str, Any],
        logger: logging.Logger,
    ) -> bool:
        """
        スケジューラーのパラメータ値を検証.

        Args:
            scheduler_name (str): スケジューラー名
            scheduler_params (Dict[str, Any]): スケジューラーパラメータ
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: 検証成功時はTrue、失敗時はFalse
        """
        if scheduler_name == "StepLR":
            return self._validate_step_lr(scheduler_params, logger)
        elif scheduler_name == "MultiStepLR":
            return self._validate_multi_step_lr(scheduler_params, logger)
        elif scheduler_name == "CosineAnnealingLR":
            return self._validate_cosine_annealing_lr(scheduler_params, logger)
        elif scheduler_name == "ExponentialLR":
            return self._validate_exponential_lr(scheduler_params, logger)
        elif scheduler_name == "LinearLR":
            return self._validate_linear_lr(scheduler_params, logger)

        return True

    def _validate_step_lr(self, params: Dict[str, Any], logger: logging.Logger) -> bool:
        """StepLRパラメータの検証."""
        step_size = params.get("step_size")
        if not self._validate_required_type(
            step_size, "StepLR の step_size", int, logger
        ):
            return False
        if not self._validate_positive(step_size, "StepLR の step_size", logger):
            return False

        gamma = params.get("gamma")
        if not self._validate_required_type(
            gamma, "StepLR の gamma", (int, float), logger
        ):
            return False
        if not self._validate_positive(gamma, "StepLR の gamma", logger):
            return False

        return True

    def _validate_multi_step_lr(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """MultiStepLRパラメータの検証."""
        milestones = params.get("milestones")
        if not isinstance(milestones, list) or not milestones:
            logger.error("MultiStepLR の milestones はリスト型である必要があります。")
            return False
        if not all(isinstance(m, int) and m > 0 for m in milestones):
            logger.error(
                "MultiStepLR の milestones はすべて正の整数である必要があります。"
            )
            return False

        gamma = params.get("gamma")
        if not self._validate_required_type(
            gamma, "MultiStepLR の gamma", (int, float), logger
        ):
            return False
        if not self._validate_positive(gamma, "MultiStepLR の gamma", logger):
            return False

        return True

    def _validate_cosine_annealing_lr(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """CosineAnnealingLRパラメータの検証."""
        t_max = params.get("T_max")
        if not self._validate_required_type(
            t_max, "CosineAnnealingLR の T_max", int, logger
        ):
            return False
        if not self._validate_positive(t_max, "CosineAnnealingLR の T_max", logger):
            return False

        return True

    def _validate_exponential_lr(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """ExponentialLRパラメータの検証."""
        gamma = params.get("gamma")
        if not self._validate_required_type(
            gamma, "ExponentialLR の gamma", (int, float), logger
        ):
            return False
        if not self._validate_positive(gamma, "ExponentialLR の gamma", logger):
            return False

        return True

    def _validate_linear_lr(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """LinearLRパラメータの検証."""
        start_factor = params.get("start_factor")
        if not self._validate_required_type(
            start_factor, "LinearLR の start_factor", (int, float), logger
        ):
            return False
        if not self._validate_range(
            start_factor, "LinearLR の start_factor", logger, ge=0
        ):
            return False

        end_factor = params.get("end_factor")
        if not self._validate_required_type(
            end_factor, "LinearLR の end_factor", (int, float), logger
        ):
            return False
        if not self._validate_range(end_factor, "LinearLR の end_factor", logger, ge=0):
            return False

        total_iters = params.get("total_iters")
        if not self._validate_required_type(
            total_iters, "LinearLR の total_iters", int, logger
        ):
            return False
        if not self._validate_positive(total_iters, "LinearLR の total_iters", logger):
            return False

        return True
