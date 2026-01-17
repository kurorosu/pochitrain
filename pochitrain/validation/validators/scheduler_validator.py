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
        if scheduler_name not in self.SUPPORTED_SCHEDULERS:
            logger.error(
                f"サポートされていないスケジューラー: {scheduler_name}. "
                f"サポートされているスケジューラー: "
                f"{list(self.SUPPORTED_SCHEDULERS.keys())}"
            )
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
            step_size = scheduler_params.get("step_size")
            if not isinstance(step_size, int):
                logger.error("StepLR の step_size は整数型である必要があります。")
                return False
            if step_size is not None and step_size <= 0:
                logger.error("StepLR の step_size は正数である必要があります。")
                return False

            gamma = scheduler_params.get("gamma")
            if not isinstance(gamma, (int, float)):
                logger.error("StepLR の gamma は数値型である必要があります。")
                return False
            if gamma is not None and gamma <= 0:
                logger.error("StepLR の gamma は正数である必要があります。")
                return False

        elif scheduler_name == "MultiStepLR":
            milestones = scheduler_params.get("milestones")
            if not isinstance(milestones, list) or not milestones:
                logger.error(
                    "MultiStepLR の milestones はリスト型である必要があります。"
                )
                return False
            if not all(isinstance(m, int) and m > 0 for m in milestones):
                logger.error(
                    "MultiStepLR の milestones はすべて正の整数である必要があります。"
                )
                return False

            gamma = scheduler_params.get("gamma")
            if not isinstance(gamma, (int, float)):
                logger.error("MultiStepLR の gamma は数値型である必要があります。")
                return False
            if gamma is not None and gamma <= 0:
                logger.error("MultiStepLR の gamma は正数である必要があります。")
                return False

        elif scheduler_name == "CosineAnnealingLR":
            t_max = scheduler_params.get("T_max")
            if not isinstance(t_max, int):
                logger.error(
                    "CosineAnnealingLR の T_max は整数型である必要があります。"
                )
                return False
            if t_max is not None and t_max <= 0:
                logger.error("CosineAnnealingLR の T_max は正数である必要があります。")
                return False

        elif scheduler_name == "ExponentialLR":
            gamma = scheduler_params.get("gamma")
            if not isinstance(gamma, (int, float)):
                logger.error("ExponentialLR の gamma は数値型である必要があります。")
                return False
            if gamma is not None and gamma <= 0:
                logger.error("ExponentialLR の gamma は正数である必要があります。")
                return False

        elif scheduler_name == "LinearLR":
            start_factor = scheduler_params.get("start_factor")
            end_factor = scheduler_params.get("end_factor")
            total_iters = scheduler_params.get("total_iters")

            if not isinstance(start_factor, (int, float)):
                logger.error("LinearLR の start_factor は数値型である必要があります。")
                return False
            if start_factor is not None and start_factor < 0:
                logger.error("LinearLR の start_factor は非負数である必要があります。")
                return False

            if not isinstance(end_factor, (int, float)):
                logger.error("LinearLR の end_factor は数値型である必要があります。")
                return False
            if end_factor is not None and end_factor < 0:
                logger.error("LinearLR の end_factor は非負数である必要があります。")
                return False

            if not isinstance(total_iters, int):
                logger.error("LinearLR の total_iters は整数型である必要があります。")
                return False
            if total_iters is not None and total_iters <= 0:
                logger.error("LinearLR の total_iters は正数である必要があります。")
                return False

        return True
