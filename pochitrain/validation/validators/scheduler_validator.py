"""スケジューラー設定のバリデーションを行うモジュール."""

import logging
from typing import Any, Dict

from ..base_validator import BaseValidator


class SchedulerValidator(BaseValidator):
    """
    スケジューラー設定のバリデーションを行うクラス.

    サポートするスケジューラー（全パラメータ明示的設定必須）:
    - 'StepLR': step_size (int), gamma (float) が必須
    - 'MultiStepLR': milestones (List[int]), gamma (float) が必須
    - 'CosineAnnealingLR': T_max (int), eta_min (float) が必須
    - None: スケジューラーなし（固定学習率）
    """

    def __init__(self) -> None:
        """SchedulerValidatorを初期化."""
        # サポートするスケジューラーの必須パラメータ定義（全パラメータ明示的設定必須）
        self.supported_schedulers = {
            "StepLR": ["step_size", "gamma"],
            "MultiStepLR": ["milestones", "gamma"],
            "CosineAnnealingLR": ["T_max", "eta_min"],
        }

    def validate(self, config: Dict[str, Any], logger: logging.Logger) -> bool:
        """
        スケジューラー設定をバリデーション.

        Args:
            config (Dict[str, Any]): 設定辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: バリデーション成功時True、失敗時False
        """
        scheduler = config.get("scheduler")
        scheduler_params = config.get("scheduler_params")

        # scheduler設定の確認
        if scheduler is None:
            logger.info("スケジューラー: なし（固定学習率）")
            return True

        # 未サポートスケジューラーのチェック
        if scheduler not in self.supported_schedulers:
            supported_list = list(self.supported_schedulers.keys()) + ["None"]
            logger.error(
                f"未サポートのスケジューラーです: {scheduler}. "
                f"サポート対象: {supported_list}"
            )
            return False

        # scheduler_paramsの存在確認
        if scheduler_params is None:
            logger.error(
                f"スケジューラー '{scheduler}' を使用する場合、"
                f"scheduler_paramsが必須です"
            )
            return False

        if not isinstance(scheduler_params, dict):
            logger.error(
                f"scheduler_paramsは辞書型である必要があります。"
                f"現在の型: {type(scheduler_params)}"
            )
            return False

        # 必須パラメータの確認
        required_params = self.supported_schedulers[scheduler]
        missing_params = []

        for param in required_params:
            if param not in scheduler_params:
                missing_params.append(param)

        if missing_params:
            logger.error(
                f"スケジューラー '{scheduler}' に必須パラメータが不足しています: "
                f"{missing_params}"
            )
            return False

        # パラメータ値の型・範囲検証
        if not self._validate_scheduler_params(scheduler, scheduler_params, logger):
            return False

        # 成功時のログ出力
        logger.info(f"スケジューラー: {scheduler}")
        logger.info(f"スケジューラーパラメータ: {scheduler_params}")

        return True

    def _validate_scheduler_params(
        self, scheduler: str, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """
        スケジューラー別のパラメータ詳細検証.

        Args:
            scheduler (str): スケジューラー名
            params (Dict[str, Any]): パラメータ辞書
            logger (logging.Logger): ロガーインスタンス

        Returns:
            bool: 検証成功時True、失敗時False
        """
        if scheduler == "StepLR":
            return self._validate_step_lr_params(params, logger)
        elif scheduler == "MultiStepLR":
            return self._validate_multi_step_lr_params(params, logger)
        elif scheduler == "CosineAnnealingLR":
            return self._validate_cosine_annealing_lr_params(params, logger)
        else:
            return True

    def _validate_step_lr_params(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """StepLRのパラメータ検証."""
        step_size = params.get("step_size")

        if not isinstance(step_size, int):
            logger.error(
                f"StepLRのstep_sizeは整数である必要があります。"
                f"現在の値: {step_size} (型: {type(step_size)})"
            )
            return False

        if step_size <= 0:
            logger.error(
                f"StepLRのstep_sizeは正の整数である必要があります。"
                f"現在の値: {step_size}"
            )
            return False

        # gammaパラメータの検証（必須）
        gamma = params.get("gamma")
        if gamma is None:
            logger.error(
                "StepLRのgammaパラメータが必須です。"
                "configs/pochi_config.pyで設定してください。"
            )
            return False

        if not isinstance(gamma, (int, float)):
            logger.error(
                f"StepLRのgammaは数値である必要があります。"
                f"現在の値: {gamma} (型: {type(gamma)})"
            )
            return False

        if gamma <= 0 or gamma >= 1:
            logger.error(
                f"StepLRのgammaは0〜1の範囲である必要があります。" f"現在の値: {gamma}"
            )
            return False

        return True

    def _validate_multi_step_lr_params(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """MultiStepLRのパラメータ検証."""
        milestones = params.get("milestones")

        if not isinstance(milestones, list):
            logger.error(
                f"MultiStepLRのmilestonesはリストである必要があります。"
                f"現在の型: {type(milestones)}"
            )
            return False

        if not milestones:
            logger.error("MultiStepLRのmilestonesは空リストにできません")
            return False

        # 各要素の型確認
        for i, milestone in enumerate(milestones):
            if not isinstance(milestone, int):
                logger.error(
                    f"MultiStepLRのmilestones[{i}]は整数である必要があります。"
                    f"現在の値: {milestone} (型: {type(milestone)})"
                )
                return False

            if milestone <= 0:
                logger.error(
                    f"MultiStepLRのmilestones[{i}]は正の整数である必要があります。"
                    f"現在の値: {milestone}"
                )
                return False

        # 昇順確認
        sorted_milestones = sorted(milestones)
        if milestones != sorted_milestones:
            logger.error(
                f"MultiStepLRのmilestonesは昇順である必要があります。"
                f"現在: {milestones}, 期待: {sorted_milestones}"
            )
            return False

        # gammaパラメータの検証（必須）
        gamma = params.get("gamma")
        if gamma is None:
            logger.error(
                "MultiStepLRのgammaパラメータが必須です。"
                "configs/pochi_config.pyで設定してください。"
            )
            return False

        if not isinstance(gamma, (int, float)):
            logger.error(
                f"MultiStepLRのgammaは数値である必要があります。"
                f"現在の値: {gamma} (型: {type(gamma)})"
            )
            return False

        if gamma <= 0 or gamma >= 1:
            logger.error(
                f"MultiStepLRのgammaは0〜1の範囲である必要があります。"
                f"現在の値: {gamma}"
            )
            return False

        return True

    def _validate_cosine_annealing_lr_params(
        self, params: Dict[str, Any], logger: logging.Logger
    ) -> bool:
        """CosineAnnealingLRのパラメータ検証."""
        T_max = params.get("T_max")

        if not isinstance(T_max, int):
            logger.error(
                f"CosineAnnealingLRのT_maxは整数である必要があります。"
                f"現在の値: {T_max} (型: {type(T_max)})"
            )
            return False

        if T_max <= 0:
            logger.error(
                f"CosineAnnealingLRのT_maxは正の整数である必要があります。"
                f"現在の値: {T_max}"
            )
            return False

        # eta_minパラメータの検証（必須）
        eta_min = params.get("eta_min")
        if eta_min is None:
            logger.error(
                "CosineAnnealingLRのeta_minパラメータが必須です。"
                "configs/pochi_config.pyで設定してください。"
            )
            return False

        if not isinstance(eta_min, (int, float)):
            logger.error(
                f"CosineAnnealingLRのeta_minは数値である必要があります。"
                f"現在の値: {eta_min} (型: {type(eta_min)})"
            )
            return False

        if eta_min < 0:
            logger.error(
                f"CosineAnnealingLRのeta_minは非負の値である必要があります。"
                f"現在の値: {eta_min}"
            )
            return False

        return True
