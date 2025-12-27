"""最適化結果エクスポーター実装（SRP: 単一責任原則）."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

from pochitrain.optimization.interfaces import IResultExporter


class JsonResultExporter(IResultExporter):
    """JSON形式で最適化結果をエクスポートする."""

    def export(
        self,
        best_params: dict[str, Any],
        best_value: float,
        study: optuna.Study,
        output_path: str,
    ) -> None:
        """最適化結果をJSONファイルにエクスポートする.

        Args:
            best_params: 最適なパラメータ
            best_value: 最適な目的関数値
            study: Optuna Studyオブジェクト
            output_path: 出力先パス
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ベストパラメータをJSON保存
        best_params_file = output_dir / "best_params.json"
        result = {
            "study_name": study.study_name,
            "best_value": best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat(),
        }

        with open(best_params_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 試行履歴をJSON保存
        trials_file = output_dir / "trials_history.json"
        trials_data = []
        for trial in study.trials:
            trials_data.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                    "datetime_start": (
                        trial.datetime_start.isoformat()
                        if trial.datetime_start
                        else None
                    ),
                    "datetime_complete": (
                        trial.datetime_complete.isoformat()
                        if trial.datetime_complete
                        else None
                    ),
                }
            )

        with open(trials_file, "w", encoding="utf-8") as f:
            json.dump(trials_data, f, indent=2, ensure_ascii=False)


class ConfigExporter(IResultExporter):
    """Python設定ファイル形式で最適化結果をエクスポートする.

    最適パラメータをpochi_train_config.py形式で出力し、
    そのまま本格訓練に使用可能。
    """

    def __init__(self, base_config: dict[str, Any]) -> None:
        """初期化.

        Args:
            base_config: ベース設定（最適化対象外のパラメータを含む）
        """
        self._base_config = base_config

    def export(
        self,
        best_params: dict[str, Any],
        best_value: float,
        study: optuna.Study,
        output_path: str,
    ) -> None:
        """最適化結果をPython設定ファイルにエクスポートする.

        Args:
            best_params: 最適なパラメータ
            best_value: 最適な目的関数値
            study: Optuna Studyオブジェクト
            output_path: 出力先パス
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_file = output_dir / "optimized_config.py"

        # ベース設定と最適パラメータをマージ
        merged_config = {**self._base_config, **best_params}

        # Python設定ファイルを生成
        lines = [
            '"""Optuna最適化済み設定ファイル.',
            "",
            f"Study: {study.study_name}",
            f"Best Value: {best_value:.4f}",
            f"Generated: {datetime.now().isoformat()}",
            '"""',
            "",
            "from torchvision import transforms",
            "",
            "# === 最適化されたパラメータ ===",
        ]

        # 最適化されたパラメータを出力
        for key, value in best_params.items():
            lines.append(f"{key} = {repr(value)}")

        lines.append("")
        lines.append("# === ベース設定（最適化対象外） ===")

        # ベース設定のうち、最適化対象外のものを出力
        for key, value in self._base_config.items():
            if key not in best_params:
                # callableやtransformは文字列として出力しない
                if callable(value) or "transform" in key.lower():
                    continue
                lines.append(f"{key} = {repr(value)}")

        with open(config_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
