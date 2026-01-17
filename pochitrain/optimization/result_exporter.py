"""最適化結果エクスポーター実装（SRP: 単一責任原則）."""

import json
import pprint
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

    最適パラメータをpochi_train_config.py形式で出力し,
    そのまま本格訓練に使用可能.
    """

    def __init__(self, base_config: dict[str, Any]) -> None:
        """初期化.

        Args:
            base_config: ベース設定(最適化対象外のパラメータを含む)
        """
        self._base_config = base_config

    def _format_dict(self, d: dict[str, Any]) -> str:
        """辞書を整形された文字列に変換.

        pprint.pformat()を使用してPython形式で整形.
        開き括弧後の空白は独特だが, Python構文として正しく動作する.

        Args:
            d: フォーマットする辞書

        Returns:
            整形されたPythonコード文字列
        """
        return pprint.pformat(d, width=80)

    def _serialize_transform(self, transform: Any) -> str:
        """transformオブジェクトをPythonコード文字列に変換.

        Args:
            transform: torchvision.transforms オブジェクト

        Returns:
            Pythonコードとして実行可能な文字列
        """
        # Composeオブジェクトかどうかを厳密にチェック
        if hasattr(transform, "transforms") and isinstance(transform.transforms, list):
            items = [f"transforms.{repr(t)}" for t in transform.transforms]
            return (
                "transforms.Compose(\n    [\n        "
                + ",\n        ".join(items)
                + ",\n    ]\n)"
            )
        return f"transforms.{repr(transform)}"

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
        lines.append("# === ベース設定(最適化対象外) ===")

        # ベース設定のうち、最適化対象外のものを出力
        transform_lines = []
        for key, value in self._base_config.items():
            if key not in best_params:
                # transformは専用の処理で出力（明示的なキー名指定）
                if key in ("train_transform", "val_transform"):
                    transform_lines.append(
                        f"{key} = {self._serialize_transform(value)}"
                    )
                    continue
                # callableは出力しない（transformチェック後に行う）
                if callable(value):
                    continue
                # モジュールオブジェクトは出力しない
                if isinstance(value, type(json)):
                    continue
                # 辞書型は整形して出力
                if isinstance(value, dict):
                    lines.append(f"{key} = {self._format_dict(value)}")
                else:
                    lines.append(f"{key} = {repr(value)}")

        # transform設定を最後に追加
        if transform_lines:
            lines.append("")
            lines.append("# === Transform設定 ===")
            lines.extend(transform_lines)

        with open(config_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")


class StatisticsExporter(IResultExporter):
    """パラメータ重要度と試行統計をエクスポートする.

    Optunaのget_param_importances()を使用してパラメータ重要度を計算し,
    試行全体の統計情報（最良/最悪/平均/標準偏差）と共にJSON形式で出力.
    """

    def _calculate_trial_statistics(
        self, study: optuna.Study
    ) -> dict[str, float | int | None]:
        """試行統計を計算する.

        Args:
            study: Optuna Studyオブジェクト

        Returns:
            統計情報の辞書（best, worst, mean, std, n_completed, n_pruned, n_failed）
        """
        import numpy as np

        # 完了した試行のみ抽出
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        values = [t.value for t in completed_trials if t.value is not None]

        if not values:
            return {
                "best": None,
                "worst": None,
                "mean": None,
                "std": None,
                "n_completed": 0,
                "n_pruned": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
                "n_failed": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                ),
            }

        values_arr = np.array(values)
        is_maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

        return {
            "best": float(np.max(values_arr) if is_maximize else np.min(values_arr)),
            "worst": float(np.min(values_arr) if is_maximize else np.max(values_arr)),
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "n_completed": len(completed_trials),
            "n_pruned": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "n_failed": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        }

    def _calculate_param_importances(
        self, study: optuna.Study
    ) -> dict[str, float] | None:
        """パラメータ重要度を計算する.

        Args:
            study: Optuna Studyオブジェクト

        Returns:
            パラメータ名と重要度の辞書, 計算できない場合はNone
        """
        try:
            importances = optuna.importance.get_param_importances(study)
            return {k: float(v) for k, v in importances.items()}
        except Exception:
            # 試行数が少ない場合やエラー時はNoneを返す
            return None

    def export(
        self,
        _best_params: dict[str, Any],
        _best_value: float,
        study: optuna.Study,
        output_path: str,
    ) -> None:
        """パラメータ重要度と統計情報をJSONにエクスポートする.

        Args:
            _best_params: 最適なパラメータ（インターフェース準拠, 未使用）
            _best_value: 最適な目的関数値（インターフェース準拠, 未使用）
            study: Optuna Studyオブジェクト
            output_path: 出力先パス
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 統計情報を計算
        statistics = self._calculate_trial_statistics(study)

        # パラメータ重要度を計算
        importances = self._calculate_param_importances(study)

        result = {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "statistics": statistics,
            "param_importances": importances,
            "timestamp": datetime.now().isoformat(),
        }

        # JSON保存
        output_file = output_dir / "study_statistics.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


class VisualizationExporter(IResultExporter):
    """Plotlyを使用してインタラクティブなHTML可視化をエクスポートする.

    最適化履歴とパラメータ重要度のグラフをHTML形式で出力.
    """

    def __init__(self) -> None:
        """初期化."""
        import logging

        self._logger = logging.getLogger(__name__)

    def _export_optimization_history(
        self, study: optuna.Study, output_dir: Path
    ) -> bool:
        """最適化履歴グラフをHTMLにエクスポートする.

        Args:
            study: Optuna Studyオブジェクト
            output_dir: 出力ディレクトリ

        Returns:
            成功した場合True
        """
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(str(output_dir / "optimization_history.html"))
            return True
        except Exception as e:
            self._logger.warning(f"最適化履歴グラフの生成をスキップしました: {e}")
            return False

    def _export_param_importances(self, study: optuna.Study, output_dir: Path) -> bool:
        """パラメータ重要度グラフをHTMLにエクスポートする.

        Args:
            study: Optuna Studyオブジェクト
            output_dir: 出力ディレクトリ

        Returns:
            成功した場合True
        """
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(output_dir / "param_importances.html"))
            return True
        except Exception as e:
            self._logger.warning(f"パラメータ重要度グラフの生成をスキップしました: {e}")
            return False

    def _export_contour(self, study: optuna.Study, output_dir: Path) -> bool:
        """パラメータ間の等高線プロットをHTMLにエクスポートする.

        2つのパラメータ間の関係性を等高線で可視化する.

        Args:
            study: Optuna Studyオブジェクト
            output_dir: 出力ディレクトリ

        Returns:
            成功した場合True
        """
        try:
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(str(output_dir / "contour.html"))
            return True
        except Exception as e:
            self._logger.warning(f"等高線プロットの生成をスキップしました: {e}")
            return False

    def export(
        self,
        _best_params: dict[str, Any],
        _best_value: float,
        study: optuna.Study,
        output_path: str,
    ) -> None:
        """可視化グラフをHTMLにエクスポートする.

        Args:
            _best_params: 最適なパラメータ（インターフェース準拠, 未使用）
            _best_value: 最適な目的関数値（インターフェース準拠, 未使用）
            study: Optuna Studyオブジェクト
            output_path: 出力先パス
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 最適化履歴をエクスポート
        self._export_optimization_history(study, output_dir)

        # パラメータ重要度をエクスポート
        self._export_param_importances(study, output_dir)

        # 等高線プロットをエクスポート
        self._export_contour(study, output_dir)
