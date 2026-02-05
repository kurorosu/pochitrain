"""推論結果出力の Facade.

CSV出力, 混同行列画像, クラス別精度レポートを一括で出力する.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..training.evaluator import Evaluator
from ..utils.directory_manager import InferenceWorkspaceManager
from .csv_exporter import InferenceCSVExporter


class InferenceResultExporter:
    """推論結果の出力を統括する Facade.

    CSV出力, 混同行列画像, クラス別精度レポートを一括で出力する.

    Args:
        workspace_manager: 推論ワークスペース管理クラス
        logger: ロガーインスタンス
    """

    def __init__(
        self,
        workspace_manager: InferenceWorkspaceManager,
        logger: logging.Logger,
    ):
        """InferenceResultExporterを初期化."""
        self.workspace_manager = workspace_manager
        self.workspace: Optional[Path] = None
        self.logger = logger

    def _ensure_workspace(self) -> Path:
        """必要時にワークスペースを遅延作成.

        Returns:
            Path: 作成されたワークスペースのパス
        """
        if self.workspace is None:
            self.workspace = self.workspace_manager.create_workspace()
            self.logger.debug(f"推論ワークスペースを作成: {self.workspace}")
        return self.workspace

    def export(
        self,
        image_paths: List[str],
        predicted_labels: List[int],
        true_labels: List[int],
        confidence_scores: List[float],
        class_names: List[str],
        model_info: Dict[str, Any],
        results_filename: str = "inference_results.csv",
        summary_filename: Optional[str] = None,
        cm_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """推論結果を一括エクスポート.

        実行内容:
        1. ワークスペース確保 (遅延作成)
        2. 精度計算 (Evaluator 経由)
        3. 詳細結果CSV出力 (InferenceCSVExporter 経由)
        4. サマリーCSV出力 (summary_filename が指定されている場合のみ)
        5. モデル情報JSON保存
        6. 混同行列画像生成
        7. クラス別精度レポート生成

        Args:
            image_paths: 画像パスのリスト
            predicted_labels: 推論ラベルのリスト
            true_labels: 正解ラベルのリスト
            confidence_scores: 信頼度のリスト
            class_names: クラス名のリスト
            model_info: モデル情報の辞書
            results_filename: 詳細結果CSVファイル名
            summary_filename: サマリーCSVファイル名 (None の場合は出力スキップ)
            cm_config: 混同行列可視化設定

        Returns:
            Tuple[Path, Optional[Path]]: (詳細結果CSVパス, サマリーCSVパス)
        """
        # 推論ワークスペースを確保(遅延作成)
        inference_workspace = self._ensure_workspace()

        # CSV出力器を作成(推論ワークスペースを指定)
        csv_exporter = InferenceCSVExporter(
            output_dir=str(inference_workspace), logger=self.logger
        )

        # 精度計算 (Evaluator 経由)
        evaluator = Evaluator(
            device=None,  # type: ignore[arg-type]
            logger=self.logger,
        )
        accuracy_info = evaluator.calculate_accuracy(predicted_labels, true_labels)

        # 詳細結果のCSV出力
        results_csv = csv_exporter.export_results(
            image_paths=image_paths,
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            confidence_scores=confidence_scores,
            class_names=class_names,
            model_info=model_info,
            filename=results_filename,
        )

        # サマリーのCSV出力
        summary_csv = None
        if summary_filename:
            summary_csv = csv_exporter.export_summary(
                accuracy_info=accuracy_info,
                model_info=model_info,
                filename=summary_filename,
            )

        # モデル情報もJSONで保存
        self.workspace_manager.save_model_info(model_info)

        # 混同行列画像を自動生成
        try:
            confusion_matrix_path = self.save_confusion_matrix_image(
                predicted_labels=predicted_labels,
                true_labels=true_labels,
                class_names=class_names,
                cm_config=cm_config,
            )
            self.logger.debug(f"混同行列画像も生成されました: {confusion_matrix_path}")
        except Exception as e:
            self.logger.warning(f"混同行列画像生成に失敗しました: {e}")

        # クラス別精度レポートを自動生成
        try:
            from ..utils.inference_utils import save_classification_report

            report_path = save_classification_report(
                predicted_labels=predicted_labels,
                true_labels=true_labels,
                class_names=class_names,
                output_dir=inference_workspace,
            )
            self.logger.debug(f"クラス別精度レポートも生成されました: {report_path}")
        except Exception as e:
            self.logger.warning(f"クラス別精度レポート生成に失敗しました: {e}")

        return results_csv, summary_csv

    def save_confusion_matrix_image(
        self,
        predicted_labels: List[int],
        true_labels: List[int],
        class_names: List[str],
        filename: str = "confusion_matrix.png",
        cm_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """混同行列の画像を保存.

        Args:
            predicted_labels: 予測ラベルのリスト
            true_labels: 正解ラベルのリスト
            class_names: クラス名のリスト
            filename: 保存ファイル名
            cm_config: 混同行列可視化設定

        Returns:
            Path: 保存されたファイルのパス
        """
        from ..utils.inference_utils import save_confusion_matrix_image

        # 推論ワークスペースを確保(遅延作成)
        inference_workspace = self._ensure_workspace()

        return save_confusion_matrix_image(
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            class_names=class_names,
            output_dir=inference_workspace,
            filename=filename,
            cm_config=cm_config,
        )

    def get_workspace_info(self) -> Dict[str, Any]:
        """推論ワークスペース情報を取得.

        Returns:
            Dict[str, Any]: ワークスペース情報
        """
        if self.workspace is not None:
            return self.workspace_manager.get_workspace_info()
        else:
            return {
                "workspace": None,
                "workspace_name": None,
                "base_dir": str(self.workspace_manager.base_dir),
                "exists": False,
            }
