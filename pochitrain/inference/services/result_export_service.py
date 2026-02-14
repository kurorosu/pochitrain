"""推論結果の出力を担当するサービス層."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pochitrain.logging import LoggerManager
from pochitrain.utils import (
    save_classification_report,
    save_confusion_matrix_image,
    write_inference_csv,
    write_inference_summary,
)

from ..types.result_export_types import ResultExportRequest, ResultExportResult


class ResultExportService:
    """推論結果の出力処理を一括で実行するサービス."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        self.logger = logger or LoggerManager().get_logger(__name__)

    def export(self, request: ResultExportRequest) -> ResultExportResult:
        """CSV, summary, および補助成果物を出力する.

        Args:
            request: 出力処理に必要な入力パラメータ.

        Returns:
            生成した成果物のパスと算出した精度(%).
        """
        accuracy = (
            (request.correct / request.num_samples) * 100.0
            if request.num_samples > 0
            else 0.0
        )

        results_csv_path = write_inference_csv(
            output_dir=request.output_dir,
            image_paths=request.image_paths,
            predictions=request.predictions,
            true_labels=request.true_labels,
            confidences=request.confidences,
            class_names=request.class_names,
            filename=request.results_filename,
        )

        summary_path = write_inference_summary(
            output_dir=request.output_dir,
            model_path=request.model_path,
            data_path=request.data_path,
            num_samples=request.num_samples,
            accuracy=accuracy,
            avg_time_per_image=request.avg_time_per_image,
            total_samples=request.total_samples,
            warmup_samples=request.warmup_samples,
            avg_total_time_per_image=request.avg_total_time_per_image,
            input_size=request.input_size,
            filename=request.summary_filename,
            extra_info=request.extra_info,
        )

        confusion_matrix_path = None
        try:
            confusion_matrix_path = save_confusion_matrix_image(
                predicted_labels=request.predictions,
                true_labels=request.true_labels,
                class_names=request.class_names,
                output_dir=request.output_dir,
                filename=request.confusion_matrix_filename,
                cm_config=request.cm_config,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                f"混同行列画像の保存に失敗しました, error: {exc}",
            )

        classification_report_path = None
        try:
            classification_report_path = save_classification_report(
                predicted_labels=request.predictions,
                true_labels=request.true_labels,
                class_names=request.class_names,
                output_dir=request.output_dir,
                filename=request.classification_report_filename,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                f"分類レポートの保存に失敗しました, error: {exc}",
            )

        model_info_path = self._write_model_info(
            output_dir=request.output_dir,
            model_info=request.model_info,
            filename=request.model_info_filename,
        )

        return ResultExportResult(
            results_csv_path=results_csv_path,
            summary_path=summary_path,
            confusion_matrix_path=confusion_matrix_path,
            classification_report_path=classification_report_path,
            model_info_path=model_info_path,
            accuracy=accuracy,
        )

    def _write_model_info(
        self,
        output_dir: Path,
        model_info: Optional[dict[str, Any]],
        filename: str,
    ) -> Optional[Path]:
        """モデル情報をJSONで保存する.

        Args:
            output_dir: 出力ディレクトリ.
            model_info: モデル情報辞書.
            filename: 保存ファイル名.

        Returns:
            保存ファイルパス. model_infoが未指定、または保存失敗時はNone.
        """
        if model_info is None:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        model_info_path = output_dir / filename
        try:
            with open(model_info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            return model_info_path
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                f"モデル情報JSONの保存に失敗しました, error: {exc}",
            )
            return None
