"""
推論結果のCSV出力機能.

推論結果を構造化されたCSVファイルとして出力します。
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pochitrain.exporters import BaseCSVExporter


class InferenceCSVExporter(BaseCSVExporter):
    """
    推論結果をCSVファイルに出力するクラス.

    Args:
        output_dir (str): 出力ディレクトリ
        logger (logging.Logger, optional): ロガーインスタンス
    """

    def __init__(
        self,
        output_dir: str = "inference_results",
        logger: Optional[logging.Logger] = None,
    ):
        """InferenceCSVExporterを初期化."""
        super().__init__(output_dir=output_dir, logger=logger)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_results(
        self,
        image_paths: List[str],
        predicted_labels: List[int],
        true_labels: List[int],
        confidence_scores: List[float],
        class_names: List[str],
        model_info: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        推論結果をCSVファイルに出力.

        Args:
            image_paths (List[str]): 画像パスのリスト
            predicted_labels (List[int]): 推論ラベルのリスト
            true_labels (List[int]): 正解ラベルのリスト
            confidence_scores (List[float]): 信頼度のリスト
            class_names (List[str]): クラス名のリスト
            model_info (Dict[str, any], optional): モデル情報
            filename (str, optional): 出力ファイル名

        Returns:
            Path: 出力されたCSVファイルのパス
        """
        filename = self._generate_filename("inference_results", filename)
        output_path = self._build_output_path(filename)

        # データの検証
        self._validate_data(
            image_paths, predicted_labels, true_labels, confidence_scores
        )

        # CSVファイルの書き込み
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # ヘッダーの書き込み
            self._write_header(writer, model_info)

            # データの書き込み
            self._write_data(
                writer,
                image_paths,
                predicted_labels,
                true_labels,
                confidence_scores,
                class_names,
            )

        self.logger.info(f"推論結果をCSVに出力: {output_path}")
        return output_path

    def _validate_data(
        self,
        image_paths: List[str],
        predicted_labels: List[int],
        true_labels: List[int],
        confidence_scores: List[float],
    ) -> None:
        """データの整合性をチェック."""
        lengths = [
            len(image_paths),
            len(predicted_labels),
            len(true_labels),
            len(confidence_scores),
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                f"データの長さが一致しません: "
                f"paths={lengths[0]}, predicted={lengths[1]}, "
                f"true={lengths[2]}, confidence={lengths[3]}"
            )

    def _write_header(self, writer: Any, model_info: Optional[Dict[str, Any]]) -> None:
        """CSVのヘッダー部分を書き込み."""
        # モデル情報をコメントとして追加
        if model_info:
            writer.writerow([f"# Model: {model_info.get('model_name', 'unknown')}"])
            writer.writerow([f"# Classes: {model_info.get('num_classes', 'unknown')}"])
            writer.writerow([f"# Device: {model_info.get('device', 'unknown')}"])
            writer.writerow(
                [f"# Model Path: {model_info.get('model_path', 'unknown')}"]
            )

            if model_info.get("best_accuracy") is not None:
                writer.writerow(
                    [f"# Best Accuracy: {model_info['best_accuracy']:.2f}%"]
                )

            if model_info.get("epoch") is not None:
                writer.writerow([f"# Epoch: {model_info['epoch']}"])

            writer.writerow([])  # 空行

        # カラムヘッダー
        writer.writerow(
            [
                "image_path",
                "predicted_class",
                "true_class",
                "predicted_label",
                "true_label",
                "is_correct",
                "confidence",
            ]
        )

    def _write_data(
        self,
        writer: Any,
        image_paths: List[str],
        predicted_labels: List[int],
        true_labels: List[int],
        confidence_scores: List[float],
        class_names: List[str],
    ) -> None:
        """データ行を書き込み."""
        for i, (path, pred_label, true_label, confidence) in enumerate(
            zip(image_paths, predicted_labels, true_labels, confidence_scores)
        ):
            # クラス名の取得
            pred_class = (
                class_names[pred_label] if pred_label < len(class_names) else "unknown"
            )
            true_class = (
                class_names[true_label] if true_label < len(class_names) else "unknown"
            )

            # 正解判定
            is_correct = pred_label == true_label

            # データ行の書き込み
            writer.writerow(
                [
                    path,
                    pred_class,
                    true_class,
                    pred_label,
                    true_label,
                    is_correct,
                    f"{confidence:.4f}",
                ]
            )

    def export_summary(
        self,
        accuracy_info: Dict[str, float],
        model_info: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        推論結果のサマリーをCSVファイルに出力.

        Args:
            accuracy_info (Dict[str, float]): 精度情報
            model_info (Dict[str, any], optional): モデル情報
            filename (str, optional): 出力ファイル名

        Returns:
            Path: 出力されたCSVファイルのパス
        """
        filename = self._generate_filename("inference_summary", filename)
        output_path = self._build_output_path(filename)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # ヘッダー
            writer.writerow(["metric", "value"])

            # 精度情報
            writer.writerow(["total_samples", accuracy_info.get("total_samples", 0)])
            writer.writerow(
                ["correct_predictions", accuracy_info.get("correct_predictions", 0)]
            )
            writer.writerow(
                [
                    "accuracy_percentage",
                    f"{accuracy_info.get('accuracy_percentage', 0):.2f}%",
                ]
            )

            # モデル情報
            if model_info:
                writer.writerow([])
                writer.writerow(["model_info", ""])
                for key, value in model_info.items():
                    writer.writerow([key, str(value)])

        self.logger.info(f"推論サマリーをCSVに出力: {output_path}")
        return output_path
