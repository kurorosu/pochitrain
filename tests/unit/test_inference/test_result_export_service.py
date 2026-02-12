"""ResultExportService のテスト."""

import json
import logging
from pathlib import Path

import pytest

from pochitrain.inference.services.result_export_service import ResultExportService
from pochitrain.inference.types.result_export_types import ResultExportRequest


class _ListHandler(logging.Handler):
    """warning ログを収集するテスト用ハンドラ."""

    def __init__(self) -> None:
        """ハンドラを初期化する."""
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """ログメッセージを記録する.

        Args:
            record: ログレコード.
        """
        self.messages.append(record.getMessage())


def _build_test_logger(name: str) -> tuple[logging.Logger, _ListHandler]:
    """テスト用ロガーと収集ハンドラを返す.

    Args:
        name: ロガー名.

    Returns:
        ロガーと warning 収集用ハンドラ.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    return logger, handler


def _build_request(tmp_path: Path) -> ResultExportRequest:
    """テスト用の最小リクエストを生成する."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return ResultExportRequest(
        output_dir=output_dir,
        model_path=tmp_path / "model.onnx",
        data_path=tmp_path / "data",
        image_paths=["a.jpg", "b.jpg"],
        predictions=[0, 1],
        true_labels=[0, 0],
        confidences=[0.9, 0.8],
        class_names=["cat", "dog"],
        num_samples=2,
        correct=1,
        avg_time_per_image=1.5,
        total_samples=2,
        warmup_samples=1,
        model_info={"model_name": "resnet18", "num_classes": 2},
        avg_total_time_per_image=2.0,
        input_size=(3, 224, 224),
        results_filename="results.csv",
        summary_filename="summary.txt",
        extra_info={"pipeline": "gpu"},
        cm_config={"title": "cm"},
    )


def test_export_writes_all_artifacts(tmp_path: Path):
    """全成果物を実際に出力し, 返却パスと整合することを確認する."""
    request = _build_request(tmp_path)

    results_path = request.output_dir / request.results_filename
    summary_path = request.output_dir / request.summary_filename
    cm_path = request.output_dir / request.confusion_matrix_filename
    report_path = request.output_dir / request.classification_report_filename
    model_info_path = request.output_dir / request.model_info_filename

    logger, _ = _build_test_logger("test_result_export_service_success")
    service = ResultExportService(logger=logger)
    result = service.export(request)

    assert result.results_csv_path == results_path
    assert result.summary_path == summary_path
    assert result.confusion_matrix_path == cm_path
    assert result.classification_report_path == report_path
    assert result.model_info_path == model_info_path
    assert result.accuracy == pytest.approx(50.0)
    assert results_path.exists()
    assert summary_path.exists()
    assert cm_path.exists()
    assert report_path.exists()
    assert model_info_path.exists()
    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)
    assert model_info["model_name"] == "resnet18"


def test_export_continues_when_optional_exports_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """補助成果物の出力失敗時も処理継続することを確認する."""
    request = _build_request(tmp_path)

    results_path = request.output_dir / request.results_filename
    summary_path = request.output_dir / request.summary_filename

    def _raise_cm_error(**_: object) -> Path:
        raise RuntimeError("cm failed")

    def _raise_report_error(**_: object) -> Path:
        raise RuntimeError("report failed")

    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.save_confusion_matrix_image",
        _raise_cm_error,
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.save_classification_report",
        _raise_report_error,
    )

    logger, handler = _build_test_logger("test_result_export_service_failure")
    service = ResultExportService(logger=logger)
    result = service.export(request)

    assert result.results_csv_path == results_path
    assert result.summary_path == summary_path
    assert result.confusion_matrix_path is None
    assert result.classification_report_path is None
    assert result.model_info_path == request.output_dir / request.model_info_filename
    assert result.accuracy == pytest.approx(50.0)
    assert results_path.exists()
    assert summary_path.exists()
    assert len(handler.messages) == 2
