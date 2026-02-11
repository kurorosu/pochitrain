"""ResultExportService のテスト."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pochitrain.inference.services.result_export_service import ResultExportService
from pochitrain.inference.types.result_export_types import ResultExportRequest


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
        avg_total_time_per_image=2.0,
        input_size=(3, 224, 224),
        results_filename="results.csv",
        summary_filename="summary.txt",
        extra_info={"pipeline": "gpu"},
        cm_config={"title": "cm"},
    )


def test_export_writes_all_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """全成果物を出力し, パスを返すことを確認する."""
    request = _build_request(tmp_path)

    results_path = request.output_dir / request.results_filename
    summary_path = request.output_dir / request.summary_filename
    cm_path = request.output_dir / request.confusion_matrix_filename
    report_path = request.output_dir / request.classification_report_filename

    mock_csv = MagicMock(return_value=results_path)
    mock_summary = MagicMock(return_value=summary_path)
    mock_cm = MagicMock(return_value=cm_path)
    mock_report = MagicMock(return_value=report_path)

    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.write_inference_csv",
        mock_csv,
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.write_inference_summary",
        mock_summary,
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.save_confusion_matrix_image",
        mock_cm,
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.save_classification_report",
        mock_report,
    )

    service = ResultExportService(logger=MagicMock())
    result = service.export(request)

    assert result.results_csv_path == results_path
    assert result.summary_path == summary_path
    assert result.confusion_matrix_path == cm_path
    assert result.classification_report_path == report_path
    assert result.accuracy == pytest.approx(50.0)

    mock_csv.assert_called_once()
    mock_summary.assert_called_once()
    mock_cm.assert_called_once()
    mock_report.assert_called_once()
    assert mock_summary.call_args.kwargs["accuracy"] == pytest.approx(50.0)


def test_export_continues_when_optional_exports_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """補助成果物の出力失敗時も処理継続することを確認する."""
    request = _build_request(tmp_path)

    results_path = request.output_dir / request.results_filename
    summary_path = request.output_dir / request.summary_filename

    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.write_inference_csv",
        MagicMock(return_value=results_path),
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.write_inference_summary",
        MagicMock(return_value=summary_path),
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.save_confusion_matrix_image",
        MagicMock(side_effect=RuntimeError("cm failed")),
    )
    monkeypatch.setattr(
        "pochitrain.inference.services.result_export_service.save_classification_report",
        MagicMock(side_effect=RuntimeError("report failed")),
    )

    mock_logger = MagicMock()
    service = ResultExportService(logger=mock_logger)
    result = service.export(request)

    assert result.results_csv_path == results_path
    assert result.summary_path == summary_path
    assert result.confusion_matrix_path is None
    assert result.classification_report_path is None
    assert result.accuracy == pytest.approx(50.0)
    assert mock_logger.warning.call_count == 2
