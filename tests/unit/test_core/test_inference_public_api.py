"""pochitrain.inference の公開APIテスト."""

import pochitrain.inference as inference


def test_public_api_exports_expected_symbols():
    """公開APIが新しいサービス/型のシンボルのみを公開する."""
    expected = {
        "ExecutionService",
        "ResultExportService",
        "ExecutionRequest",
        "ExecutionResult",
        "ResultExportRequest",
        "ResultExportResult",
    }
    assert set(inference.__all__) == expected


def test_public_symbols_are_importable():
    """公開シンボルがモジュール属性として参照できる."""
    for name in inference.__all__:
        assert hasattr(inference, name)


def test_legacy_exporters_are_not_public():
    """旧Exporterの公開が除外されている."""
    assert not hasattr(inference, "InferenceCSVExporter")
    assert not hasattr(inference, "InferenceResultExporter")
