"""API スキーマのテスト."""

import pytest
from pydantic import ValidationError

from pochitrain.api.schemas import PredictRequest, PredictResponse


class TestPredictRequest:
    """PredictRequest のテスト."""

    def test_default_format(self):
        """デフォルト format が raw であることを確認."""
        req = PredictRequest(image_data="abc123", shape=[480, 640, 3])
        assert req.format == "raw"
        assert req.dtype == "uint8"

    def test_jpeg_format(self):
        """jpeg 形式でリクエストが作成できることを確認."""
        req = PredictRequest(image_data="abc123", format="jpeg")
        assert req.format == "jpeg"

    def test_raw_without_shape_raises(self):
        """raw 形式で shape 未指定時に ValidationError が発生することを確認."""
        with pytest.raises(ValidationError, match="shape"):
            PredictRequest(image_data="abc123", format="raw")

    def test_invalid_dtype_raises(self):
        """不正な dtype で ValidationError が発生することを確認."""
        with pytest.raises(ValidationError, match="dtype"):
            PredictRequest(image_data="abc123", shape=[480, 640, 3], dtype="invalid")

    def test_raw_with_shape(self):
        """raw 形式で shape を指定できることを確認."""
        req = PredictRequest(
            image_data="abc123",
            format="raw",
            shape=[480, 640, 3],
        )
        assert req.shape == [480, 640, 3]

    def test_invalid_format(self):
        """不正な format で ValidationError が発生することを確認."""
        with pytest.raises(ValidationError):
            PredictRequest(image_data="abc123", format="bmp")


class TestPredictResponse:
    """PredictResponse のテスト."""

    def test_valid_response(self):
        """正常なレスポンスが作成できることを確認."""
        resp = PredictResponse(
            class_id=2,
            class_name="cat",
            confidence=0.95,
            probabilities=[0.01, 0.04, 0.95],
            e2e_time_ms=12.3,
            backend="pytorch",
        )
        assert resp.class_id == 2
        assert resp.class_name == "cat"
        assert resp.confidence == 0.95
