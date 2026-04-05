"""画像シリアライザのテスト."""

import numpy as np
import pytest

from pochitrain.api.serializers import (
    JpegSerializer,
    RawArraySerializer,
    create_serializer,
)

try:
    import cv2  # noqa: F401

    _has_cv2 = True
except ImportError:
    _has_cv2 = False

_requires_cv2 = pytest.mark.skipif(not _has_cv2, reason="cv2 が必要")


@pytest.fixture
def sample_image():
    """テスト用の BGR 画像配列を生成する."""
    return np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)


class TestRawArraySerializer:
    """RawArraySerializer のテスト."""

    def test_roundtrip(self, sample_image):
        """シリアライズ → デシリアライズで元の配列に戻ることを確認."""
        serializer = RawArraySerializer()
        data = serializer.serialize(sample_image)
        restored = serializer.deserialize(data)
        np.testing.assert_array_equal(restored, sample_image)

    def test_serialized_format(self, sample_image):
        """シリアライズ結果のフォーマットを確認."""
        serializer = RawArraySerializer()
        data = serializer.serialize(sample_image)
        assert data["format"] == "raw"
        assert data["shape"] == [48, 64, 3]
        assert data["dtype"] == "uint8"
        assert isinstance(data["image_data"], str)


@_requires_cv2
class TestJpegSerializer:
    """JpegSerializer のテスト."""

    def test_roundtrip_shape(self, sample_image):
        """シリアライズ → デシリアライズで同じ shape に戻ることを確認."""
        serializer = JpegSerializer()
        data = serializer.serialize(sample_image)
        restored = serializer.deserialize(data)
        assert restored.shape == sample_image.shape
        assert restored.dtype == np.uint8

    def test_serialized_format(self, sample_image):
        """シリアライズ結果のフォーマットを確認."""
        serializer = JpegSerializer()
        data = serializer.serialize(sample_image)
        assert data["format"] == "jpeg"
        assert isinstance(data["image_data"], str)
        assert "shape" not in data

    def test_compression_reduces_size(self, sample_image):
        """JPEG 圧縮で raw よりデータ量が小さくなることを確認."""
        raw = RawArraySerializer()
        jpeg = JpegSerializer()
        raw_data = raw.serialize(sample_image)
        jpeg_data = jpeg.serialize(sample_image)
        assert len(jpeg_data["image_data"]) < len(raw_data["image_data"])


class TestCreateSerializer:
    """create_serializer のテスト."""

    def test_raw(self):
        """raw 形式のシリアライザが生成されることを確認."""
        serializer = create_serializer("raw")
        assert isinstance(serializer, RawArraySerializer)

    def test_jpeg(self):
        """jpeg 形式のシリアライザが生成されることを確認."""
        serializer = create_serializer("jpeg")
        assert isinstance(serializer, JpegSerializer)

    def test_invalid_format(self):
        """不正な形式で ValueError が発生することを確認."""
        with pytest.raises(ValueError, match="サポートされていない形式"):
            create_serializer("png")
