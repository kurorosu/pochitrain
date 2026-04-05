"""画像シリアライズ Strategy."""

import base64
from typing import Any, Protocol

import cv2
import numpy as np


class IImageSerializer(Protocol):
    """画像シリアライズの Strategy インターフェース."""

    def serialize(self, image: np.ndarray) -> dict[str, Any]:
        """Numpy 配列をシリアライズ可能な辞書に変換する.

        Args:
            image: 画像配列 (H, W, 3) uint8 BGR.

        Returns:
            シリアライズされた辞書.
        """
        ...

    def deserialize(self, data: dict[str, Any]) -> np.ndarray:
        """辞書から numpy 配列を復元する.

        Args:
            data: シリアライズされた辞書.

        Returns:
            画像配列 (H, W, 3) uint8 BGR.
        """
        ...


class RawArraySerializer:
    """raw numpy 配列の base64 エンコード. ローカル・開発用."""

    def serialize(self, image: np.ndarray) -> dict[str, Any]:
        """Numpy 配列を base64 エンコードする."""
        return {
            "image_data": base64.b64encode(image.tobytes()).decode("ascii"),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "format": "raw",
        }

    def deserialize(self, data: dict[str, Any]) -> np.ndarray:
        """base64 デコードして numpy 配列を復元する.

        Raises:
            ValueError: shape が未指定または不正な場合.
        """
        if "shape" not in data or data["shape"] is None:
            raise ValueError("raw フォーマットでは shape が必須です")

        shape = tuple(data["shape"])
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(
                f"shape は (H, W, 3) である必要があります. 受け取った: {shape}"
            )

        raw_bytes = base64.b64decode(data["image_data"])
        dtype = np.dtype(data.get("dtype", "uint8"))
        return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)


class JpegSerializer:
    """JPEG 圧縮. ネットワーク転送用."""

    def __init__(self, quality: int = 90) -> None:
        """初期化する.

        Args:
            quality: JPEG 品質 (1-100).
        """
        self.quality = quality

    def serialize(self, image: np.ndarray) -> dict[str, Any]:
        """画像を JPEG 圧縮して base64 エンコードする."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, buf = cv2.imencode(".jpg", image, encode_params)
        return {
            "image_data": base64.b64encode(buf.tobytes()).decode("ascii"),
            "format": "jpeg",
        }

    def deserialize(self, data: dict[str, Any]) -> np.ndarray:
        """base64 デコードして JPEG を numpy 配列に復元する.

        Raises:
            ValueError: JPEG デコードに失敗した場合.
        """
        jpeg_bytes = base64.b64decode(data["image_data"])
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG デコード失敗: 不正または破損した画像データ")
        return image


def create_serializer(fmt: str) -> IImageSerializer:
    """指定された形式のシリアライザを生成する.

    Args:
        fmt: 形式 ("raw" or "jpeg").

    Returns:
        シリアライザインスタンス.

    Raises:
        ValueError: サポートされていない形式の場合.
    """
    if fmt == "raw":
        return RawArraySerializer()
    if fmt == "jpeg":
        return JpegSerializer()
    raise ValueError(
        f"サポートされていない形式: {fmt}. 'raw' or 'jpeg' を指定してください"
    )
