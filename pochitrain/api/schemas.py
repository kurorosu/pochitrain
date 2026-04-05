"""API リクエスト/レスポンスのスキーマ定義."""

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictRequest(BaseModel):
    """推論リクエスト.

    cv2 でキャプチャした numpy 配列を base64 エンコードして送信する.
    format で raw (生配列) / jpeg (圧縮) を切り替える.
    """

    image_data: str = Field(description="base64 エンコードされた画像データ")
    format: Literal["raw", "jpeg"] = Field(
        default="raw",
        description="画像データ形式",
    )
    shape: list[int] | None = Field(
        default=None,
        description="numpy 配列の shape (raw 形式時に必須, 例: [480, 640, 3])",
    )
    dtype: str = Field(
        default="uint8",
        description="numpy 配列の dtype (raw 形式時に使用)",
    )

    @model_validator(mode="after")
    def validate_raw_format(self) -> Self:
        """Raw フォーマット時に shape が必須であることを検証する."""
        if self.format == "raw" and self.shape is None:
            raise ValueError("raw フォーマットでは shape が必須です")
        if self.format == "raw" and self.shape is not None and len(self.shape) != 3:
            raise ValueError("shape は [height, width, 3] の形式である必要があります")
        return self

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Dtype が有効な numpy dtype であることを検証する."""
        try:
            np.dtype(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"無効な dtype: {v}") from e
        return v


class PredictResponse(BaseModel):
    """推論レスポンス."""

    class_id: int = Field(description="予測クラス ID")
    class_name: str = Field(description="予測クラス名")
    confidence: float = Field(description="信頼度 (0.0-1.0)")
    probabilities: list[float] = Field(description="全クラスの確率")
    processing_time_ms: float = Field(description="推論時間 (ミリ秒)")
    backend: str = Field(description="使用バックエンド")


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス."""

    status: str = Field(description="サーバー状態")
    model_loaded: bool = Field(description="モデルロード済みか")
    backend: str = Field(description="使用バックエンド")


class ModelInfoResponse(BaseModel):
    """モデル情報レスポンス."""

    model_name: str = Field(description="モデル名")
    num_classes: int = Field(description="分類クラス数")
    class_names: list[str] = Field(description="クラス名一覧")
    device: str = Field(description="推論デバイス")
    backend: str = Field(description="使用バックエンド")


class VersionResponse(BaseModel):
    """バージョン情報レスポンス."""

    pochitrain_version: str = Field(description="pochitrain バージョン")
    api_version: str = Field(description="API バージョン")

    backend_versions: dict[str, str] = Field(
        description="バックエンドライブラリのバージョン",
    )
