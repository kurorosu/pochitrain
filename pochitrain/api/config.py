"""API サーバー設定."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """推論 API サーバーの起動設定."""

    model_path: Path = Field(description="学習済みモデルファイルパス")
    config_path: Path | None = Field(
        default=None,
        description="pochitrain 設定ファイルパス. 未指定時はモデルパスから自動検出",
    )
    backend: Literal["pytorch"] = Field(
        default="pytorch",
        description="推論バックエンド",
    )
    host: str = Field(
        default="127.0.0.1",
        description="バインドホスト",
    )
    port: int = Field(
        default=8000,
        description="バインドポート",
    )
