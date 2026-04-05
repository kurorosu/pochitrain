"""FastAPI アプリケーション."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI

from pochitrain.api.config import ServerConfig
from pochitrain.api.dependencies import InferenceEngine
from pochitrain.api.routers import health, inference
from pochitrain.utils.config_loader import ConfigLoader
from pochitrain.utils.inference_utils import auto_detect_config_path

logger = logging.getLogger(__name__)

_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """グローバル推論エンジンを取得する.

    Raises:
        RuntimeError: エンジンが初期化されていない場合.
    """
    if _engine is None:
        raise RuntimeError("推論エンジンが初期化されていません")
    return _engine


def _resolve_config_path(server_config: ServerConfig) -> Path:
    """設定ファイルパスを解決する."""
    if server_config.config_path is not None:
        return server_config.config_path
    return auto_detect_config_path(server_config.model_path)


def _load_class_names(config: dict[str, Any]) -> list[str]:
    """Config の val_data_root からクラス名を取得する."""
    val_data_root = config.get("val_data_root")
    if val_data_root is None:
        return []

    data_path = Path(val_data_root)
    if not data_path.exists():
        return []

    return sorted(d.name for d in data_path.iterdir() if d.is_dir())


def _create_lifespan(
    server_config: ServerConfig,
) -> Callable[["FastAPI"], Any]:
    """Lifespan コンテキストマネージャを生成する."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _engine

        config_path = _resolve_config_path(server_config)
        logger.info("設定ファイル読み込み: %s", config_path)

        config = ConfigLoader.load_config(str(config_path))

        logger.info("モデル読み込み: %s", server_config.model_path)
        _engine = InferenceEngine(
            model_path=server_config.model_path,
            config=config,
            backend=server_config.backend,
        )

        class_names = _load_class_names(config)
        if class_names:
            _engine.set_class_names(class_names)
            logger.info("クラス名: %s", class_names)

        logger.info(
            "サーバー起動完了 (backend=%s, device=%s)",
            server_config.backend,
            _engine.device_name,
        )

        yield

        _engine = None
        logger.info("サーバーシャットダウン完了")

    return lifespan


def create_app(server_config: ServerConfig | None = None) -> FastAPI:
    """アプリケーションを生成する.

    Args:
        server_config: サーバー設定. テスト時に None を渡すと lifespan なしで生成する.

    Returns:
        FastAPI アプリケーション.
    """
    lifespan = _create_lifespan(server_config) if server_config else None

    app = FastAPI(
        title="pochitrain Inference API",
        version="1.0.0",
        description="pochitrain 推論 API サーバー",
        lifespan=lifespan,
    )

    app.include_router(inference.router)
    app.include_router(health.router)

    return app
