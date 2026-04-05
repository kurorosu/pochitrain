"""ヘルスチェック・モデル情報エンドポイント."""

import sys

from fastapi import APIRouter

from pochitrain.api.schemas import HealthResponse, ModelInfoResponse, VersionResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """サーバーの状態を返す."""
    from pochitrain.api.app import get_engine

    try:
        engine = get_engine()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            backend=engine.backend,
        )
    except RuntimeError:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            backend="unknown",
        )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """ロード済みモデルの情報を返す."""
    from pochitrain.api.app import get_engine

    engine = get_engine()
    info = engine.get_model_info()
    return ModelInfoResponse(**info)


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """バージョン情報を返す."""
    import torch

    import pochitrain

    backend_versions: dict[str, str] = {
        "pytorch": torch.__version__,
        "python": sys.version.split()[0],
    }

    return VersionResponse(
        pochitrain_version=pochitrain.__version__,
        api_version="1.0.0",
        backend_versions=backend_versions,
    )
