"""推論エンドポイント."""

import time

from fastapi import APIRouter, HTTPException

from pochitrain.api.schemas import PredictRequest, PredictResponse
from pochitrain.api.serializers import create_serializer
from pochitrain.logging import LoggerManager

logger = LoggerManager().get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """単一画像を推論する.

    cv2 でキャプチャした画像データを受け取り, クラス予測と信頼度を返す.
    """
    from pochitrain.api.app import get_engine

    engine = get_engine()

    try:
        serializer = create_serializer(request.format)
        image = serializer.deserialize(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("画像デシリアライズエラー")
        raise HTTPException(
            status_code=500,
            detail="画像処理中にエラーが発生しました",
        ) from e

    start = time.perf_counter()
    try:
        result = engine.predict(image)
    except Exception as e:
        logger.exception("推論エラー")
        raise HTTPException(
            status_code=500,
            detail="推論中にエラーが発生しました",
        ) from e
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "推論完了: class=%d, confidence=%.3f, e2e_time=%.1fms",
        result["class_id"],
        result["confidence"],
        elapsed_ms,
    )

    return PredictResponse(
        class_id=result["class_id"],
        class_name=result["class_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        e2e_time_ms=round(elapsed_ms, 3),
        backend=engine.backend,
    )
