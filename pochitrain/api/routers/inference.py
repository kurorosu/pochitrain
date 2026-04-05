"""推論エンドポイント."""

import time

from fastapi import APIRouter, HTTPException

from pochitrain.api.schemas import PredictRequest, PredictResponse
from pochitrain.api.serializers import create_serializer

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
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"画像データのデシリアライズに失敗: {e}",
        ) from e

    start = time.perf_counter()
    result = engine.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        class_id=result["class_id"],
        class_name=result["class_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=round(elapsed_ms, 3),
        backend=engine.backend,
    )
