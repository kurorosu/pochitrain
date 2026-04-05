"""API ルーターのテスト."""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pochitrain.api import app as app_module
from pochitrain.api.app import create_app
from pochitrain.api.dependencies import InferenceEngine
from pochitrain.api.serializers import RawArraySerializer


@pytest.fixture
def mock_engine():
    """モック推論エンジンを生成する."""
    engine = object.__new__(InferenceEngine)
    engine.backend = "pytorch"
    engine.model_name = "resnet18"
    engine.num_classes = 3
    engine.device_name = "cpu"
    engine.class_names = ["cat", "dog", "bird"]
    engine.get_model_info = lambda: {
        "model_name": "resnet18",
        "num_classes": 3,
        "class_names": ["cat", "dog", "bird"],
        "device": "cpu",
        "backend": "pytorch",
    }
    return engine


@pytest.fixture
def client(mock_engine):
    """テスト用 FastAPI クライアントを生成する."""
    app = create_app()
    original = app_module._engine
    app_module._engine = mock_engine
    try:
        yield TestClient(app)
    finally:
        app_module._engine = original


class TestHealthEndpoint:
    """GET /api/v1/health のテスト."""

    def test_health(self, client):
        """ヘルスチェックが正常に応答することを確認."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["backend"] == "pytorch"


class TestModelInfoEndpoint:
    """GET /api/v1/model-info のテスト."""

    def test_model_info(self, client):
        """モデル情報が正しく返されることを確認."""
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "resnet18"
        assert data["num_classes"] == 3
        assert data["class_names"] == ["cat", "dog", "bird"]


class TestVersionEndpoint:
    """GET /api/v1/version のテスト."""

    def test_version(self, client):
        """バージョン情報が返されることを確認."""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        data = response.json()
        assert "pochitrain_version" in data
        assert data["api_version"] == "1.0.0"
        assert "pytorch" in data["backend_versions"]


class TestPredictEndpoint:
    """POST /api/v1/predict のテスト."""

    def test_predict_raw(self, client, mock_engine):
        """raw 形式で推論が成功することを確認."""
        mock_engine.predict = lambda image: {
            "class_id": 0,
            "class_name": "cat",
            "confidence": 0.9,
            "probabilities": [0.9, 0.05, 0.05],
        }

        image = np.zeros((48, 64, 3), dtype=np.uint8)
        serializer = RawArraySerializer()
        payload = serializer.serialize(image)

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["class_id"] == 0
        assert data["class_name"] == "cat"
        assert data["confidence"] == 0.9
        assert data["backend"] == "pytorch"
        assert "processing_time_ms" in data

    def test_predict_invalid_data(self, client):
        """不正なデータで 400 エラーが返ることを確認."""
        response = client.post(
            "/api/v1/predict",
            json={
                "image_data": "invalid_base64!!!",
                "format": "raw",
                "shape": [48, 64, 3],
            },
        )
        assert response.status_code == 400
