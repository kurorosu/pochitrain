# 推論 API クライアント使用例

pochitrain 推論 API サーバーにリクエストを送信するクライアントコードの例です.

## サーバー起動

```bash
uv run pochi serve work_dirs/20260206_001/models/best_epoch50.pth
```

## raw 形式 (numpy 配列)

cv2 でキャプチャした numpy 配列をそのまま送信する方式. ローカル環境向け.

```python
import base64

import numpy as np
import requests

# cv2 でキャプチャした画像 (H, W, 3) uint8 BGR
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

payload = {
    "image_data": base64.b64encode(image.tobytes()).decode(),
    "shape": list(image.shape),
    "dtype": "uint8",
    "format": "raw",
}

response = requests.post("http://localhost:8000/api/v1/predict", json=payload)
result = response.json()

print(f"クラス: {result['class_name']} (ID: {result['class_id']})")
print(f"信頼度: {result['confidence']:.3f}")
print(f"推論時間: {result['processing_time_ms']:.1f}ms")
```

## jpeg 形式 (圧縮転送)

JPEG 圧縮して送信する方式. ネットワーク越しの通信向け (データ量が約 1/20 に削減).

```python
import base64

import cv2
import requests

# cv2 でキャプチャした画像
image = cv2.imread("test_image.jpg")

# JPEG 圧縮
_, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])

payload = {
    "image_data": base64.b64encode(buf.tobytes()).decode(),
    "format": "jpeg",
}

response = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(response.json())
```

## pochitrain のシリアライザを使用

pochitrain に含まれるシリアライザを使うと, エンコード処理を省略できます.

```python
import numpy as np
import requests

from pochitrain.api.serializers import RawArraySerializer

image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

serializer = RawArraySerializer()
payload = serializer.serialize(image)

response = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(response.json())
```

## pochivision からの使用例

pochivision のプロセッサまたはフィーチャーエクストラクタとして組み込む場合のイメージ:

```python
import base64

import numpy as np
import requests

from pochivision.processors.base import BaseProcessor


class ApiClassifierProcessor(BaseProcessor):
    """pochitrain API サーバーに画像を送信して分類結果を取得するプロセッサ."""

    def __init__(self, name: str, config: dict) -> None:
        super().__init__(name, config)
        self.api_url = config.get("api_url", "http://localhost:8000/api/v1/predict")

    def process(self, image: np.ndarray) -> np.ndarray:
        """画像を API に送信し, 分類結果をオーバーレイして返す."""
        payload = {
            "image_data": base64.b64encode(image.tobytes()).decode(),
            "shape": list(image.shape),
            "dtype": "uint8",
            "format": "raw",
        }
        response = requests.post(self.api_url, json=payload)
        result = response.json()

        # 結果をオーバーレイ表示 (cv2 はpochivision側で利用可能)
        import cv2

        text = f"{result['class_name']} ({result['confidence']:.2f})"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

    @staticmethod
    def get_default_config() -> dict:
        return {"api_url": "http://localhost:8000/api/v1/predict"}
```

## 補助エンドポイント

```python
import requests

BASE = "http://localhost:8000/api/v1"

# ヘルスチェック
print(requests.get(f"{BASE}/health").json())
# {"status": "healthy", "model_loaded": true, "backend": "pytorch"}

# モデル情報
print(requests.get(f"{BASE}/model-info").json())
# {"model_name": "resnet18", "num_classes": 4, "class_names": [...], ...}

# バージョン情報
print(requests.get(f"{BASE}/version").json())
# {"pochitrain_version": "1.8.3", "api_version": "1.0.0", ...}
```

## レスポンス形式

```json
{
  "class_id": 2,
  "class_name": "cat",
  "confidence": 0.95,
  "probabilities": [0.01, 0.04, 0.95, 0.00],
  "processing_time_ms": 12.3,
  "backend": "pytorch"
}
```
