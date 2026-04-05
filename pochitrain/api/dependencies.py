"""FastAPI 依存性注入."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from pochitrain.models.pochi_models import create_model
from pochitrain.utils.model_loading import load_model_from_checkpoint


class InferenceEngine:
    """推論エンジン. サーバー起動時にモデルをロードし, リクエストごとに推論を実行する."""

    def __init__(
        self,
        model_path: Path,
        config: dict[str, Any],
        backend: str = "pytorch",
    ) -> None:
        """推論エンジンを初期化する.

        Args:
            model_path: 学習済みモデルファイルパス.
            config: pochitrain 設定辞書.
            backend: 推論バックエンド.
        """
        self.backend = backend
        self.model_name: str = config["model_name"]
        self.num_classes: int = config["num_classes"]
        self.device_name: str = config.get("device", "cuda")
        self.device = torch.device(self.device_name)
        self.val_transform = config.get("val_transform")
        self.class_names: list[str] = []
        self.model_path = model_path

        model = create_model(self.model_name, self.num_classes, pretrained=False)
        self.metadata = load_model_from_checkpoint(model, model_path, self.device)
        model.to(self.device)
        model.eval()
        self.model = model

    def set_class_names(self, class_names: list[str]) -> None:
        """クラス名を設定する.

        Args:
            class_names: クラス名のリスト.
        """
        self.class_names = class_names

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """単一画像を推論する.

        Args:
            image: BGR 画像配列 (H, W, 3) uint8.

        Returns:
            推論結果辞書.
        """
        # BGR → RGB → PIL Image (val_transform が ToTensor を含む前提)
        rgb = image[:, :, ::-1].copy()
        pil_image = Image.fromarray(rgb)

        if self.val_transform is None:
            raise RuntimeError("val_transform が設定されていません")

        tensor = self.val_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence_val, class_id_val = probs.max(dim=1)

        probabilities: list[float] = probs[0].cpu().tolist()

        class_id = int(class_id_val.item())
        confidence = float(confidence_val.item())

        class_name = (
            self.class_names[class_id]
            if class_id < len(self.class_names)
            else str(class_id)
        )

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def get_model_info(self) -> dict[str, Any]:
        """モデル情報を返す."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": self.device_name,
            "backend": self.backend,
        }
