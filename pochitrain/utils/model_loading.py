"""モデル読み込みの共通ユーティリティ."""

from pathlib import Path
from typing import Any

import torch


def _load_torch_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    """weights_only 互換付きでチェックポイントを読み込む."""
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # 古いPyTorchとの互換のため, weights_only未対応時のみフォールバック
        return torch.load(checkpoint_path, map_location=device)


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """チェックポイントを読み込み, モデルへ state_dict を適用する.

    Args:
        model: state_dictを適用するモデル.
        checkpoint_path: チェックポイントファイルパス.
        device: map_location に使うデバイス.

    Returns:
        復元メタ情報. キーは `epoch`, `best_accuracy`.

    Raises:
        FileNotFoundError: チェックポイントが存在しない場合.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {checkpoint_path}")

    checkpoint = _load_torch_checkpoint(checkpoint_path, device)

    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            metadata["epoch"] = checkpoint["epoch"]
        if "best_accuracy" in checkpoint:
            metadata["best_accuracy"] = checkpoint["best_accuracy"]
        return metadata

    model.load_state_dict(checkpoint)
    return metadata
