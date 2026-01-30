"""チェックポイントの保存・読み込みを管理するモジュール."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn, optim


class CheckpointStore:
    """チェックポイントの保存・読み込みを管理するクラス.

    Repository パターンに基づき, モデル状態の永続化・復元を担当する.

    Args:
        work_dir: チェックポイントの保存先ディレクトリ
        logger: ロガーインスタンス
    """

    def __init__(self, work_dir: Path, logger: logging.Logger) -> None:
        """CheckpointStoreを初期化."""
        self.work_dir = work_dir
        self.logger = logger

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        best_accuracy: float,
    ) -> Path:
        """チェックポイントの保存.

        Args:
            filename: 保存ファイル名
            epoch: 現在のエポック数
            model: モデル
            optimizer: オプティマイザ
            scheduler: スケジューラ
            best_accuracy: ベスト精度

        Returns:
            保存先のパス
        """
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "best_accuracy": best_accuracy,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        checkpoint_path = self.work_dir / filename
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def save_best_model(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        best_accuracy: float,
    ) -> None:
        """ベストモデルの保存(エポック数付き, 上書き).

        既存のベストモデルファイルを削除してから新しいベストモデルを保存する.

        Args:
            epoch: 現在のエポック数
            model: モデル
            optimizer: オプティマイザ
            scheduler: スケジューラ
            best_accuracy: ベスト精度
        """
        # 既存のベストモデルファイルを削除
        for existing_file in self.work_dir.glob("best_epoch*.pth"):
            existing_file.unlink()
            self.logger.info(f"既存のベストモデルを削除: {existing_file}")

        # 新しいベストモデルを保存
        best_filename = f"best_epoch{epoch}.pth"
        checkpoint_path = self.save_checkpoint(
            best_filename, epoch, model, optimizer, scheduler, best_accuracy
        )
        self.logger.info(f"ベストモデルを保存: {checkpoint_path}")

    def save_last_model(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        best_accuracy: float,
    ) -> None:
        """ラストモデルの保存(上書き).

        Args:
            epoch: 現在のエポック数
            model: モデル
            optimizer: オプティマイザ
            scheduler: スケジューラ
            best_accuracy: ベスト精度
        """
        self.save_checkpoint(
            "last_model.pth", epoch, model, optimizer, scheduler, best_accuracy
        )

    def load_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    ) -> Dict[str, Any]:
        """チェックポイントの読み込み.

        Args:
            filename: 読み込むファイル名
            model: モデル (state_dict がロードされる)
            device: デバイス
            optimizer: オプティマイザ (state_dict がロードされる)
            scheduler: スケジューラ (state_dict がロードされる)

        Returns:
            復元された状態情報 {"epoch": int, "best_accuracy": float}

        Raises:
            FileNotFoundError: チェックポイントファイルが存在しない場合
        """
        checkpoint_path = self.work_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"チェックポイントが見つかりません: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"チェックポイントを読み込み: {checkpoint_path}")

        return {
            "epoch": checkpoint["epoch"],
            "best_accuracy": checkpoint["best_accuracy"],
        }
