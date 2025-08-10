"""
pochitrain.pochi_trainer: Pochiトレーナー.

複雑なレジストリシステムを使わない、直接的なトレーナー
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models.pochi_models import create_model
from .utils.directory_manager import PochiWorkspaceManager


class PochiTrainer:
    """
    Pochiトレーナークラス.

    Args:
        model_name (str): モデル名 ('resnet18', 'resnet34', 'resnet50')
        num_classes (int): 分類クラス数
        pretrained (bool): 事前学習済みモデルを使用するか
        device (str): デバイス ('cuda' or 'cpu') - 必須設定
        work_dir (str, optional): 作業ディレクトリ
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str,
        pretrained: bool = True,
        work_dir: str = "work_dirs",
    ):
        """PochiTrainerを初期化."""
        # デバイスの設定（バリデーション済みの設定を使用）
        self.device = torch.device(device)

        # ワークスペースマネージャーの初期化
        self.workspace_manager = PochiWorkspaceManager(work_dir)
        self.current_workspace = self.workspace_manager.create_workspace()
        self.work_dir = self.workspace_manager.get_models_dir()

        # ロガーの設定
        self.logger = self._setup_logger()
        self.logger.info(f"使用デバイス: {self.device}")
        self.logger.info(f"ワークスペース: {self.current_workspace}")
        self.logger.info(f"モデル保存先: {self.work_dir}")

        # モデルの作成
        self.model = create_model(model_name, num_classes, pretrained)
        self.model.to(self.device)

        # モデル情報の表示
        model_info = self.model.get_model_info()
        self.logger.info(f"モデル: {model_info['model_name']}")
        self.logger.info(f"クラス数: {model_info['num_classes']}")
        self.logger.info(f"総パラメータ数: {model_info['total_params']:,}")
        self.logger.info(f"訓練可能パラメータ数: {model_info['trainable_params']:,}")

        # 訓練状態の管理
        self.epoch = 0
        self.best_accuracy = 0.0

        # 最適化器・損失関数は後で設定
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定."""
        from pochitrain.logging import LoggerManager

        logger_manager = LoggerManager()
        return logger_manager.get_logger("pochitrain")

    def setup_training(
        self,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """
        訓練の設定.

        Args:
            learning_rate (float): 学習率
            optimizer_name (str): 最適化器名 ('Adam', 'SGD')
            scheduler_name (str, optional): スケジューラー名
                ('StepLR', 'MultiStepLR', 'CosineAnnealingLR')
            scheduler_params (dict, optional): スケジューラーのパラメータ
            class_weights (List[float], optional): クラス毎の損失重み
            num_classes (int, optional): クラス数（重みのバリデーション用）
        """
        # 損失関数の設定（クラス重み対応）
        if class_weights is not None:
            # クラス数の整合性チェック
            if num_classes is not None and len(class_weights) != num_classes:
                raise ValueError(
                    f"クラス重みの長さ({len(class_weights)})がクラス数({num_classes})と一致しません"
                )
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            self.logger.info(f"クラス重みを設定: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 最適化器の設定
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        else:
            raise ValueError(f"サポートされていない最適化器: {optimizer_name}")

        # スケジューラーの設定（バリデーション済みパラメータを使用）
        if scheduler_name:
            if scheduler_params is None:
                raise ValueError(
                    f"スケジューラー '{scheduler_name}' を使用する場合、"
                    f"scheduler_paramsが必須です。configs/pochi_config.pyで設定してください。"
                )

            if scheduler_name == "StepLR":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, **scheduler_params
                )
            elif scheduler_name == "MultiStepLR":
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, **scheduler_params
                )
            elif scheduler_name == "CosineAnnealingLR":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, **scheduler_params
                )
            else:
                raise ValueError(
                    f"サポートされていないスケジューラー: {scheduler_name}"
                )

        self.logger.info(f"最適化器: {optimizer_name} (学習率: {learning_rate})")
        if scheduler_name:
            self.logger.info(f"スケジューラー: {scheduler_name}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """1エポックの訓練."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 勾配をゼロにリセット
            if self.optimizer is not None:
                self.optimizer.zero_grad()

            # 順伝播
            output = self.model(data)
            if self.criterion is not None:
                loss = self.criterion(output, target)
            else:
                raise RuntimeError("criterion is not set")

            # 逆伝播
            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()

            # 統計情報の更新
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # ログ出力
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"エポック {self.epoch}, バッチ {batch_idx}/{len(train_loader)}, "
                    f"損失: {loss.item():.4f}, 精度: {100.0 * correct / total:.2f}%"
                )

        # エポックの統計情報
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """検証."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                if self.criterion is not None:
                    loss = self.criterion(output, target)
                else:
                    raise RuntimeError("criterion is not set")

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def save_checkpoint(
        self, filename: str = "checkpoint.pth", is_best: bool = False
    ) -> None:
        """チェックポイントの保存."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "best_accuracy": self.best_accuracy,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # チェックポイントの保存
        checkpoint_path = self.work_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def save_best_model(self, epoch: int) -> None:
        """
        ベストモデルの保存（エポック数付き、上書き）.

        Args:
            epoch (int): 現在のエポック数
        """
        # 既存のベストモデルファイルを削除
        for existing_file in self.work_dir.glob("best_epoch*.pth"):
            existing_file.unlink()
            self.logger.info(f"既存のベストモデルを削除: {existing_file}")

        # 新しいベストモデルを保存
        best_filename = f"best_epoch{epoch}.pth"
        self.save_checkpoint(best_filename)
        self.logger.info(f"ベストモデルを保存: {self.work_dir / best_filename}")

    def save_last_model(self) -> None:
        """ラストモデルの保存（上書き）."""
        last_filename = "last_model.pth"
        self.save_checkpoint(last_filename)

    def load_checkpoint(self, filename: str = "checkpoint.pth") -> None:
        """チェックポイントの読み込み."""
        checkpoint_path = self.work_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"チェックポイントが見つかりません: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if (
            self.optimizer is not None
            and checkpoint["optimizer_state_dict"] is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"チェックポイントを読み込み: {checkpoint_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> None:
        """
        訓練の実行.

        Args:
            train_loader (DataLoader): 訓練データローダー
            val_loader (DataLoader, optional): 検証データローダー
            epochs (int): エポック数
        """
        self.logger.info(f"訓練を開始 - エポック数: {epochs}")

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            self.logger.info(f"エポック {epoch}/{epochs} を開始")

            # 1エポック訓練
            train_metrics = self.train_epoch(train_loader)

            # 検証
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)

            # スケジューラーの更新
            if self.scheduler is not None:
                self.scheduler.step()

            # ログ出力
            self.logger.info(
                f"エポック {epoch} 完了 - "
                f"訓練損失: {train_metrics['loss']:.4f}, "
                f"訓練精度: {train_metrics['accuracy']:.2f}%"
            )

            if val_metrics:
                self.logger.info(
                    f"検証損失: {val_metrics['val_loss']:.4f}, "
                    f"検証精度: {val_metrics['val_accuracy']:.2f}%"
                )

                # ベストモデルの更新（精度が前回以上なら保存）
                if val_metrics["val_accuracy"] >= self.best_accuracy:
                    self.best_accuracy = val_metrics["val_accuracy"]
                    self.save_best_model(epoch)

            # ラストモデルの保存（毎エポック上書き）
            self.save_last_model()

        self.logger.info("訓練が完了しました")
        if val_loader:
            self.logger.info(f"最高精度: {self.best_accuracy:.2f}%")

    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        予測の実行.

        Args:
            data_loader (DataLoader): 予測データローダー

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (予測値, 確信度)
        """
        self.model.eval()
        predictions = []
        confidences = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)

                # ソフトマックスで確率に変換
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)

                predictions.extend(predicted.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())

        return torch.tensor(predictions), torch.tensor(confidences)

    def get_workspace_info(self) -> dict:
        """
        現在のワークスペース情報を取得.

        Returns:
            dict: ワークスペース情報
        """
        return self.workspace_manager.get_workspace_info()

    def save_training_config(self, config_path: Path) -> Path:
        """
        訓練に使用した設定ファイルを保存.

        Args:
            config_path (Path): 設定ファイルのパス

        Returns:
            Path: 保存されたファイルのパス
        """
        return self.workspace_manager.save_config(config_path)

    def save_image_list(self, image_paths: list) -> Path:
        """
        使用した画像リストを保存.

        Args:
            image_paths (list): 画像パスのリスト

        Returns:
            Path: 保存されたファイルのパス
        """
        return self.workspace_manager.save_image_list(image_paths)

    def save_dataset_paths(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Tuple[Path, Optional[Path]]:
        """
        訓練・検証データのファイルパスを保存.

        Args:
            train_loader (DataLoader): 訓練データローダー
            val_loader (DataLoader, optional): 検証データローダー

        Returns:
            Tuple[Path, Optional[Path]]: 保存されたファイルのパス (train.txt, val.txt)
        """
        # 訓練データのパスを取得
        train_paths = []
        if hasattr(train_loader.dataset, "get_file_paths"):
            train_paths = train_loader.dataset.get_file_paths()
        else:
            self.logger.warning("訓練データセットにget_file_pathsメソッドがありません")

        # 検証データのパスを取得
        val_paths: Optional[list] = None
        if val_loader is not None:
            if hasattr(val_loader.dataset, "get_file_paths"):
                val_paths = val_loader.dataset.get_file_paths()
            else:
                self.logger.warning(
                    "検証データセットにget_file_pathsメソッドがありません"
                )

        # パスを保存
        train_file_path, val_file_path = self.workspace_manager.save_dataset_paths(
            train_paths, val_paths
        )

        self.logger.info(f"訓練データパスを保存: {train_file_path}")
        if val_file_path is not None:
            self.logger.info(f"検証データパスを保存: {val_file_path}")

        return train_file_path, val_file_path
