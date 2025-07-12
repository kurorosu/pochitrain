"""
pochitrain.pochi_trainer: Pochiトレーナー

複雑なレジストリシステムを使わない、直接的なトレーナー
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models.pochi_models import create_model


class PochiTrainer:
    """
    Pochiトレーナークラス

    Args:
        model_name (str): モデル名 ('resnet18', 'resnet34', 'resnet50')
        num_classes (int): 分類クラス数
        pretrained (bool): 事前学習済みモデルを使用するか
        device (str, optional): デバイス ('cuda' or 'cpu')
        work_dir (str, optional): 作業ディレクトリ
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained: bool = True,
                 device: Optional[str] = None,
                 work_dir: str = 'work_dirs'):

        # デバイスの設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # ロガーの設定
        self.logger = self._setup_logger()
        self.logger.info(f"使用デバイス: {self.device}")

        # モデルの作成
        self.model = create_model(model_name, num_classes, pretrained)
        self.model.to(self.device)

        # モデル情報の表示
        model_info = self.model.get_model_info()
        self.logger.info(f"モデル: {model_info['model_name']}")
        self.logger.info(f"クラス数: {model_info['num_classes']}")
        self.logger.info(f"総パラメータ数: {model_info['total_params']:,}")
        self.logger.info(f"訓練可能パラメータ数: {model_info['trainable_params']:,}")

        # 作業ディレクトリの作成
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 訓練状態の管理
        self.epoch = 0
        self.best_accuracy = 0.0

        # 最適化器・損失関数は後で設定
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('pochitrain_pochi')
        logger.setLevel(logging.INFO)

        # ハンドラーが既に存在する場合は追加しない
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def setup_training(self,
                       learning_rate: float = 0.001,
                       optimizer_name: str = 'Adam',
                       scheduler_name: Optional[str] = None,
                       scheduler_params: Optional[dict] = None) -> None:
        """
        訓練の設定

        Args:
            learning_rate (float): 学習率
            optimizer_name (str): 最適化器名 ('Adam', 'SGD')
            scheduler_name (str, optional): スケジューラー名 ('StepLR', 'CosineAnnealingLR')
            scheduler_params (dict, optional): スケジューラーのパラメータ
        """
        # 損失関数の設定
        self.criterion = nn.CrossEntropyLoss()

        # 最適化器の設定
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"サポートされていない最適化器: {optimizer_name}")

        # スケジューラーの設定
        if scheduler_name:
            if scheduler_name == 'StepLR':
                params = scheduler_params or {'step_size': 30, 'gamma': 0.1}
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **params)
            elif scheduler_name == 'CosineAnnealingLR':
                params = scheduler_params or {'T_max': 100}
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, **params)
            else:
                raise ValueError(f"サポートされていないスケジューラー: {scheduler_name}")

        self.logger.info(f"最適化器: {optimizer_name} (学習率: {learning_rate})")
        if scheduler_name:
            self.logger.info(f"スケジューラー: {scheduler_name}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 勾配をゼロにリセット
            self.optimizer.zero_grad()

            # 順伝播
            output = self.model(data)
            loss = self.criterion(output, target)

            # 逆伝播
            loss.backward()
            self.optimizer.step()

            # 統計情報の更新
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # ログ出力
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'エポック {self.epoch}, バッチ {batch_idx}/{len(train_loader)}, '
                    f'損失: {loss.item():.4f}, 精度: {100.0 * correct / total:.2f}%'
                )

        # エポックの統計情報
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False) -> None:
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # チェックポイントの保存
        checkpoint_path = self.work_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # ベストモデルの場合は別途保存
        if is_best:
            best_path = self.work_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"ベストモデルを保存: {best_path}")

    def load_checkpoint(self, filename: str = 'checkpoint.pth') -> None:
        """チェックポイントの読み込み"""
        checkpoint_path = self.work_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"チェックポイントを読み込み: {checkpoint_path}")

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 10,
              save_every: int = 5) -> None:
        """
        訓練の実行

        Args:
            train_loader (DataLoader): 訓練データローダー
            val_loader (DataLoader, optional): 検証データローダー
            epochs (int): エポック数
            save_every (int): チェックポイント保存間隔
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
            if self.scheduler:
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

                # ベストモデルの更新
                if val_metrics['val_accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['val_accuracy']
                    self.save_checkpoint('best_checkpoint.pth', is_best=True)

            # 定期的なチェックポイント保存
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        self.logger.info("訓練が完了しました")
        if val_loader:
            self.logger.info(f"最高精度: {self.best_accuracy:.2f}%")

    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        予測の実行

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
