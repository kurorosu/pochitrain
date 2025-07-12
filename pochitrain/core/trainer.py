"""
pochitrain.core.trainer: トレーナーモジュール

訓練ループと最適化を担当するトレーナーモジュール
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import Config
from .registry import MODELS, DATASETS, OPTIMIZERS, SCHEDULERS, LOSSES
from .evaluator import Evaluator


class Trainer:
    """
    トレーナークラス

    モデルの訓練を管理する

    Args:
        config (Config): 設定オブジェクト
        work_dir (str, optional): 作業ディレクトリ
        logger (logging.Logger, optional): ロガー

    Examples:
        >>> config = Config.from_file('configs/resnet/resnet18_cifar10.py')
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """

    def __init__(self,
                 config: Config,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.work_dir = work_dir or 'work_dirs'
        self.logger = logger or self._setup_logger()

        # デバイスの設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用デバイス: {self.device}")

        # 訓練状態の管理
        self.epoch = 0
        self.step = 0
        self.best_accuracy = 0.0

        # モデル・データセット・最適化器の準備
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # 評価器の準備
        self.evaluator = Evaluator(logger=self.logger)

        # 作業ディレクトリの作成
        self._create_work_dir()

        # 設定の妥当性を確認
        self._validate_config()

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('pochitrain')
        logger.setLevel(logging.INFO)

        # ハンドラーが既に存在する場合は追加しない
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_work_dir(self) -> None:
        """作業ディレクトリの作成"""
        work_dir = Path(self.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"作業ディレクトリ: {work_dir}")

    def _validate_config(self) -> None:
        """設定の妥当性確認"""
        try:
            self.config.validate()
        except ValueError as e:
            self.logger.error(f"設定エラー: {e}")
            raise

    def build_model(self) -> None:
        """モデルの構築"""
        self.logger.info("モデルを構築しています...")
        self.model = MODELS.build(self.config.model)
        self.model.to(self.device)

        # モデルの概要を表示
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"モデル: {self.config.model['type']}")
        self.logger.info(f"総パラメータ数: {total_params:,}")
        self.logger.info(f"訓練可能パラメータ数: {trainable_params:,}")

    def build_dataset(self) -> None:
        """データセットの構築"""
        self.logger.info("データセットを構築しています...")

        # 訓練データセット
        train_dataset = DATASETS.build(self.config.dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # 検証データセット（あれば）
        if hasattr(self.config, 'val_dataset'):
            val_dataset = DATASETS.build(self.config.val_dataset)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.training.get('num_workers', 4),
                pin_memory=True if self.device.type == 'cuda' else False
            )

        self.logger.info(f"訓練データ数: {len(train_dataset)}")
        if self.val_loader:
            self.logger.info(f"検証データ数: {len(self.val_loader.dataset)}")

    def build_optimizer(self) -> None:
        """最適化器の構築"""
        self.logger.info("最適化器を構築しています...")

        optimizer_config = self.config.training.get('optimizer', 'Adam')
        if isinstance(optimizer_config, str):
            optimizer_config = {'type': optimizer_config}

        # 学習率の設定
        optimizer_config['lr'] = self.config.training.learning_rate

        # PyTorchの標準最適化器を使用
        if optimizer_config['type'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                **{k: v for k, v in optimizer_config.items() if k not in ['type', 'lr']}
            )
        elif optimizer_config['type'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                **{k: v for k, v in optimizer_config.items() if k not in ['type', 'lr']}
            )
        else:
            raise ValueError(f"サポートされていない最適化器: {optimizer_config['type']}")

        self.logger.info(
            f"最適化器: {optimizer_config['type']} (学習率: {optimizer_config['lr']})")

    def build_scheduler(self) -> None:
        """学習率スケジューラの構築"""
        scheduler_config = self.config.training.get('scheduler')
        if scheduler_config:
            self.logger.info("学習率スケジューラを構築しています...")

            if scheduler_config['type'] == 'StepLR':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 30),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_config['type'] == 'CosineAnnealingLR':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.training.epochs
                )

            self.logger.info(f"学習率スケジューラ: {scheduler_config['type']}")

    def build_criterion(self) -> None:
        """損失関数の構築"""
        self.logger.info("損失関数を構築しています...")

        criterion_config = self.config.training.get('criterion', 'CrossEntropyLoss')
        if isinstance(criterion_config, str):
            criterion_config = {'type': criterion_config}

        # PyTorchの標準損失関数を使用
        if criterion_config['type'] == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_config['type'] == 'MSELoss':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"サポートされていない損失関数: {criterion_config['type']}")

        self.logger.info(f"損失関数: {criterion_config['type']}")

    def prepare(self) -> None:
        """訓練の準備"""
        self.logger.info("訓練の準備を開始しています...")

        self.build_model()
        self.build_dataset()
        self.build_optimizer()
        self.build_scheduler()
        self.build_criterion()

        self.logger.info("訓練の準備が完了しました")

    def train_epoch(self) -> Dict[str, float]:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
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

            self.step += 1

            # ログ出力
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'エポック {self.epoch}, バッチ {batch_idx}/{len(self.train_loader)}, '
                    f'損失: {loss.item():.4f}, 精度: {100.0 * correct / total:.2f}%'
                )

        # エポックの統計情報
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self) -> Dict[str, float]:
        """検証"""
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config.to_dict()
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 最新のチェックポイントを保存
        checkpoint_path = Path(self.work_dir) / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # ベストモデルの場合は別途保存
        if is_best:
            best_path = Path(self.work_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"ベストモデルを保存しました: {best_path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """チェックポイントの読み込み"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_accuracy = checkpoint['best_accuracy']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"チェックポイントを読み込みました: {checkpoint_path}")

    def train(self) -> None:
        """訓練の実行"""
        self.logger.info("訓練を開始します...")

        # 準備
        self.prepare()

        # 訓練ループ
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch + 1

            self.logger.info(f"エポック {self.epoch}/{self.config.training.epochs} を開始")

            # 1エポック訓練
            train_metrics = self.train_epoch()

            # 検証
            val_metrics = self.validate()

            # 学習率スケジューラの更新
            if self.scheduler:
                self.scheduler.step()

            # ログ出力
            self.logger.info(
                f"エポック {self.epoch} 完了 - "
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
                    self.save_checkpoint(is_best=True)

            # チェックポイントの保存
            self.save_checkpoint()

        self.logger.info("訓練が完了しました")
        self.logger.info(f"最高精度: {self.best_accuracy:.2f}%")
