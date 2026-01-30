"""
pochitrain.pochi_trainer: Pochiトレーナー.

複雑なレジストリシステムを使わない、直接的なトレーナー
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models.pochi_models import create_model
from .training.checkpoint_store import CheckpointStore
from .training.early_stopping import EarlyStopping
from .training.metrics_tracker import MetricsTracker
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
        create_workspace (bool, optional): ワークスペースを作成するか（推論時はFalse）
    """

    # 型アノテーション（推論時にワークスペース作成をスキップするため、current_workspaceがNoneになる可能性がある）
    current_workspace: Optional[
        Path
    ]  # 推論モードではNone、訓練モードではPathオブジェクト
    work_dir: Path  # ワークスペース作成の有無に関わらず常にPathオブジェクト

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str,
        pretrained: bool = True,
        work_dir: str = "work_dirs",
        create_workspace: bool = True,
        cudnn_benchmark: bool = False,
    ):
        """PochiTrainerを初期化."""
        # モデル設定の保存
        self.model_name = model_name
        self.num_classes = num_classes

        # デバイスの設定（バリデーション済みの設定を使用）
        self.device = torch.device(device)

        # cuDNN自動チューニングの設定
        if self.device.type == "cuda" and cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            self.cudnn_benchmark = True
        else:
            self.cudnn_benchmark = False

        # ワークスペースマネージャーの初期化
        self.workspace_manager = PochiWorkspaceManager(work_dir)
        if create_workspace:
            self.current_workspace = self.workspace_manager.create_workspace()
            self.work_dir = self.workspace_manager.get_models_dir()
        else:
            self.current_workspace = None
            self.work_dir = Path(work_dir)

        # ロガーの設定
        self.logger = self._setup_logger()
        self.logger.info(f"使用デバイス: {self.device}")
        if self.cudnn_benchmark:
            self.logger.info("cuDNN自動チューニング: 有効")
        self.logger.info(f"ワークスペース: {self.current_workspace}")
        self.logger.info(f"モデル保存先: {self.work_dir}")

        # チェックポイントストアの初期化
        self.checkpoint_store = CheckpointStore(self.work_dir, self.logger)

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

        # 混同行列計算のためのクラス数（後で設定）
        self.num_classes_for_cm: Optional[int] = None

        # 最適化器・損失関数は後で設定
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.criterion: Optional[nn.Module] = None

        # メトリクス・勾配の設定（CLIから設定される）
        self.enable_metrics_export = True  # デフォルトで有効
        self.enable_gradient_tracking = False  # デフォルトでOFF（計算コスト考慮）
        self.gradient_tracking_config: Dict[str, Any] = {
            "record_frequency": 1,  # 記録頻度（1 = 毎エポック）
        }

        # Early Stopping（訓練時のみ初期化）
        self.early_stopping: Optional[Any] = None  # EarlyStopping
        self.early_stopping_config: Optional[Dict[str, Any]] = None

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
        enable_layer_wise_lr: bool = False,
        layer_wise_lr_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        訓練の設定.

        Args:
            learning_rate (float): 学習率
            optimizer_name (str): 最適化器名 ('Adam', 'AdamW', 'SGD')
            scheduler_name (str, optional): スケジューラー名
                ('StepLR', 'MultiStepLR', 'CosineAnnealingLR',
                'ExponentialLR', 'LinearLR')
            scheduler_params (dict, optional): スケジューラーのパラメータ
            class_weights (List[float], optional): クラス毎の損失重み
            num_classes (int, optional): クラス数（重みのバリデーション用）
            enable_layer_wise_lr (bool): 層別学習率を有効にするか
            layer_wise_lr_config (Dict[str, Any], optional): 層別学習率の設定
        """
        # 損失関数の設定（クラス重み対応）
        if class_weights is not None:
            # クラス数の整合性チェック
            if num_classes is not None and len(class_weights) != num_classes:
                raise ValueError(
                    f"クラス重みの長さ({len(class_weights)})がクラス数({num_classes})と一致しません"
                )
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            self.logger.info(f"クラス重みを設定: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 層別学習率の設定を保存
        self.enable_layer_wise_lr = enable_layer_wise_lr
        self.layer_wise_lr_config = layer_wise_lr_config or {}
        self.layer_wise_lr_graph_config = self.layer_wise_lr_config.get(
            "graph_config", {}
        )
        self.base_learning_rate = learning_rate  # 基本学習率を保存

        # 最適化器の設定（層別学習率対応）
        if self.enable_layer_wise_lr:
            # 層別学習率が有効な場合、パラメータグループを作成
            param_groups = self._build_layer_wise_param_groups(learning_rate)
            self._log_layer_wise_lr(param_groups)
        else:
            # 通常の場合、全パラメータに同じ学習率を適用
            param_groups = [{"params": self.model.parameters(), "lr": learning_rate}]

        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(param_groups)
        elif optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                param_groups,
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
            elif scheduler_name == "ExponentialLR":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, **scheduler_params
                )
            elif scheduler_name == "LinearLR":
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer, **scheduler_params
                )
            else:
                raise ValueError(
                    f"サポートされていないスケジューラー: {scheduler_name}"
                )

        self.logger.info(f"最適化器: {optimizer_name} (学習率: {learning_rate})")
        if scheduler_name:
            self.logger.info(f"スケジューラー: {scheduler_name}")

        # 混同行列計算のためのクラス数を設定
        if num_classes:
            self.num_classes_for_cm = num_classes
            self.logger.info(f"混同行列計算を有効化しました (クラス数: {num_classes})")

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

        # 混同行列計算のためのリスト
        all_predicted = []
        all_targets = []

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

                # 混同行列用にデータを保存
                all_predicted.append(predicted)
                all_targets.append(target)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        # 混同行列の計算と出力（Fortranエラー回避のため純粋PyTorch実装）
        if self.num_classes_for_cm is not None and all_predicted and all_targets:
            # バッチごとに収集した予測値と正解値を連結
            # 各バッチのテンソルを1つの大きなテンソルにまとめる
            all_predicted_tensor = torch.cat(all_predicted, dim=0)
            all_targets_tensor = torch.cat(all_targets, dim=0)

            # 混同行列を計算（sklearn/torchmetrics不使用）
            # 基本的なPyTorchテンソル操作のみを使用してFortranエラーを回避
            cm = self._compute_confusion_matrix_pytorch(
                all_predicted_tensor, all_targets_tensor, self.num_classes_for_cm
            )

            # 簡易形式でログファイルに追記出力
            # epoch番号と行列データを1行で記録し、後続分析に使用可能
            self._log_confusion_matrix(cm)

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def _compute_confusion_matrix_pytorch(
        self, predicted: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """
        純粹なPyTorchテンソル操作による混同行列計算.

        sklearn.metrics.confusion_matrixやtorchmetricsを使用せず、
        基本的なPyTorchテンソル操作のみで混同行列を計算します。
        これにより、Ctrl+C割り込み時のFortranランタイムエラーを回避できます。

        Args:
            predicted (torch.Tensor): 予測ラベル（各要素は0からnum_classes-1の整数）
            targets (torch.Tensor): 正解ラベル（各要素は0からnum_classes-1の整数）
            num_classes (int): クラス数

        Returns:
            torch.Tensor: 混同行列（[num_classes, num_classes]の2次元テンソル）
                         行が正解ラベル、列が予測ラベルに対応
                         confusion_matrix[i, j] = 正解がi、予測がjの個数

        Note:
            - メモリ効率のため、要素数の多い場合は時間がかかる可能性があります
            - GPU/CPUデバイス間の一貫性を保つため、計算前にデバイスを統一します
        """
        # デバイスを統一（GPU/CPUの混在を防ぐ）
        predicted = predicted.to(self.device)
        targets = targets.to(self.device)

        # 混同行列を初期化（行=正解、列=予測）
        # dtype=int64で十分な範囲の整数カウントに対応
        confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.int64, device=self.device
        )

        # 各サンプルの（正解, 予測）ペアをカウント
        # view(-1)でテンソルを1次元に平坦化してからペアワイズ処理
        for t, p in zip(targets.view(-1), predicted.view(-1)):
            # .long()でint64型に変換してインデックスとして使用
            confusion_matrix[t.long(), p.long()] += 1

        return confusion_matrix

    def _log_confusion_matrix(self, confusion_matrix: torch.Tensor) -> None:
        """
        混同行列をログファイルに出力.

        各エポックの検証時に混同行列を.logファイルに追記します。
        出力形式は簡易的で、後続の分析やレポート生成に使用可能です。

        Args:
            confusion_matrix (torch.Tensor): PyTorchテンソル形式の混同行列
                                           [num_classes, num_classes]の2次元テンソル

        Output Format:
            epoch1
             [row1]
             [row2]
             [row3]
            epoch2
             [row1]
             [row2]
             [row3]

        Example (3クラス分類):
            epoch1
             [10, 0, 0]
             [0, 8, 2]
             [1, 0, 9]
            epoch2
             [9, 1, 0]
             [0, 9, 1]
             [0, 1, 9]

        Note:
            - ワークスペースが存在しない場合は何もしません
            - ファイルは追記モードで開くため、訓練継続時も過去のデータが保持されます
            - UTF-8エンコーディングで保存されます
        """
        # ワークスペースが未初期化の場合はスキップ
        if self.current_workspace is None:
            return

        # ログファイルパスの構築
        log_file = Path(self.current_workspace) / "confusion_matrix.log"

        # PyTorchテンソルをNumPy配列に変換し、int型にキャスト
        # CPUに移動してからnumpy()でNumPy配列化
        cm_numpy = confusion_matrix.cpu().numpy().astype(int)

        # ファイルに追記モードで書き込み
        # エポック番号を単独行で出力し、その後に混同行列の各行を縦に並べる
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"epoch{self.epoch}\n")  # エポック番号を単独行で出力
            for row in cm_numpy:
                f.write(f" {row.tolist()}\n")  # 各行を改行で区切って出力

    def save_checkpoint(
        self, filename: str = "checkpoint.pth", is_best: bool = False
    ) -> None:
        """チェックポイントの保存."""
        self.checkpoint_store.save_checkpoint(
            filename=filename,
            epoch=self.epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_accuracy=self.best_accuracy,
        )

    def save_best_model(self, epoch: int) -> None:
        """
        ベストモデルの保存（エポック数付き、上書き）.

        Args:
            epoch (int): 現在のエポック数
        """
        self.checkpoint_store.save_best_model(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_accuracy=self.best_accuracy,
        )

    def save_last_model(self) -> None:
        """ラストモデルの保存（上書き）."""
        self.checkpoint_store.save_last_model(
            epoch=self.epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_accuracy=self.best_accuracy,
        )

    def load_checkpoint(self, filename: str = "checkpoint.pth") -> None:
        """チェックポイントの読み込み."""
        result = self.checkpoint_store.load_checkpoint(
            filename=filename,
            model=self.model,
            device=self.device,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.epoch = result["epoch"]
        self.best_accuracy = result["best_accuracy"]

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]] = None,
        epochs: int = 10,
        stop_flag_callback: Optional[Any] = None,
    ) -> None:
        """
        訓練の実行.

        Args:
            train_loader (DataLoader): 訓練データローダー
            val_loader (DataLoader, optional): 検証データローダー
            epochs (int): エポック数
            stop_flag_callback (callable, optional): 停止フラグをチェックするコールバック関数
        """
        self.logger.info(f"訓練を開始 - エポック数: {epochs}")

        # MetricsTrackerの初期化（ワークスペースがある場合のみ）
        tracker = None
        if self.current_workspace is not None:
            tracker = MetricsTracker(
                logger=self.logger,
                visualization_dir=self.workspace_manager.get_visualization_dir(),
                enable_metrics_export=self.enable_metrics_export,
                enable_gradient_tracking=self.enable_gradient_tracking,
                gradient_tracking_config=self.gradient_tracking_config,
                layer_wise_lr_graph_config=self.layer_wise_lr_graph_config,
            )
            tracker.initialize()

        # Early Stoppingの初期化
        if self.early_stopping_config is not None:
            es_config = self.early_stopping_config
            if es_config.get("enabled", False):
                self.early_stopping = EarlyStopping(
                    patience=es_config.get("patience", 10),
                    min_delta=es_config.get("min_delta", 0.0),
                    monitor=es_config.get("monitor", "val_accuracy"),
                    logger=self.logger,
                )
                self.logger.info(
                    f"Early Stopping: 有効 "
                    f"(patience={self.early_stopping.patience}, "
                    f"min_delta={self.early_stopping.min_delta}, "
                    f"monitor={self.early_stopping.monitor})"
                )

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            # 停止フラグのチェック（エポック開始前）
            if stop_flag_callback and stop_flag_callback():
                self.logger.warning(
                    f"安全停止が要求されました。エポック {epoch-1} で訓練を終了します。"
                )
                self.save_last_model()  # 現在の状態を保存
                break

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

                # Early Stopping判定
                if self.early_stopping is not None:
                    monitor = self.early_stopping.monitor
                    monitor_value = val_metrics.get(monitor)
                    if monitor_value is not None:
                        if self.early_stopping.step(monitor_value, epoch):
                            self.save_last_model()
                            break

            # ラストモデルの保存（毎エポック上書き）
            self.save_last_model()

            # メトリクスと勾配の記録
            if tracker is not None:
                layer_wise_rates = {}
                if self.is_layer_wise_lr_enabled():
                    layer_rates = self.get_layer_wise_learning_rates()
                    for layer_name, lr in layer_rates.items():
                        layer_wise_rates[f"lr_{layer_name}"] = lr

                tracker.record_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    model=self.model,
                    learning_rate=self._get_base_learning_rate(),
                    layer_wise_lr_enabled=self.is_layer_wise_lr_enabled(),
                    layer_wise_rates=layer_wise_rates,
                )

            # 停止フラグのチェック（エポック完了後）
            if stop_flag_callback and stop_flag_callback():
                self.logger.warning(
                    f"安全停止が要求されました。エポック {epoch} で訓練を終了します。"
                )
                break

        # Early Stoppingによる停止の場合、その旨をログ出力
        if self.early_stopping is not None and self.early_stopping.should_stop:
            self.logger.info(
                f"Early Stoppingにより訓練を停止しました "
                f"(最終エポック: {self.epoch}, "
                f"ベスト{self.early_stopping.monitor}: "
                f"{self.early_stopping.best_value:.4f}, "
                f"ベストエポック: {self.early_stopping.best_epoch})"
            )
        else:
            self.logger.info("訓練が完了しました")
        if val_loader:
            self.logger.info(f"最高精度: {self.best_accuracy:.2f}%")

        # 訓練完了後のエクスポート処理
        if tracker is not None:
            csv_path, graph_paths = tracker.finalize()
            if csv_path:
                self.logger.info(f"メトリクスCSVを出力: {csv_path}")
            if graph_paths:
                for graph_path in graph_paths:
                    self.logger.info(f"メトリクスグラフを出力: {graph_path}")

            # サマリー情報の表示
            summary = tracker.get_summary()
            if summary:
                self.logger.info("=== 訓練サマリー ===")
                self.logger.info(f"総エポック数: {summary['total_epochs']}")
                self.logger.info(f"最終訓練損失: {summary['final_train_loss']:.4f}")
                self.logger.info(
                    f"最終訓練精度: {summary['final_train_accuracy']:.2f}%"
                )
                if "best_val_accuracy" in summary:
                    self.logger.info(
                        f"最高検証精度: {summary['best_val_accuracy']:.2f}% "
                        f"(エポック {summary['best_val_accuracy_epoch']})"
                    )

    def predict(
        self,
        data_loader: DataLoader[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        予測の実行.

        Args:
            data_loader (DataLoader): 予測データローダー

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (予測値, 確信度)
        """
        import time

        self.model.eval()
        predictions: List[Any] = []
        confidences: List[Any] = []
        total_samples = 0
        warmup_samples = 0
        inference_time_ms = 0.0
        is_first_batch = True

        use_cuda = self.device.type == "cuda"

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                batch_size = data.size(0)

                if is_first_batch:
                    # ウォームアップ: 最初のバッチは計測対象外
                    # CUDAカーネルのコンパイルやcuDNNアルゴリズム選択を事前に行う
                    output = self.model(data)
                    if use_cuda:
                        torch.cuda.synchronize()
                    warmup_samples = batch_size
                    is_first_batch = False
                else:
                    # 推論時間計測（モデル推論部分のみ）
                    if use_cuda:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        output = self.model(data)
                        end_event.record()
                        torch.cuda.synchronize()
                        inference_time_ms += start_event.elapsed_time(end_event)
                    else:
                        start_time = time.perf_counter()
                        output = self.model(data)
                        inference_time_ms += (time.perf_counter() - start_time) * 1000

                    total_samples += batch_size

                # 後処理（計測対象外）
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)

                predictions.extend(predicted.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())

        # 1枚あたりの平均推論時間を表示（ウォームアップ分を除く）
        if total_samples > 0:
            avg_time_per_image = inference_time_ms / total_samples
            self.logger.info(
                f"平均推論時間: {avg_time_per_image:.2f} ms/image "
                f"(計測: {total_samples}枚, ウォームアップ除外: {warmup_samples}枚)"
            )

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

    def _get_layer_group(self, param_name: str) -> str:
        """
        パラメータ名から層グループ名を取得.

        Args:
            param_name (str): パラメータ名

        Returns:
            str: 層グループ名
        """
        # ResNetの構造に基づいて層を分類（順序重要：より具体的なものから先に判定）
        if "layer1" in param_name:
            return "layer1"
        elif "layer2" in param_name:
            return "layer2"
        elif "layer3" in param_name:
            return "layer3"
        elif "layer4" in param_name:
            return "layer4"
        elif "conv1" in param_name:
            return "conv1"
        elif "bn1" in param_name:
            return "bn1"
        elif "fc" in param_name:
            return "fc"
        else:
            # 未知の層は "other" グループに分類
            return "other"

    def _build_layer_wise_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """
        層別学習率のパラメータグループを構築.

        Args:
            base_lr (float): 基本学習率

        Returns:
            List[Dict[str, Any]]: パラメータグループのリスト
        """
        layer_rates = self.layer_wise_lr_config.get("layer_rates", {})

        # 層ごとにパラメータを分類
        layer_params: Dict[str, List] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_group = self._get_layer_group(name)
                if layer_group not in layer_params:
                    layer_params[layer_group] = []
                layer_params[layer_group].append(param)

        # パラメータグループを作成
        param_groups = []
        for layer_name, params in layer_params.items():
            # 設定された学習率を使用、なければ基本学習率を使用
            lr = layer_rates.get(layer_name, base_lr)
            param_groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "layer_name": layer_name,  # デバッグ用
                }
            )

        return param_groups

    def _log_layer_wise_lr(self, param_groups: List[Dict[str, Any]]) -> None:
        """
        層別学習率の設定をログ出力.

        Args:
            param_groups (List[Dict[str, Any]]): パラメータグループのリスト
        """
        self.logger.info("=== 層別学習率設定 ===")
        for group in param_groups:
            layer_name = group.get("layer_name", "unknown")
            lr = group["lr"]
            param_count = sum(p.numel() for p in group["params"])
            self.logger.info(f"  {layer_name}: lr={lr:.6f}, params={param_count:,}")
        self.logger.info("=====================")

    def _get_base_learning_rate(self) -> float:
        """
        現在の基本学習率を取得.

        Returns:
            float: 基本学習率（層別学習率有効時は設定値、無効時は実際の学習率）
        """
        if self.enable_layer_wise_lr:
            # 層別学習率有効時は設定ファイルの固定値を返す
            return self.base_learning_rate
        else:
            # 通常時はスケジューラーによる動的な値を返す
            if self.optimizer is not None:
                lr: float = self.optimizer.param_groups[0]["lr"]
                return lr
            return 0.0

    def is_layer_wise_lr_enabled(self) -> bool:
        """
        層別学習率が有効かどうかを判定.

        Returns:
            bool: 層別学習率が有効な場合True
        """
        return self.enable_layer_wise_lr

    def get_layer_wise_learning_rates(self) -> Dict[str, float]:
        """
        各層の現在の学習率を取得.

        Returns:
            Dict[str, float]: 層名をキー、学習率を値とする辞書
        """
        layer_rates = {}
        if self.enable_layer_wise_lr and self.optimizer:
            for group in self.optimizer.param_groups:
                layer_name = group.get("layer_name", "unknown")
                layer_rates[layer_name] = group["lr"]
        return layer_rates
