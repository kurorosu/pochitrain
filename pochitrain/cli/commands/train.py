"""train サブコマンドの実装."""

import argparse
import signal
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

import pochitrain.cli.cli_commons as cli_commons
from pochitrain import (
    PochiConfig,
    PochiTrainer,
    create_data_loaders,
)
from pochitrain.cli.cli_commons import (
    create_signal_handler,
    setup_logging,
)
from pochitrain.utils import ConfigLoader


def train_command(args: argparse.Namespace) -> None:
    """訓練サブコマンドの実行."""
    signal.signal(signal.SIGINT, create_signal_handler(debug=args.debug))

    logger = setup_logging(debug=args.debug)
    logger.info("=== pochitrain 訓練モード ===")
    logger.info(
        "安全終了: 訓練中にCtrl+Cを押すと、現在のエポック完了後に安全に終了します。"
    )

    try:
        config = ConfigLoader.load_config(args.config)
        logger.info(f"設定ファイルを読み込みました: {args.config}")
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {args.config}")
        logger.error("configs/pochi_train_config.py を作成してください。")
        return

    try:
        pochi_config = PochiConfig.from_dict(config)
    except ValidationError as e:
        logger.error(f"設定にエラーがあります:\n{e}")
        return

    if not Path(pochi_config.train_data_root).exists():
        logger.error(f"訓練データパスが存在しません: {pochi_config.train_data_root}")
        return
    if (
        pochi_config.val_data_root is not None
        and not Path(pochi_config.val_data_root).exists()
    ):
        logger.error(f"検証データパスが存在しません: {pochi_config.val_data_root}")
        return

    logger.debug("=== 設定確認 ===")
    logger.debug(f"モデル: {pochi_config.model_name}")
    logger.debug(f"デバイス: {pochi_config.device}")
    logger.debug(f"学習率: {pochi_config.learning_rate}")
    logger.debug(f"オプティマイザー: {pochi_config.optimizer}")

    scheduler_name = pochi_config.scheduler
    if scheduler_name is None:
        logger.info("スケジューラー: なし（固定学習率）")
    else:
        logger.debug(f"スケジューラー: {scheduler_name}")
        scheduler_params = pochi_config.scheduler_params
        logger.debug(f"スケジューラーパラメータ: {scheduler_params}")

    class_weights = pochi_config.class_weights
    if class_weights is None:
        logger.debug("クラス重み: なし（均等扱い）")
    else:
        logger.debug(f"クラス重み: {class_weights}")

    enable_layer_wise_lr = pochi_config.enable_layer_wise_lr
    if enable_layer_wise_lr:
        logger.debug("層別学習率: 有効")
        layer_wise_lr_config = cast(
            dict[str, Any], pochi_config.layer_wise_lr_config.model_dump()
        )
        layer_rates = layer_wise_lr_config.get("layer_rates", {})
        logger.debug(f"層別学習率設定: {layer_rates}")
    else:
        logger.debug("層別学習率: 無効")

    logger.debug("データローダーを作成しています...")
    try:
        if pochi_config.val_data_root is None:
            logger.error("val_data_root が設定されていません。")
            return
        train_loader, val_loader, classes = create_data_loaders(
            train_root=pochi_config.train_data_root,
            val_root=pochi_config.val_data_root,
            batch_size=pochi_config.batch_size,
            num_workers=pochi_config.num_workers,
            pin_memory=pochi_config.train_pin_memory,
            train_transform=pochi_config.train_transform,
            val_transform=pochi_config.val_transform,
        )

        logger.debug(f"クラス数: {len(classes)}")
        logger.debug(f"クラス名: {classes}")
        logger.debug(f"訓練バッチ数: {len(train_loader)}")
        logger.debug(f"検証バッチ数: {len(val_loader)}")

        config["num_classes"] = len(classes)
        pochi_config.num_classes = len(classes)

    except Exception as e:
        logger.error(f"データローダーの作成に失敗しました: {e}")
        return

    logger.debug("トレーナーを作成しています...")
    trainer = PochiTrainer(
        model_name=pochi_config.model_name,
        num_classes=pochi_config.num_classes,
        device=pochi_config.device,
        pretrained=pochi_config.pretrained,
        work_dir=pochi_config.work_dir,
        cudnn_benchmark=pochi_config.cudnn_benchmark,
    )

    logger.debug("訓練設定を行っています...")
    trainer.setup_training(
        learning_rate=pochi_config.learning_rate,
        optimizer_name=pochi_config.optimizer,
        scheduler_name=pochi_config.scheduler,
        scheduler_params=pochi_config.scheduler_params,
        class_weights=pochi_config.class_weights,
        num_classes=len(classes),
        enable_layer_wise_lr=pochi_config.enable_layer_wise_lr,
        layer_wise_lr_config=cast(
            dict[str, Any], pochi_config.layer_wise_lr_config.model_dump()
        ),
        early_stopping_config=(
            cast(dict[str, Any], pochi_config.early_stopping.model_dump())
            if pochi_config.early_stopping is not None
            else None
        ),
    )

    logger.debug("データセットパスを保存しています...")
    train_paths = []
    if hasattr(train_loader.dataset, "get_file_paths"):
        train_paths = train_loader.dataset.get_file_paths()
    else:
        logger.warning("訓練データセットにget_file_pathsメソッドがありません")
    val_paths = None
    if hasattr(val_loader.dataset, "get_file_paths"):
        val_paths = val_loader.dataset.get_file_paths()
    else:
        logger.warning("検証データセットにget_file_pathsメソッドがありません")
    train_file, val_file = trainer.workspace_manager.save_dataset_paths(
        train_paths, val_paths
    )
    logger.debug(f"訓練データパスを保存: {train_file}")
    if val_file is not None:
        logger.debug(f"検証データパスを保存: {val_file}")

    logger.debug("設定ファイルを保存しています...")
    config_path_obj = Path(args.config)
    saved_config_path = trainer.workspace_manager.save_config(config_path_obj)
    logger.debug(f"設定ファイルを保存しました: {saved_config_path}")

    trainer.enable_metrics_export = pochi_config.enable_metrics_export
    if trainer.enable_metrics_export:
        logger.debug("訓練メトリクスのCSV出力とグラフ生成が有効です")

    early_stopping_config = (
        cast(dict[str, Any], pochi_config.early_stopping.model_dump())
        if pochi_config.early_stopping is not None
        else None
    )
    if not (early_stopping_config and early_stopping_config.get("enabled", False)):
        logger.debug("Early Stopping: 無効")

    trainer.enable_gradient_tracking = pochi_config.enable_gradient_tracking
    if trainer.enable_gradient_tracking:
        logger.debug("勾配トレース機能が有効です")

    trainer.enable_tensorboard = pochi_config.enable_tensorboard
    if trainer.enable_tensorboard:
        logger.debug("TensorBoard 記録機能が有効です")
        gradient_config = cast(
            dict[str, Any], pochi_config.gradient_tracking_config.model_dump()
        )
        trainer.gradient_tracking_config.update(gradient_config)

    logger.info("訓練を開始します...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=pochi_config.epochs,
        stop_flag_callback=lambda: cli_commons.training_interrupted,
    )

    logger.info("訓練が完了しました！")
    logger.info(f"結果は {pochi_config.work_dir} に保存されています。")
