#!/usr/bin/env python3
"""
pochitrain クイックスタート実行スクリプト.

設定ファイルを読み込み、シンプルな訓練パイプラインを実行
"""

import importlib.util
from pathlib import Path

from pochitrain import LoggerManager, PochiTrainer, create_data_loaders


def load_config(config_path: str) -> dict:
    """
    設定ファイルを読み込む.

    Args:
        config_path (str): 設定ファイルのパス

    Returns:
        dict: 設定辞書
    """
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    # モジュールとして読み込み
    spec = importlib.util.spec_from_file_location("config", config_path_obj)
    if spec is None:
        raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"設定ファイルのローダーが見つかりません: {config_path}")

    spec.loader.exec_module(config_module)

    # 設定辞書を構築
    config = {}
    for key in dir(config_module):
        if not key.startswith("_"):
            value = getattr(config_module, key)
            # 関数やメソッドは除外するが、transformsオブジェクトは含める
            if not callable(value) or hasattr(value, "transforms"):
                config[key] = value

    return config


def main():
    """メイン関数."""
    # ロガーの設定
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger("pochitrain")

    logger.info("=== pochitrain クイックスタート ===")

    # 設定ファイルの読み込み
    config_path = "configs/pochi_config.py"
    config_path_obj = Path(config_path)

    try:
        config = load_config(config_path)
        logger.info(f"設定ファイルを読み込みました: {config_path}")
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        logger.error("configs/pochi_config.py を作成してください。")
        return

    # データローダーの作成
    logger.info("データローダーを作成しています...")
    try:
        train_loader, val_loader, classes = create_data_loaders(
            train_root=config["train_data_root"],
            val_root=config.get("val_data_root"),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_transform=config.get("train_transform"),
            val_transform=config.get("val_transform"),
        )

        logger.info(f"クラス数: {len(classes)}")
        logger.info(f"クラス名: {classes}")
        logger.info(f"訓練バッチ数: {len(train_loader)}")
        if val_loader:
            logger.info(f"検証バッチ数: {len(val_loader)}")

        # 設定のクラス数を更新
        config["num_classes"] = len(classes)

    except Exception as e:
        logger.error(f"データローダーの作成に失敗しました: {e}")
        return

    # トレーナーの作成
    logger.info("トレーナーを作成しています...")
    trainer = PochiTrainer(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
        device=config.get("device"),
        work_dir=config["work_dir"],
    )

    # 訓練設定
    logger.info("訓練設定を行っています...")
    trainer.setup_training(
        learning_rate=config["learning_rate"],
        optimizer_name=config["optimizer"],
        scheduler_name=config.get("scheduler"),
        scheduler_params=config.get("scheduler_params"),
        class_weights=config.get("class_weights"),
        num_classes=len(classes),
    )

    # データセットパスの保存
    logger.info("データセットパスを保存しています...")
    trainer.save_dataset_paths(train_loader, val_loader)

    # 設定ファイルの保存
    logger.info("設定ファイルを保存しています...")
    saved_config_path = trainer.save_training_config(config_path_obj)
    logger.info(f"設定ファイルを保存しました: {saved_config_path}")

    # 訓練実行
    logger.info("訓練を開始します...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
    )

    logger.info("訓練が完了しました！")
    logger.info(f"結果は {config['work_dir']} に保存されています。")


if __name__ == "__main__":
    main()
