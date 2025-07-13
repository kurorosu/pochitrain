#!/usr/bin/env python3
"""
pochitrain クイックスタート.

最速で検証まで到達するサンプルコード
"""

import importlib.util
from pathlib import Path

from pochitrain.pochi_dataset import create_data_loaders

# pochitrainモジュールのインポート
from pochitrain.pochi_trainer import PochiTrainer


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
            if not callable(value):
                config[key] = value

    return config


def main():
    """メイン関数."""
    print("=== pochitrain クイックスタート ===")

    # 設定ファイルの読み込み
    config_path = "configs/pochi_config.py"

    try:
        config = load_config(config_path)
        print(f"設定ファイルを読み込みました: {config_path}")
    except FileNotFoundError:
        print(f"設定ファイルが見つかりません: {config_path}")
        print("configs/pochi_config.py を作成してください。")
        return

    # データローダーの作成
    print("\nデータローダーを作成しています...")
    try:
        train_loader, val_loader, classes = create_data_loaders(
            train_root=config["train_data_root"],
            val_root=config.get("val_data_root"),
            batch_size=config["batch_size"],
            image_size=config["image_size"],
            num_workers=config["num_workers"],
        )

        print(f"クラス数: {len(classes)}")
        print(f"クラス名: {classes}")
        print(f"訓練バッチ数: {len(train_loader)}")
        if val_loader:
            print(f"検証バッチ数: {len(val_loader)}")

        # 設定のクラス数を更新
        config["num_classes"] = len(classes)

    except Exception as e:
        print(f"データローダーの作成に失敗しました: {e}")
        print("データディレクトリの構造を確認してください。")
        return

    # トレーナーの作成
    print("\nトレーナーを作成しています...")
    trainer = PochiTrainer(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
        device=config.get("device"),
        work_dir=config["work_dir"],
    )

    # 訓練設定
    print("\n訓練設定を行っています...")
    trainer.setup_training(
        learning_rate=config["learning_rate"],
        optimizer_name=config["optimizer"],
        scheduler_name=config.get("scheduler"),
        scheduler_params=config.get("scheduler_params"),
    )

    # データセットパスの保存
    print("\nデータセットパスを保存しています...")
    trainer.save_dataset_paths(train_loader, val_loader)

    # 訓練実行
    print("\n訓練を開始します...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
    )

    print("\n訓練が完了しました！")
    print(f"結果は {config['work_dir']} に保存されています。")


if __name__ == "__main__":
    main()
