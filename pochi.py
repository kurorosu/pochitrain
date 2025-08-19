#!/usr/bin/env python3
"""
pochitrain 統一CLI エントリーポイント.

訓練と推論を統合したコマンドライン インターフェース
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from pochitrain import (
    LoggerManager,
    PochiImageDataset,
    PochiPredictor,
    PochiTrainer,
    create_data_loaders,
)
from pochitrain.validation import ConfigValidator


def setup_logging(logger_name: str = "pochitrain"):
    """
    ログ設定の初期化.

    Args:
        logger_name (str): ロガー名

    Returns:
        logger: 設定済みロガー
    """
    logger_manager = LoggerManager()
    return logger_manager.get_logger(logger_name)


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


def find_best_model(work_dir: str) -> Path:
    """
    work_dir内でベストモデルを自動検出.

    Args:
        work_dir (str): 作業ディレクトリパス

    Returns:
        Path: ベストモデルのパス

    Raises:
        FileNotFoundError: モデルが見つからない場合
    """
    work_path = Path(work_dir)
    models_dir = work_path / "models"

    if not models_dir.exists():
        raise FileNotFoundError(f"モデルディレクトリが見つかりません: {models_dir}")

    # best_epoch*.pth ファイルを検索
    model_files = list(models_dir.glob("best_epoch*.pth"))

    if not model_files:
        raise FileNotFoundError(
            f"ベストモデルが見つかりません: {models_dir}/best_epoch*.pth"
        )

    # 最新のモデルを選択（エポック番号が最大のもの）
    best_model = max(model_files, key=lambda x: x.name)
    return best_model


def validate_config(config: dict, logger) -> bool:
    """
    設定のバリデーション.

    Args:
        config (dict): 設定辞書
        logger: ロガー

    Returns:
        bool: バリデーション結果
    """
    validator = ConfigValidator(logger)
    return validator.validate(config)


def train_command(args):
    """訓練サブコマンドの実行."""
    logger = setup_logging()
    logger.info("=== pochitrain 訓練モード ===")

    # 設定ファイルの読み込み
    try:
        config = load_config(args.config)
        logger.info(f"設定ファイルを読み込みました: {args.config}")
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {args.config}")
        logger.error("configs/pochi_train_config.py を作成してください。")
        return

    # 設定のバリデーション
    if not validate_config(config, logger):
        logger.error("設定にエラーがあります。修正してください。")
        return

    # 設定確認ログ
    logger.info("=== 設定確認 ===")
    logger.info(f"モデル: {config['model_name']}")
    logger.info(f"デバイス: {config['device']}")
    logger.info(f"学習率: {config['learning_rate']}")
    logger.info(f"オプティマイザー: {config['optimizer']}")

    # スケジューラー設定の明示的ログ出力
    scheduler_name = config.get("scheduler")
    if scheduler_name is None:
        logger.info("スケジューラー: なし（固定学習率）")
    else:
        logger.info(f"スケジューラー: {scheduler_name}")
        scheduler_params = config.get("scheduler_params")
        logger.info(f"スケジューラーパラメータ: {scheduler_params}")

    # クラス重み設定の明示的ログ出力
    class_weights = config.get("class_weights")
    if class_weights is None:
        logger.info("クラス重み: なし（均等扱い）")
    else:
        logger.info(f"クラス重み: {class_weights}")

    logger.info("==================")

    # データローダーの作成
    logger.info("データローダーを作成しています...")
    try:
        train_loader, val_loader, classes = create_data_loaders(
            train_root=config["train_data_root"],
            val_root=config["val_data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_transform=config.get("train_transform"),
            val_transform=config.get("val_transform"),
        )

        logger.info(f"クラス数: {len(classes)}")
        logger.info(f"クラス名: {classes}")
        logger.info(f"訓練バッチ数: {len(train_loader)}")
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
        device=config["device"],
        pretrained=config["pretrained"],
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
    config_path_obj = Path(args.config)
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


def infer_command(args):
    """推論サブコマンドの実行."""
    logger = setup_logging()
    logger.info("=== pochitrain 推論モード ===")

    # 設定ファイル読み込み
    config_path = Path(args.config_path)
    try:
        config = load_config(str(config_path))
        logger.info(f"設定ファイルを読み込み: {config_path}")
    except Exception as e:
        logger.error(f"設定ファイル読み込みエラー: {e}")
        logger.error(
            f"設定ファイルが存在することを確認してください: {args.config_path}"
        )
        return

    # モデルパス確認
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"指定されたモデルファイルが見つかりません: {model_path}")
        return
    logger.info(f"使用するモデル: {model_path}")

    # データパス確認
    data_root = args.data
    if not Path(data_root).exists():
        logger.error(f"データディレクトリが見つかりません: {data_root}")
        return
    logger.info(f"推論データ: {data_root}")

    # 出力ディレクトリの決定（モデルと同じディレクトリ）
    if args.output:
        output_dir = args.output
    else:
        # モデルファイルと同じディレクトリに出力
        model_dir = model_path.parent
        output_dir = str(model_dir / "inference_results")

    logger.info(f"推論結果出力先: {output_dir}")

    # 推論器作成
    logger.info("推論器を作成しています...")
    try:
        predictor = PochiPredictor(
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            device=config["device"],
            model_path=str(model_path),
            work_dir=output_dir,
        )
        logger.info("✅ 推論器の作成成功")
    except Exception as e:
        logger.error(f"推論器作成エラー: {e}")
        return

    # データローダー作成（訓練時と同じval_transformを使用）
    logger.info("データローダーを作成しています...")
    try:
        val_dataset = PochiImageDataset(data_root, transform=config["val_transform"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 0),
            pin_memory=True,
        )

        logger.info(f"📊 推論データ: {len(val_dataset)}枚の画像")
        logger.info("📋 使用されたTransform (設定ファイルから):")
        for i, transform in enumerate(config["val_transform"].transforms):
            logger.info(f"   {i+1}. {transform}")

    except Exception as e:
        logger.error(f"データローダー作成エラー: {e}")
        return

    # 推論実行
    logger.info("推論を開始します...")
    try:
        predictions, confidences = predictor.predict(val_loader)

        # 結果整理
        image_paths = val_dataset.get_file_paths()
        predicted_labels = predictions.tolist()
        confidence_scores = confidences.tolist()
        true_labels = val_dataset.labels
        class_names = val_dataset.get_classes()

        logger.info("✅ 推論完了")

    except Exception as e:
        logger.error(f"推論実行エラー: {e}")
        return

    # CSV出力
    logger.info("結果をCSVに出力しています...")
    try:
        results_csv, summary_csv = predictor.export_results_to_workspace(
            image_paths=image_paths,
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            confidence_scores=confidence_scores,
            class_names=class_names,
            results_filename="inference_results.csv",
            summary_filename="inference_summary.csv",
        )

        # 精度計算・表示
        accuracy_info = predictor.calculate_accuracy(predicted_labels, true_labels)

        logger.info("=== 推論結果 ===")
        logger.info(f"処理画像数: {accuracy_info['total_samples']}枚")
        logger.info(f"正解数: {accuracy_info['correct_predictions']}")
        logger.info(f"精度: {accuracy_info['accuracy_percentage']:.2f}%")
        logger.info(f"詳細結果: {results_csv}")
        logger.info(f"サマリー: {summary_csv}")

        # ワークスペース情報
        workspace_info = predictor.get_inference_workspace_info()
        logger.info(f"ワークスペース: {workspace_info['workspace_name']}")

        logger.info("推論が完了しました！")

    except Exception as e:
        logger.error(f"CSV出力エラー: {e}")
        return


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="pochitrain - 統合CLI（訓練・推論）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 訓練
  python pochi.py train
    --config configs/pochi_train_config.py

  # 推論（基本）
  python pochi.py infer
    -m work_dirs/20250813_003/models/best_epoch40.pth
    -d data/val
    -c work_dirs/20250813_003/config.py

  # 推論（カスタム出力先）
  python pochi.py infer
    --model-path work_dirs/20250813_003/models/best_epoch40.pth
    --data data/test
    --config-path work_dirs/20250813_003/config.py
    --output custom_results
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # 訓練サブコマンド
    train_parser = subparsers.add_parser("train", help="モデル訓練")
    train_parser.add_argument(
        "--config",
        default="configs/pochi_train_config.py",
        help="設定ファイルパス (default: configs/pochi_train_config.py)",
    )

    # 推論サブコマンド
    infer_parser = subparsers.add_parser("infer", help="モデル推論")
    infer_parser.add_argument(
        "--model-path", "-m", required=True, help="モデルファイルパス"
    )
    infer_parser.add_argument("--data", "-d", required=True, help="推論データパス")
    infer_parser.add_argument(
        "--config-path",
        "-c",
        required=True,
        help="設定ファイルパス（work_dir/config.py）",
    )
    infer_parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ（default: モデルと同じディレクトリ/inference_results）",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
