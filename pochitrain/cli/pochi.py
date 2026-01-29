#!/usr/bin/env python3
"""
pochitrain 統一CLI エントリーポイント.

訓練・推論・ハイパーパラメータ最適化を統合したコマンドライン インターフェース
"""

import argparse
import logging
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from pochitrain import (
    LoggerManager,
    PochiImageDataset,
    PochiPredictor,
    PochiTrainer,
    create_data_loaders,
)
from pochitrain.utils import (
    ConfigLoader,
    load_config_auto,
    validate_data_path,
    validate_model_path,
)
from pochitrain.validation import ConfigValidator

# グローバル変数で訓練停止フラグを管理
training_interrupted = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Ctrl+Cのシグナルハンドラー."""
    global training_interrupted
    training_interrupted = True

    # シグナルハンドラー内で直接ロガーを作成
    logger = setup_logging()
    logger.warning("訓練を安全に停止しています... (Ctrl+Cが検出されました)")
    logger.warning("現在のエポックが完了次第、訓練を終了します。")


def setup_logging(logger_name: str = "pochitrain") -> logging.Logger:
    """
    ログ設定の初期化.

    Args:
        logger_name (str): ロガー名

    Returns:
        logger: 設定済みロガー
    """
    logger_manager = LoggerManager()
    return logger_manager.get_logger(logger_name)


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


def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
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


def train_command(args: argparse.Namespace) -> None:
    """訓練サブコマンドの実行."""
    # Ctrl+Cの安全な処理を設定
    signal.signal(signal.SIGINT, signal_handler)

    logger = setup_logging()
    logger.info("=== pochitrain 訓練モード ===")
    logger.info(
        "安全終了: 訓練中にCtrl+Cを押すと、現在のエポック完了後に安全に終了します。"
    )

    # 設定ファイルの読み込み
    try:
        config = ConfigLoader.load_config(args.config)
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

    # 層別学習率設定の明示的ログ出力
    enable_layer_wise_lr = config.get("enable_layer_wise_lr", False)
    if enable_layer_wise_lr:
        logger.info("層別学習率: 有効")
        layer_wise_lr_config = config.get("layer_wise_lr_config", {})
        layer_rates = layer_wise_lr_config.get("layer_rates", {})
        logger.info(f"層別学習率設定: {layer_rates}")
    else:
        logger.info("層別学習率: 無効")

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
        cudnn_benchmark=config.get("cudnn_benchmark", False),
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
        enable_layer_wise_lr=config.get("enable_layer_wise_lr", False),
        layer_wise_lr_config=config.get("layer_wise_lr_config"),
    )

    # データセットパスの保存
    logger.info("データセットパスを保存しています...")
    trainer.save_dataset_paths(train_loader, val_loader)

    # 設定ファイルの保存
    logger.info("設定ファイルを保存しています...")
    config_path_obj = Path(args.config)
    saved_config_path = trainer.save_training_config(config_path_obj)
    logger.info(f"設定ファイルを保存しました: {saved_config_path}")

    # メトリクスエクスポート設定の適用
    trainer.enable_metrics_export = config.get("enable_metrics_export", True)
    if trainer.enable_metrics_export:
        logger.info("訓練メトリクスのCSV出力とグラフ生成が有効です")

    # Early Stopping設定の適用
    early_stopping_config = config.get("early_stopping")
    if early_stopping_config and early_stopping_config.get("enabled", False):
        trainer.early_stopping_config = early_stopping_config
        logger.info(
            f"Early Stopping: 有効 "
            f"(patience={early_stopping_config.get('patience', 10)}, "
            f"min_delta={early_stopping_config.get('min_delta', 0.0)}, "
            f"monitor={early_stopping_config.get('monitor', 'val_accuracy')})"
        )
    else:
        logger.info("Early Stopping: 無効")

    # 勾配トレース設定の適用
    trainer.enable_gradient_tracking = config.get("enable_gradient_tracking", False)
    if trainer.enable_gradient_tracking:
        logger.info("勾配トレース機能が有効です")
        gradient_config = config.get("gradient_tracking_config", {})
        trainer.gradient_tracking_config.update(gradient_config)

    # 訓練実行
    logger.info("訓練を開始します...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        stop_flag_callback=lambda: training_interrupted,
    )

    logger.info("訓練が完了しました！")
    logger.info(f"結果は {config['work_dir']} に保存されています。")


def get_indexed_output_dir(base_dir: str) -> Path:
    """インデックス付きの出力ディレクトリを取得する.

    既存のディレクトリがある場合、連番を付与して新しいディレクトリ名を生成.
    例: optuna_results, optuna_results_001, optuna_results_002, ...

    Args:
        base_dir: ベースとなる出力ディレクトリパス

    Returns:
        インデックス付きの出力ディレクトリパス
    """
    base_path = Path(base_dir)

    # ベースディレクトリが存在しない場合はそのまま返す
    if not base_path.exists():
        return base_path

    # 連番を探索
    parent = base_path.parent
    stem = base_path.name
    index = 1

    while True:
        indexed_path = parent / f"{stem}_{index:03d}"
        if not indexed_path.exists():
            return indexed_path
        index += 1


def infer_command(args: argparse.Namespace) -> None:
    """推論サブコマンドの実行."""
    import time

    import torch

    logger = setup_logging()
    logger.info("=== pochitrain 推論モード ===")

    # モデルパス確認
    model_path = Path(args.model_path)
    validate_model_path(model_path)
    logger.info(f"使用するモデル: {model_path}")

    # 設定ファイル読み込み（自動検出または指定）
    if args.config_path:
        config_path = Path(args.config_path)
        try:
            config = ConfigLoader.load_config(str(config_path))
            logger.info(f"設定ファイルを読み込み: {config_path}")
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return
    else:
        config = load_config_auto(model_path)

    # データパスの決定（--data指定 or configのval_data_root）
    if args.data:
        data_path = Path(args.data)
    elif "val_data_root" in config:
        data_path = Path(config["val_data_root"])
        logger.info(f"データパスをconfigから取得: {data_path}")
    else:
        logger.error("--data を指定するか、configにval_data_rootを設定してください")
        return
    validate_data_path(data_path)
    logger.info(f"推論データ: {data_path}")

    # 出力ディレクトリの決定（modelsと同階層）
    if args.output:
        output_dir = args.output
    else:
        # modelsフォルダと同階層に出力
        model_dir = model_path.parent  # models フォルダ
        work_dir = model_dir.parent  # work_dirs/YYYYMMDD_XXX フォルダ
        output_dir = str(work_dir / "inference_results")

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
        logger.info("推論器の作成成功")

    except Exception as e:
        logger.error(f"推論器作成エラー: {e}")
        return

    # データローダー作成（訓練時と同じval_transformを使用）
    logger.info("データローダーを作成しています...")
    try:
        val_dataset = PochiImageDataset(
            str(data_path), transform=config["val_transform"]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 0),
            pin_memory=True,
        )

        logger.info(f"推論データ: {len(val_dataset)}枚の画像")
        logger.info("使用されたTransform (設定ファイルから):")
        for i, transform in enumerate(config["val_transform"].transforms):
            logger.info(f"   {i+1}. {transform}")

    except Exception as e:
        logger.error(f"データローダー作成エラー: {e}")
        return

    # 推論実行（時間計測付き）
    logger.info("推論を開始します...")
    try:
        use_gpu = config["device"] == "cuda" and torch.cuda.is_available()

        if use_gpu:
            # GPU時間計測: CUDA Eventを使用
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            predictions, confidences = predictor.predict(val_loader)
            end_event.record()
            torch.cuda.synchronize()
            total_inference_time_ms = start_event.elapsed_time(end_event)
        else:
            # CPU時間計測
            start_time = time.perf_counter()
            predictions, confidences = predictor.predict(val_loader)
            total_inference_time_ms = (time.perf_counter() - start_time) * 1000

        # 結果整理
        image_paths = val_dataset.get_file_paths()
        predicted_labels = predictions.tolist()
        confidence_scores = confidences.tolist()
        true_labels = val_dataset.labels
        class_names = val_dataset.get_classes()

        logger.info("推論完了")

    except Exception as e:
        logger.error(f"推論実行エラー: {e}")
        return

    # CSV出力
    logger.info("結果をCSVに出力しています...")
    try:
        # 混同行列設定を取得（設定ファイルにあれば使用）
        cm_config = config.get("confusion_matrix_config", None)

        results_csv, summary_csv = predictor.export_results_to_workspace(
            image_paths=image_paths,
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            confidence_scores=confidence_scores,
            class_names=class_names,
            results_filename="inference_results.csv",
            summary_filename="inference_summary.csv",
            cm_config=cm_config,
        )

        # 精度計算・表示
        accuracy_info = predictor.calculate_accuracy(predicted_labels, true_labels)
        num_samples = accuracy_info["total_samples"]
        avg_time_per_image = (
            total_inference_time_ms / num_samples if num_samples > 0 else 0
        )
        throughput = 1000 / avg_time_per_image if avg_time_per_image > 0 else 0

        logger.info("=== 推論結果 ===")
        logger.info(f"処理画像数: {num_samples}枚")
        logger.info(f"正解数: {accuracy_info['correct_predictions']}")
        logger.info(f"精度: {accuracy_info['accuracy_percentage']:.2f}%")
        logger.info(
            f"平均処理時間: {avg_time_per_image:.2f} ms/image（データ読み込み含む）"
        )
        logger.info(f"スループット: {throughput:.1f} images/sec")
        logger.info(f"詳細結果: {results_csv}")
        logger.info(f"サマリー: {summary_csv}")

        # ワークスペース情報
        workspace_info = predictor.get_inference_workspace_info()
        logger.info(f"ワークスペース: {workspace_info['workspace_name']}")

        logger.info("推論が完了しました！")

    except Exception as e:
        logger.error(f"CSV出力エラー: {e}")
        return


def optimize_command(args: argparse.Namespace) -> None:
    """最適化サブコマンドの実行."""
    logger = setup_logging()
    logger.info("=== pochitrain Optuna最適化モード ===")

    # 設定ファイルの読み込み
    try:
        config = ConfigLoader.load_config(args.config)
        logger.info(f"設定ファイルを読み込みました: {args.config}")
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {args.config}")
        return

    # Optuna関連のインポート（遅延インポート）
    try:
        from pochitrain.optimization import (
            ClassificationObjective,
            DefaultParamSuggestor,
            JsonResultExporter,
            OptunaStudyManager,
            StatisticsExporter,
            VisualizationExporter,
        )
        from pochitrain.optimization.result_exporter import ConfigExporter
    except ImportError as e:
        logger.error(f"Optunaモジュールのインポートに失敗しました: {e}")
        logger.error(
            "Optunaがインストールされているか確認してください: pip install optuna"
        )
        return

    # 出力ディレクトリの決定（インデックス付き）
    output_dir = get_indexed_output_dir(args.output)
    logger.info(f"出力ディレクトリ: {output_dir}")

    # データローダーの作成
    logger.info("データローダーを作成しています...")
    try:
        train_loader, val_loader, classes = create_data_loaders(
            train_root=config.get("train_data_root", "data/train"),
            val_root=config.get("val_data_root", "data/val"),
            batch_size=config.get("batch_size", 32),
            num_workers=config.get("num_workers", 0),
            train_transform=config.get("train_transform"),
            val_transform=config.get("val_transform"),
        )
        config["num_classes"] = len(classes)
        logger.info(f"クラス数: {len(classes)}")
        logger.info(f"クラス名: {classes}")
        logger.info(f"訓練サンプル数: {len(train_loader.dataset)}")
        logger.info(f"検証サンプル数: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"データローダーの作成に失敗しました: {e}")
        return

    # パラメータサジェスターを作成
    search_space = config.get("search_space", {})
    if not search_space:
        logger.error("search_spaceが設定されていません")
        return
    param_suggestor = DefaultParamSuggestor(search_space)
    logger.info(f"探索空間: {list(search_space.keys())}")

    # 目的関数を作成
    objective = ClassificationObjective(
        base_config=config,
        param_suggestor=param_suggestor,
        train_loader=train_loader,
        val_loader=val_loader,
        optuna_epochs=config.get("optuna_epochs", 10),
        device=config.get("device", "cuda"),
    )

    # Study管理を作成
    storage = config.get("storage")
    study_manager = OptunaStudyManager(storage=storage)

    # Studyを作成
    logger.info("Optuna Studyを作成しています...")
    study = study_manager.create_study(
        study_name=config.get("study_name", "pochitrain_optimization"),
        direction=config.get("direction", "maximize"),
        sampler=config.get("sampler", "TPESampler"),
        pruner=config.get("pruner"),
    )
    logger.info(f"Study名: {study.study_name}")
    logger.info(f"方向: {config.get('direction', 'maximize')}")
    logger.info(f"サンプラー: {config.get('sampler', 'TPESampler')}")

    # 最適化を実行
    n_trials = config.get("n_trials", 20)
    logger.info(f"最適化を開始します（{n_trials}試行）...")
    study_manager.optimize(
        objective=objective,
        n_trials=n_trials,
        n_jobs=config.get("n_jobs", 1),
    )

    # 結果を取得
    best_params = study_manager.get_best_params()
    best_value = study_manager.get_best_value()

    logger.info("=" * 50)
    logger.info("最適化完了！")
    logger.info("=" * 50)
    logger.info(f"最良スコア: {best_value:.4f}")
    logger.info("最良パラメータ:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # 結果をエクスポート
    logger.info(f"結果を出力しています: {output_dir}")

    # JSON形式でエクスポート
    json_exporter = JsonResultExporter()
    json_exporter.export(best_params, best_value, study, str(output_dir))

    # Python設定ファイル形式でエクスポート
    config_exporter = ConfigExporter(config)
    config_exporter.export(best_params, best_value, study, str(output_dir))

    # 統計情報とパラメータ重要度をエクスポート
    statistics_exporter = StatisticsExporter()
    statistics_exporter.export(best_params, best_value, study, str(output_dir))

    # 可視化グラフをエクスポート（Plotly HTML）
    visualization_exporter = VisualizationExporter()
    visualization_exporter.export(best_params, best_value, study, str(output_dir))

    logger.info("生成されたファイル:")
    logger.info(f"  - {output_dir}/best_params.json")
    logger.info(f"  - {output_dir}/trials_history.json")
    logger.info(f"  - {output_dir}/optimized_config.py")
    logger.info(f"  - {output_dir}/study_statistics.json")
    logger.info(f"  - {output_dir}/optimization_history.html")
    logger.info(f"  - {output_dir}/param_importances.html")
    logger.info("最適化パラメータで訓練するには:")
    logger.info(f"  uv run pochi train --config {output_dir}/optimized_config.py")


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="pochitrain - 統合CLI（訓練・推論・最適化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 訓練
  uv run pochi train --config configs/pochi_train_config.py

  # 推論（基本）
  uv run pochi infer
    -m work_dirs/20250813_003/models/best_epoch40.pth
    -d data/val
    -c work_dirs/20250813_003/config.py

  # 推論（カスタム出力先）
  uv run pochi infer
    --model-path work_dirs/20250813_003/models/best_epoch40.pth
    --data data/test
    --config-path work_dirs/20250813_003/config.py
    --output custom_results

  # ハイパーパラメータ最適化
  uv run pochi optimize --config configs/pochi_train_config.py
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
    infer_parser.add_argument("model_path", help="モデルファイルパス")
    infer_parser.add_argument(
        "--data", "-d", help="推論データパス（省略時はconfigのval_data_rootを使用）"
    )
    infer_parser.add_argument(
        "--config-path",
        "-c",
        help="設定ファイルパス（省略時はモデルパスから自動検出）",
    )
    infer_parser.add_argument(
        "--output",
        "-o",
        help="結果出力ディレクトリ（default: モデルと同じディレクトリ/inference_results）",
    )

    # 最適化サブコマンド
    optimize_parser = subparsers.add_parser("optimize", help="ハイパーパラメータ最適化")
    optimize_parser.add_argument(
        "--config",
        default="configs/pochi_train_config.py",
        help="設定ファイルパス (default: configs/pochi_train_config.py)",
    )
    optimize_parser.add_argument(
        "--output",
        "-o",
        default="work_dirs/optuna_results",
        help="結果出力ディレクトリ (default: work_dirs/optuna_results)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "optimize":
        optimize_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
