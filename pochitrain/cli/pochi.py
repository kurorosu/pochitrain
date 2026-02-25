#!/usr/bin/env python3
"""
pochitrain 統一CLI エントリーポイント.

訓練・推論・ハイパーパラメータ最適化を統合したコマンドライン インターフェース
"""

import argparse
import logging
import re
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Optional, Sized, cast

from pydantic import ValidationError

from pochitrain import (
    LoggerManager,
    PochiConfig,
    PochiTrainer,
    create_data_loaders,
)
from pochitrain.cli.arg_types import positive_int
from pochitrain.inference.benchmark import (
    build_pytorch_benchmark_result,
    resolve_env_name,
    write_benchmark_result_json,
)
from pochitrain.inference.services import PyTorchInferenceService
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
)
from pochitrain.logging.logger_manager import LogLevel
from pochitrain.utils import (
    ConfigLoader,
    load_config_auto,
    validate_model_path,
)

training_interrupted = False


def create_signal_handler(debug: bool = False) -> Any:
    """デバッグフラグを保持するシグナルハンドラーを生成する.

    Args:
        debug (bool): デバッグモードが有効かどうか

    Returns:
        シグナルハンドラー関数
    """

    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        """Ctrl+Cのシグナルハンドラー."""
        global training_interrupted
        training_interrupted = True

        logger = setup_logging(debug=debug)
        logger.warning("訓練を安全に停止しています... (Ctrl+Cが検出されました)")
        logger.warning("現在のエポックが完了次第、訓練を終了します。")

    return signal_handler


def setup_logging(
    logger_name: str = "pochitrain", debug: bool = False
) -> logging.Logger:
    """
    ログ設定の初期化.

    Args:
        logger_name (str): ロガー名
        debug (bool): デバッグモードが有効かどうか

    Returns:
        logger: 設定済みロガー
    """
    logger_manager = LoggerManager()
    level = LogLevel.DEBUG if debug else LogLevel.INFO
    logger_manager.set_default_level(level)
    for existing_name in logger_manager.get_available_loggers():
        logger_manager.set_logger_level(existing_name, level)
    return logger_manager.get_logger(logger_name, level=level)


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

    model_files = list(models_dir.glob("best_epoch*.pth"))

    if not model_files:
        raise FileNotFoundError(
            f"ベストモデルが見つかりません: {models_dir}/best_epoch*.pth"
        )

    best_model = max(
        model_files,
        key=lambda x: (
            int(m.group(1)) if (m := re.search(r"best_epoch(\d+)", x.name)) else 0
        ),
    )
    return best_model


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
            Dict[str, Any], pochi_config.layer_wise_lr_config.model_dump()
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
            Dict[str, Any], pochi_config.layer_wise_lr_config.model_dump()
        ),
        early_stopping_config=(
            cast(Dict[str, Any], pochi_config.early_stopping.model_dump())
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
        cast(Dict[str, Any], pochi_config.early_stopping.model_dump())
        if pochi_config.early_stopping is not None
        else None
    )
    if not (early_stopping_config and early_stopping_config.get("enabled", False)):
        logger.debug("Early Stopping: 無効")

    trainer.enable_gradient_tracking = pochi_config.enable_gradient_tracking
    if trainer.enable_gradient_tracking:
        logger.debug("勾配トレース機能が有効です")
        gradient_config = cast(
            Dict[str, Any], pochi_config.gradient_tracking_config.model_dump()
        )
        trainer.gradient_tracking_config.update(gradient_config)

    logger.info("訓練を開始します...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=pochi_config.epochs,
        stop_flag_callback=lambda: training_interrupted,
    )

    logger.info("訓練が完了しました！")
    logger.info(f"結果は {pochi_config.work_dir} に保存されています。")


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

    if not base_path.exists():
        return base_path

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
    logger = setup_logging(debug=args.debug)
    logger.debug("=== pochitrain 推論モード ===")

    model_path = Path(args.model_path)
    validate_model_path(model_path)
    logger.debug(f"使用するモデル: {model_path}")

    if args.config_path:
        config_path = Path(args.config_path)
        try:
            config = ConfigLoader.load_config(str(config_path))
            logger.debug(f"設定ファイルを読み込み: {config_path}")
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return
    else:
        config = load_config_auto(model_path)

    try:
        pochi_config = PochiConfig.from_dict(config)
    except ValidationError as e:
        logger.error(f"設定にエラーがあります:\n{e}")
        return

    service = PyTorchInferenceService(logger)

    requested_pipeline = str(getattr(args, "pipeline", "current"))
    cli_request = InferenceCliRequest(
        model_path=model_path,
        data_path=Path(args.data) if args.data else None,
        output_dir=Path(args.output) if args.output else None,
        requested_pipeline=requested_pipeline,
    )
    try:
        resolved_paths = service.resolve_paths(cli_request, config)
    except ValueError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"パス解決エラー: {e}")
        return

    data_path = resolved_paths.data_path
    workspace_dir = resolved_paths.output_dir

    use_gpu = pochi_config.device == "cuda"
    pipeline = service.resolve_pipeline(
        cli_request.requested_pipeline,
        use_gpu=use_gpu,
    )
    runtime_options = service.resolve_runtime_options(
        config=config,
        pipeline=pipeline,
        use_gpu=use_gpu,
    )

    try:
        predictor = service.create_predictor(pochi_config, model_path)
    except Exception as e:
        logger.error(f"推論器作成エラー: {e}")
        return

    logger.debug("データローダーを作成しています...")
    try:
        (
            val_loader,
            val_dataset,
            pipeline,
            norm_mean,
            norm_std,
        ) = service.create_dataloader(
            config,
            data_path,
            pochi_config.val_transform,
            pipeline,
            runtime_options,
        )
    except Exception as e:
        logger.error(f"データローダー作成エラー: {e}")
        return

    input_size = service.detect_input_size(pochi_config, val_dataset)

    try:
        runtime_adapter = service.create_runtime_adapter(predictor)
        runtime_request = service.build_runtime_execution_request(
            data_loader=val_loader,
            runtime_adapter=runtime_adapter,
            use_gpu_pipeline=pipeline == "gpu",
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_cuda_timing=runtime_adapter.use_cuda_timing,
            gpu_non_blocking=bool(config.get("gpu_non_blocking", True)),
        )
        run_result = service.run(
            runtime_request,
        )
    except Exception as e:
        logger.error(f"推論実行エラー: {e}")
        return

    try:
        cm_config = (
            cast(Dict[str, Any], pochi_config.confusion_matrix_config.model_dump())
            if pochi_config.confusion_matrix_config is not None
            else None
        )

        service.aggregate_and_export(
            workspace_dir=workspace_dir,
            model_path=model_path,
            data_path=data_path,
            dataset=val_dataset,
            run_result=run_result,
            input_size=input_size,
            model_info=predictor.get_model_info(),
            cm_config=cm_config,
            results_filename="pytorch_inference_results.csv",
            summary_filename="pytorch_inference_summary.txt",
        )

        if bool(getattr(args, "benchmark_json", False)):
            configured_env_name = getattr(
                args, "benchmark_env_name", None
            ) or config.get("benchmark_env_name")
            env_name = resolve_env_name(
                use_gpu=use_gpu,
                configured_env_name=(
                    str(configured_env_name)
                    if configured_env_name is not None
                    else None
                ),
            )
            benchmark_result = build_pytorch_benchmark_result(
                use_gpu=use_gpu,
                pipeline=pipeline,
                model_name=str(config.get("model_name", model_path.stem)),
                batch_size=runtime_options.batch_size,
                gpu_non_blocking=runtime_request.execution_request.gpu_non_blocking,
                pin_memory=runtime_options.pin_memory,
                input_size=input_size,
                avg_time_per_image=run_result.avg_time_per_image,
                avg_total_time_per_image=run_result.avg_total_time_per_image,
                num_samples=run_result.num_samples,
                total_samples=run_result.total_samples,
                warmup_samples=run_result.warmup_samples,
                accuracy=run_result.accuracy_percent,
                env_name=env_name,
            )
            try:
                benchmark_json_path = write_benchmark_result_json(
                    output_dir=workspace_dir,
                    benchmark_result=benchmark_result,
                )
                logger.info(
                    f"ベンチマークJSONを出力しました: {benchmark_json_path.name}"
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    f"ベンチマークJSONの保存に失敗しました, error: {exc}",
                )

        logger.info("推論完了")
        logger.info(
            f"ワークスペース: {workspace_dir.name}へサマリーファイルを出力しました"
        )
    except Exception as e:
        logger.error(f"CSV出力エラー: {e}")
        return


def optimize_command(args: argparse.Namespace) -> None:
    """最適化サブコマンドの実行."""
    logger = setup_logging(debug=args.debug)
    logger.info("=== pochitrain Optuna最適化モード ===")

    try:
        config = ConfigLoader.load_config(args.config)
        logger.info(f"設定ファイルを読み込みました: {args.config}")
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {args.config}")
        return

    try:
        pochi_config = PochiConfig.from_dict(config)
    except ValidationError as e:
        logger.error(f"設定にエラーがあります:\n{e}")
        return

    # optional依存のため, optimize 実行時のみ読み込む.
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

    output_dir = get_indexed_output_dir(args.output)
    logger.info(f"出力ディレクトリ: {output_dir}")

    logger.debug("データローダーを作成しています...")
    try:
        train_loader, val_loader, classes = create_data_loaders(
            train_root=pochi_config.train_data_root or "data/train",
            val_root=pochi_config.val_data_root or "data/val",
            batch_size=pochi_config.batch_size,
            num_workers=pochi_config.num_workers,
            pin_memory=pochi_config.train_pin_memory,
            train_transform=pochi_config.train_transform,
            val_transform=pochi_config.val_transform,
        )
        config["num_classes"] = len(classes)
        pochi_config.num_classes = len(classes)
        logger.debug(f"クラス数: {len(classes)}")
        logger.debug(f"クラス名: {classes}")
        logger.info(f"訓練サンプル数: {len(cast(Sized, train_loader.dataset))}")
        logger.info(f"検証サンプル数: {len(cast(Sized, val_loader.dataset))}")
    except Exception as e:
        logger.error(f"データローダーの作成に失敗しました: {e}")
        return

    if pochi_config.optuna is None or not pochi_config.optuna.search_space:
        logger.error("search_spaceが設定されていません")
        return
    search_space = pochi_config.optuna.search_space
    param_suggestor = DefaultParamSuggestor(search_space)
    logger.info(f"探索空間: {list(search_space.keys())}")

    objective = ClassificationObjective(
        base_config=pochi_config,
        param_suggestor=param_suggestor,
        train_loader=train_loader,
        val_loader=val_loader,
        optuna_epochs=pochi_config.optuna.optuna_epochs,
        device=pochi_config.device,
    )

    storage = pochi_config.optuna.storage
    study_manager = OptunaStudyManager(storage=storage)

    logger.info("Optuna Studyを作成しています...")
    study = study_manager.create_study(
        study_name=pochi_config.optuna.study_name,
        direction=pochi_config.optuna.direction,
        sampler=pochi_config.optuna.sampler,
        pruner=pochi_config.optuna.pruner,
    )
    logger.info(f"Study名: {study.study_name}")
    logger.info(f"方向: {pochi_config.optuna.direction}")
    logger.info(f"サンプラー: {pochi_config.optuna.sampler}")

    n_trials = pochi_config.optuna.n_trials
    logger.info(f"最適化を開始します（{n_trials}試行）...")
    study_manager.optimize(
        objective=objective,
        n_trials=n_trials,
        n_jobs=pochi_config.optuna.n_jobs,
    )

    best_params = study_manager.get_best_params()
    best_value = study_manager.get_best_value()

    logger.info("=" * 50)
    logger.info("最適化完了！")
    logger.info("=" * 50)
    logger.info(f"最良スコア: {best_value:.4f}")
    logger.info("最良パラメータ:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"結果を出力しています: {output_dir}")

    json_exporter = JsonResultExporter()
    json_exporter.export(best_params, best_value, study, str(output_dir))

    config_exporter = ConfigExporter(pochi_config)
    config_exporter.export(best_params, best_value, study, str(output_dir))

    statistics_exporter = StatisticsExporter()
    statistics_exporter.export(best_params, best_value, study, str(output_dir))

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


def convert_command(args: argparse.Namespace) -> None:
    """TensorRTエンジン変換サブコマンドの実行."""
    logger = setup_logging(debug=args.debug)
    logger.debug("=== pochitrain TensorRT変換モード ===")

    try:
        from pochitrain.tensorrt.converter import TensorRTConverter
        from pochitrain.tensorrt.inference import check_tensorrt_availability
    except ImportError:
        logger.error(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )
        return

    if not check_tensorrt_availability():
        logger.error(
            "TensorRTが利用できません. " "TensorRT SDKをインストールしてください."
        )
        return

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        logger.error(f"ONNXモデルが見つかりません: {onnx_path}")
        return

    if args.int8:
        precision = "int8"
    elif args.fp16:
        precision = "fp16"
    else:
        precision = "fp32"

    if args.output:
        output_path = Path(args.output)
    else:
        stem = onnx_path.stem
        if precision != "fp32":
            stem = f"{stem}_{precision}"
        output_path = onnx_path.with_name(f"{stem}.engine")

    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"精度: {precision.upper()}")
    logger.info(f"出力: {output_path}")

    # 動的シェイプONNXの変換時は全精度で入力サイズ指定が必要.
    input_shape = None
    if args.input_size:
        # チャンネル数は RGB (C=3) 固定. pochitrain は RGB 画像のみ対応.
        input_shape = (3, args.input_size[0], args.input_size[1])
        logger.debug(f"CLI指定の入力形状: {input_shape}")
    else:
        try:
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            input_tensor = onnx_model.graph.input[0]
            input_dims = input_tensor.type.tensor_type.shape.dim

            dynamic_dims = [
                d.dim_param for d in input_dims[1:] if d.dim_value == 0 and d.dim_param
            ]
            if any(d.dim_value == 0 for d in input_dims[1:]):
                dynamic_info = (
                    f" (動的次元: {', '.join(dynamic_dims)})" if dynamic_dims else ""
                )
                logger.error(
                    f"ONNXモデルに動的シェイプが含まれています{dynamic_info}. "
                    "--input-size で入力サイズを明示的に指定してください. "
                    "例: --input-size 224 224"
                )
                return
        except ImportError:
            logger.debug(
                "onnxパッケージが未インストールのため動的シェイプ検出をスキップ"
            )
        except Exception as e:
            logger.debug(f"ONNX動的シェイプ検出中にエラー: {e}")

    calibrator = None
    if precision == "int8":
        from pochitrain.tensorrt.calibrator import create_int8_calibrator

        config = None
        if args.config_path:
            config_path = Path(args.config_path)
            try:
                config = ConfigLoader.load_config(str(config_path))
                logger.debug(f"設定ファイルを読み込み: {config_path}")
            except Exception as e:
                logger.error(f"設定ファイル読み込みエラー: {e}")
                return
        else:
            config = load_config_auto(onnx_path)

        if args.calib_data:
            calib_data_root = args.calib_data
        elif config.get("val_data_root"):
            calib_data_root = config["val_data_root"]
            logger.debug(f"キャリブレーションデータをconfigから取得: {calib_data_root}")
        else:
            logger.error(
                "--calib-data を指定するか, " "configにval_data_rootを設定してください"
            )
            return

        calib_data_path = Path(calib_data_root)
        if not calib_data_path.exists():
            logger.error(f"キャリブレーションデータが見つかりません: {calib_data_path}")
            return

        if "val_transform" not in config:
            logger.error(
                "configにval_transformが設定されていません. "
                "INT8キャリブレーションにはval_transformが必要です."
            )
            return
        transform = config["val_transform"]

        if input_shape is not None:
            calib_input_shape = input_shape
        else:
            try:
                import onnx

                onnx_model = onnx.load(str(onnx_path))
                input_tensor = onnx_model.graph.input[0]
                input_dims = input_tensor.type.tensor_type.shape.dim
                calib_input_shape = tuple(d.dim_value for d in input_dims[1:])
                logger.debug(f"ONNX入力形状: {calib_input_shape}")
            except Exception as e:
                logger.error(f"ONNXモデルから入力形状を取得できません: {e}")
                return

        cache_file = str(output_path.with_suffix(".cache"))

        max_calib_samples = args.calib_samples

        logger.info(
            f"キャリブレーション設定: "
            f"データ={calib_data_root}, "
            f"最大サンプル数={max_calib_samples}"
        )

        try:
            calibrator = create_int8_calibrator(
                data_root=calib_data_root,
                transform=transform,
                input_shape=calib_input_shape,
                batch_size=args.calib_batch_size,
                max_samples=max_calib_samples,
                cache_file=cache_file,
            )
        except Exception as e:
            logger.error(f"キャリブレータ作成エラー: {e}")
            return

    try:
        converter = TensorRTConverter(
            onnx_path=onnx_path,
            workspace_size=args.workspace_size,
        )
        engine_path = converter.convert(
            output_path=output_path,
            precision=precision,
            calibrator=calibrator,
            input_shape=input_shape,
        )
        logger.info(f"変換完了: {engine_path}")

    except Exception as e:
        logger.error(f"TensorRT変換エラー: {e}")
        return

    logger.info("推論するには:")
    logger.info(f"  uv run infer-trt {engine_path}")


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="pochitrain - 統合CLI（訓練・推論・最適化・変換）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  訓練
  uv run pochi train --config configs/pochi_train_config.py

  推論（基本）
  uv run pochi infer
    -m work_dirs/20250813_003/models/best_epoch40.pth
    -d data/val
    -c work_dirs/20250813_003/config.py

  推論（カスタム出力先）
  uv run pochi infer
    --model-path work_dirs/20250813_003/models/best_epoch40.pth
    --data data/test
    --config-path work_dirs/20250813_003/config.py
    --output custom_results

  ハイパーパラメータ最適化
  uv run pochi optimize --config configs/pochi_train_config.py

  TensorRT変換（INT8量子化）
  uv run pochi convert model.onnx --int8

  TensorRT変換（FP16）
  uv run pochi convert model.onnx --fp16

  TensorRT変換（キャリブレーションデータ指定）
  uv run pochi convert model.onnx --int8 --calib-data data/val
        """,
    )

    parser.add_argument("--debug", action="store_true", help="DEBUGログを有効化")

    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    train_parser = subparsers.add_parser("train", help="モデル訓練")
    train_parser.add_argument(
        "--config",
        default="configs/pochi_train_config.py",
        help="設定ファイルパス (default: configs/pochi_train_config.py)",
    )

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
    infer_parser.add_argument(
        "--pipeline",
        choices=("auto", "current", "fast", "gpu"),
        default="current",
        help="前処理パイプライン. 現在は current のみ実行されます.",
    )
    infer_parser.add_argument(
        "--benchmark-json",
        action="store_true",
        help="ベンチマーク結果を benchmark_result.json として出力する",
    )
    infer_parser.add_argument(
        "--benchmark-env-name",
        default=None,
        help="ベンチマーク結果の環境ラベル（省略時は自動決定）",
    )

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

    convert_parser = subparsers.add_parser(
        "convert", help="ONNXモデルをTensorRTエンジンに変換"
    )
    convert_parser.add_argument("onnx_path", help="ONNXモデルファイルパス (.onnx)")
    convert_parser.add_argument(
        "--fp16",
        action="store_true",
        help="FP16精度で変換",
    )
    convert_parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8精度で変換 (キャリブレーションが必要)",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        help="出力エンジンファイルパス (default: 入力ファイルと同じ場所に.engine拡張子)",
    )
    convert_parser.add_argument(
        "--config-path",
        "-c",
        help="設定ファイルパス (INT8時にval_transformとデータパスを取得, "
        "省略時はONNXパスから自動検出)",
    )
    convert_parser.add_argument(
        "--calib-data",
        help="キャリブレーションデータディレクトリ "
        "(INT8時に使用, 省略時はconfigのval_data_rootを使用)",
    )
    convert_parser.add_argument(
        "--input-size",
        nargs=2,
        type=positive_int,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ (動的シェイプONNXモデルの変換時に必要, 1以上)",
    )
    convert_parser.add_argument(
        "--calib-samples",
        type=positive_int,
        default=500,
        help="キャリブレーションサンプル数 (default: 500, 1以上)",
    )
    convert_parser.add_argument(
        "--calib-batch-size",
        type=positive_int,
        default=1,
        help="キャリブレーションバッチサイズ (default: 1, 1以上)",
    )
    convert_parser.add_argument(
        "--workspace-size",
        type=positive_int,
        default=1 << 30,
        help="TensorRTワークスペースサイズ (bytes, default: 1GB, 1以上)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "optimize":
        optimize_command(args)
    elif args.command == "convert":
        convert_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
