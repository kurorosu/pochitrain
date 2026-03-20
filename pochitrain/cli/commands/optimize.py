"""optimize サブコマンドの実装."""

import argparse
from pathlib import Path
from typing import Sized, cast

from pydantic import ValidationError

from pochitrain import (
    PochiConfig,
    create_data_loaders,
)
from pochitrain.cli.cli_commons import setup_logging
from pochitrain.utils import ConfigLoader


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
