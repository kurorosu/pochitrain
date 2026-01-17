#!/usr/bin/env python
r"""Optunaを使ったハイパーパラメータ最適化スクリプト.

使用例:
    python tools/optimize_hyperparams.py \
        --config configs/pochi_train_config.py \
        --output work_dirs/optuna_results

設定ファイルには訓練設定とOptuna設定の両方を含める必要があります.
Optuna設定項目: study_name, direction, n_trials, search_space など.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str) -> dict[str, Any]:
    """設定ファイルを読み込む.

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location("config", path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load config: {config_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # モジュールの属性を辞書に変換
    config = {}
    for key in dir(module):
        if not key.startswith("_"):
            config[key] = getattr(module, key)

    return config


def create_data_loaders(config: dict[str, Any]) -> tuple:
    """設定からデータローダーを作成する.

    Args:
        config: 設定辞書

    Returns:
        (train_loader, val_loader, classes) のタプル
    """
    from pochitrain import create_data_loaders

    return create_data_loaders(
        train_root=config.get("train_data_root", "data/train"),
        val_root=config.get("val_data_root", "data/val"),
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 0),
        train_transform=config.get("train_transform"),
        val_transform=config.get("val_transform"),
    )


def run_optimization(
    config: dict[str, Any],
    output_dir: str,
) -> None:
    """ハイパーパラメータ最適化を実行する.

    Args:
        config: 訓練設定とOptuna設定を含む統合設定
        output_dir: 出力ディレクトリ
    """
    from pochitrain.optimization import (
        ClassificationObjective,
        DefaultParamSuggestor,
        JsonResultExporter,
        OptunaStudyManager,
    )
    from pochitrain.optimization.result_exporter import ConfigExporter

    # データローダーを作成
    print("Loading data...")
    train_loader, val_loader, classes = create_data_loaders(config)
    config["num_classes"] = len(classes)
    print(f"  Classes: {classes}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # パラメータサジェスターを作成
    search_space = config.get("search_space", {})
    param_suggestor = DefaultParamSuggestor(search_space)

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
    print("\nCreating Optuna study...")
    study = study_manager.create_study(
        study_name=config.get("study_name", "pochitrain_optimization"),
        direction=config.get("direction", "maximize"),
        sampler=config.get("sampler", "TPESampler"),
        pruner=config.get("pruner"),
    )
    print(f"  Study name: {study.study_name}")
    print(f"  Direction: {config.get('direction', 'maximize')}")
    print(f"  Sampler: {config.get('sampler', 'TPESampler')}")

    # 最適化を実行
    n_trials = config.get("n_trials", 20)
    print(f"\nStarting optimization ({n_trials} trials)...")
    study_manager.optimize(
        objective=objective,
        n_trials=n_trials,
        n_jobs=config.get("n_jobs", 1),
    )

    # 結果を取得
    best_params = study_manager.get_best_params()
    best_value = study_manager.get_best_value()

    print("\n" + "=" * 50)
    print("Optimization completed!")
    print("=" * 50)
    print(f"Best value: {best_value:.4f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # 結果をエクスポート
    print(f"\nExporting results to: {output_dir}")

    # JSON形式でエクスポート
    json_exporter = JsonResultExporter()
    json_exporter.export(best_params, best_value, study, output_dir)

    # Python設定ファイル形式でエクスポート
    config_exporter = ConfigExporter(config)
    config_exporter.export(best_params, best_value, study, output_dir)

    print("\nGenerated files:")
    print(f"  - {output_dir}/best_params.json")
    print(f"  - {output_dir}/trials_history.json")
    print(f"  - {output_dir}/optimized_config.py")
    print("\nTo train with optimized parameters:")
    print(f"  python pochi.py train --config {output_dir}/optimized_config.py")


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="Optunaを使ったハイパーパラメータ最適化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="設定ファイル (訓練設定+Optuna設定を含む)",
    )
    parser.add_argument(
        "--output",
        default="work_dirs/optuna_results",
        help="結果出力ディレクトリ (デフォルト: work_dirs/optuna_results)",
    )

    args = parser.parse_args()

    # 設定を読み込み
    print("Loading configuration...")
    config = load_config(args.config)

    # 最適化を実行
    run_optimization(config, args.output)


if __name__ == "__main__":
    main()
