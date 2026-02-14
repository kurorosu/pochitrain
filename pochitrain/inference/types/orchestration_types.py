"""推論CLIオーケストレーションで共有する型定義."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class InferenceCliRequest:
    """推論CLIからオーケストレーション層へ渡す共通入力.

    Args:
        model_path: モデルファイルパス.
        data_path: 推論データディレクトリ. 未指定時は設定ファイルから解決する.
        output_dir: 結果出力ディレクトリ. 未指定時はモデル位置から自動解決する.
        requested_pipeline: ユーザー指定のパイプライン名.
    """

    model_path: Path
    data_path: Optional[Path]
    output_dir: Optional[Path]
    requested_pipeline: str


@dataclass(frozen=True)
class InferenceResolvedPaths:
    """推論前に解決済みのパス群.

    Args:
        model_path: モデルファイルパス.
        data_path: 推論データディレクトリ.
        output_dir: 結果出力ディレクトリ.
    """

    model_path: Path
    data_path: Path
    output_dir: Path


@dataclass(frozen=True)
class InferenceRuntimeOptions:
    """推論実行時の共通オプション.

    Args:
        pipeline: 解決後のパイプライン名.
        batch_size: DataLoaderのバッチサイズ.
        num_workers: DataLoaderのワーカー数.
        pin_memory: DataLoaderのpin_memory設定.
        use_gpu: 推論ランタイムがGPUを使うかどうか.
        use_gpu_pipeline: 前処理パイプラインがGPUを使うかどうか.
    """

    pipeline: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_gpu: bool
    use_gpu_pipeline: bool
