"""推論パイプライン選択に応じたデータセット生成モジュール."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import (
    FastInferenceDataset,
    GpuInferenceDataset,
    PochiImageDataset,
    build_gpu_preprocess_transform,
    convert_transform_for_fast_inference,
    extract_normalize_params,
)

logger = LoggerManager().get_logger(__name__)


@dataclass(frozen=True)
class DatasetBuildResult:
    """データセット生成結果."""

    dataset: PochiImageDataset
    pipeline: str
    mean: Optional[List[float]]
    std: Optional[List[float]]


class IPipelineStrategy(Protocol):
    """パイプライン処理インターフェース."""

    def build(self, data_path: Path, val_transform: Any) -> DatasetBuildResult:
        """パイプライン戦略に応じたデータセットを生成する.

        Args:
            data_path: 推論対象データディレクトリ.
            val_transform: 検証用 transform.

        Returns:
            生成したデータセットと付随情報.
        """


class CurrentPipelineStrategy:
    """current パイプライン処理."""

    def build(self, data_path: Path, val_transform: Any) -> DatasetBuildResult:
        """Current パイプライン用のデータセットを生成する.

        Args:
            data_path: 推論対象データディレクトリ.
            val_transform: 検証用 transform.

        Returns:
            current パイプラインのデータセット生成結果.
        """
        dataset = PochiImageDataset(str(data_path), transform=val_transform)
        return DatasetBuildResult(
            dataset=dataset, pipeline="current", mean=None, std=None
        )


class FastPipelineStrategy:
    """fast パイプライン処理."""

    def build(self, data_path: Path, val_transform: Any) -> DatasetBuildResult:
        """Fast パイプライン用のデータセットを生成する.

        Args:
            data_path: 推論対象データディレクトリ.
            val_transform: 検証用 transform.

        Returns:
            fast パイプライン, またはフォールバック先の生成結果.
        """
        fast_transform = convert_transform_for_fast_inference(val_transform)
        if fast_transform is not None:
            dataset = FastInferenceDataset(str(data_path), transform=fast_transform)
            return DatasetBuildResult(
                dataset=dataset, pipeline="fast", mean=None, std=None
            )

        logger.info("PochiImageDatasetにフォールバックします")
        dataset = PochiImageDataset(str(data_path), transform=val_transform)
        return DatasetBuildResult(
            dataset=dataset, pipeline="current", mean=None, std=None
        )


class GpuPipelineStrategy:
    """gpu パイプライン処理."""

    def build(self, data_path: Path, val_transform: Any) -> DatasetBuildResult:
        """Gpu パイプライン用のデータセットを生成する.

        Args:
            data_path: 推論対象データディレクトリ.
            val_transform: 検証用 transform.

        Returns:
            gpu パイプライン, またはフォールバック先の生成結果.
        """
        gpu_transform = build_gpu_preprocess_transform(val_transform)
        if gpu_transform is None:
            logger.info(
                "PIL専用transformが含まれるため, currentパイプラインにフォールバックします"
            )
            dataset = PochiImageDataset(str(data_path), transform=val_transform)
            return DatasetBuildResult(
                dataset=dataset, pipeline="current", mean=None, std=None
            )

        try:
            mean, std = extract_normalize_params(val_transform)
        except ValueError:
            logger.warning(
                "Normalizeが見つからないため, fastパイプラインにフォールバックします"
            )
            fast_result = FastPipelineStrategy().build(data_path, val_transform)
            return DatasetBuildResult(
                dataset=fast_result.dataset,
                pipeline=fast_result.pipeline,
                mean=None,
                std=None,
            )

        dataset = GpuInferenceDataset(
            str(data_path),
            transform=gpu_transform if gpu_transform.transforms else None,
        )
        return DatasetBuildResult(dataset=dataset, pipeline="gpu", mean=mean, std=std)


class PipelineStrategyFactory:
    """パイプライン処理ファクトリー."""

    _strategy_map: dict[str, IPipelineStrategy] = {
        "current": CurrentPipelineStrategy(),
        "fast": FastPipelineStrategy(),
        "gpu": GpuPipelineStrategy(),
    }

    @classmethod
    def create(cls, pipeline: str) -> IPipelineStrategy:
        """パイプライン名から戦略を返す.

        Args:
            pipeline: パイプライン名.

        Returns:
            対応する戦略インスタンス.

        Raises:
            ValueError: 未対応のパイプライン名が指定された場合.
        """
        try:
            return cls._strategy_map[pipeline]
        except KeyError as exc:
            raise ValueError(f"Unsupported pipeline: {pipeline}") from exc


def create_dataset_and_params(
    pipeline: str,
    data_path: Path,
    val_transform: Any,
) -> Tuple[PochiImageDataset, str, Optional[List[float]], Optional[List[float]]]:
    """パイプラインに応じたデータセットと正規化パラメータを返す.

    Args:
        pipeline: パイプライン名.
        data_path: 推論対象データディレクトリ.
        val_transform: 検証用 transform.

    Returns:
        データセット, 実際に適用されたパイプライン名, mean, std.
    """
    strategy = PipelineStrategyFactory.create(pipeline)
    result = strategy.build(data_path, val_transform)
    return result.dataset, result.pipeline, result.mean, result.std
