"""PyTorch モデル推論のオーケストレーションサービス."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from pochitrain.config import PochiConfig
from pochitrain.inference.adapters import PyTorchRuntimeAdapter
from pochitrain.inference.pipeline_strategy import create_dataset_and_params
from pochitrain.inference.services.interfaces import IInferenceService
from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import PochiImageDataset
from pochitrain.pochi_predictor import PochiPredictor

from ..types.orchestration_types import (
    InferenceRuntimeOptions,
)
from .execution_service import ExecutionService


class PyTorchInferenceService(IInferenceService):
    """PyTorch モデル推論の実行・集約・エクスポートを担うサービス.

    CLI から推論ビジネスロジックを分離し, 単体テストを可能にする.
    """

    execution_service_factory = ExecutionService

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        super().__init__(logger=logger or LoggerManager().get_logger(__name__))

    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        """入力形状から入力サイズを解決する.

        Args:
            shape: 入力shape.

        Returns:
            入力サイズ (C, H, W). 解決できない場合はNone.
        """
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            return None
        if not all(isinstance(v, int) for v in shape[1:]):
            return None
        return (shape[1], shape[2], shape[3])

    def create_predictor(self, config: PochiConfig, model_path: Path) -> PochiPredictor:
        """推論器を生成する.

        Args:
            config: アプリケーション設定.
            model_path: 学習済みモデルのパス.

        Returns:
            初期化済みの推論器.
        """
        return PochiPredictor.from_config(config, str(model_path))

    def create_runtime_adapter(
        self, predictor: PochiPredictor
    ) -> PyTorchRuntimeAdapter:
        """PyTorch推論インスタンスからランタイムアダプタを作成する.

        Args:
            predictor: 推論器.

        Returns:
            PyTorchランタイムアダプタ.
        """
        return PyTorchRuntimeAdapter(predictor)

    def create_dataloader(
        self,
        config: Dict[str, Any],
        data_path: Path,
        val_transform: Any,
        pipeline: str,
        runtime_options: InferenceRuntimeOptions,
    ) -> Tuple[
        DataLoader[Any],
        PochiImageDataset,
        str,
        Optional[List[float]],
        Optional[List[float]],
    ]:
        """推論用 DataLoader とデータセットを生成する.

        Args:
            config: 推論設定辞書.
            data_path: 推論データのディレクトリパス.
            val_transform: 検証用 transform.
            pipeline: 解決済みパイプライン名.
            runtime_options: 推論実行オプション.

        Returns:
            (DataLoader, PochiImageDataset) のタプル.
        """
        _ = config
        dataset, resolved_pipeline, norm_mean, norm_std = create_dataset_and_params(
            pipeline,
            data_path,
            val_transform,
        )
        loader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=runtime_options.batch_size,
            shuffle=False,
            num_workers=runtime_options.num_workers,
            pin_memory=runtime_options.pin_memory,
        )

        self.logger.debug("使用されたTransform (設定ファイルから):")
        if dataset.transform is not None and hasattr(dataset.transform, "transforms"):
            for i, transform in enumerate(dataset.transform.transforms):
                self.logger.debug(f"   {i + 1}. {transform}")

        return loader, dataset, resolved_pipeline, norm_mean, norm_std

    def detect_input_size(
        self, config: PochiConfig, dataset: PochiImageDataset
    ) -> Optional[Tuple[int, int, int]]:
        """Transform またはデータセットから入力サイズを推定する.

        Args:
            config: アプリケーション設定.
            dataset: 推論データセット.

        Returns:
            (C, H, W) のタプル. 取得できない場合は None.
        """
        try:
            from torchvision.transforms import CenterCrop, RandomResizedCrop, Resize

            for t in config.val_transform.transforms:
                if isinstance(t, (Resize, CenterCrop, RandomResizedCrop)):
                    size = getattr(t, "size", None)
                    if size:
                        if isinstance(size, int):
                            return (3, size, size)
                        elif isinstance(size, (list, tuple)):
                            return (3, size[0], size[1])
                        break

            if len(dataset) > 0:
                sample_img, _ = dataset[0]
                if hasattr(sample_img, "shape") and len(sample_img.shape) == 3:
                    s = sample_img.shape
                    return (int(s[0]), int(s[1]), int(s[2]))
        except Exception:
            pass

        return None
