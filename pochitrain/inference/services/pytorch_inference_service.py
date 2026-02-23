"""PyTorch モデル推論のオーケストレーションサービス."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from pochitrain.config import PochiConfig
from pochitrain.inference.adapters import PyTorchRuntimeAdapter
from pochitrain.inference.pipeline_strategy import create_dataset_and_params
from pochitrain.inference.services.interfaces import (
    IExecutionService,
    IInferenceOrchestrationService,
)
from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import PochiImageDataset, create_scaled_normalize_tensors
from pochitrain.pochi_predictor import PochiPredictor
from pochitrain.utils import log_inference_result

from ..types.execution_types import ExecutionRequest
from ..types.orchestration_types import (
    InferenceRunResult,
    RuntimeExecutionRequest,
)
from ..types.result_export_types import ResultExportRequest
from .execution_service import ExecutionService
from .result_export_service import ResultExportService


class PyTorchInferenceService(IInferenceOrchestrationService):
    """PyTorch モデル推論の実行・集約・エクスポートを担うサービス.

    CLI から推論ビジネスロジックを分離し, 単体テストを可能にする.
    """

    execution_service_factory = ExecutionService

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        self.logger = logger or LoggerManager().get_logger(__name__)

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

    def build_runtime_execution_request(
        self,
        predictor: PochiPredictor,
        val_loader: DataLoader[Any],
        *,
        use_gpu_pipeline: bool = False,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        gpu_non_blocking: bool = True,
    ) -> RuntimeExecutionRequest:
        """PyTorch推論の実行リクエストを構築する.

        Args:
            predictor: 推論器.
            val_loader: 推論データローダー.

        Returns:
            実行コンテキスト.
        """
        runtime_adapter = PyTorchRuntimeAdapter(predictor)
        mean_255 = None
        std_255 = None
        if use_gpu_pipeline:
            if norm_mean is None or norm_std is None:
                raise ValueError(
                    "gpu パイプラインには normalize パラメータが必要です.",
                )
            mean_255, std_255 = create_scaled_normalize_tensors(
                norm_mean,
                norm_std,
                device=runtime_adapter.device,
            )
        execution_request = ExecutionRequest(
            use_gpu_pipeline=use_gpu_pipeline,
            mean_255=mean_255,
            std_255=std_255,
            use_cuda_timing=runtime_adapter.use_cuda_timing,
            gpu_non_blocking=gpu_non_blocking,
        )
        return RuntimeExecutionRequest(
            data_loader=val_loader,
            runtime_adapter=runtime_adapter,
            execution_request=execution_request,
        )

    def create_predictor(self, config: PochiConfig, model_path: Path) -> PochiPredictor:
        """推論器を生成する.

        Args:
            config: アプリケーション設定.
            model_path: 学習済みモデルのパス.

        Returns:
            初期化済みの推論器.
        """
        return PochiPredictor.from_config(config, str(model_path))

    def create_dataloader(
        self,
        config: PochiConfig,
        data_path: Path,
        *,
        pipeline: str = "current",
        pin_memory: bool = True,
    ) -> Tuple[
        DataLoader[Any],
        PochiImageDataset,
        str,
        Optional[List[float]],
        Optional[List[float]],
    ]:
        """推論用 DataLoader とデータセットを生成する.

        Args:
            config: アプリケーション設定.
            data_path: 推論データのディレクトリパス.
            pin_memory: DataLoader の pin_memory 設定.

        Returns:
            (DataLoader, PochiImageDataset) のタプル.
        """
        dataset, resolved_pipeline, norm_mean, norm_std = create_dataset_and_params(
            pipeline,
            data_path,
            config.val_transform,
        )
        loader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
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

    def run_inference(
        self,
        predictor: PochiPredictor,
        val_loader: DataLoader[Any],
        execution_service: Optional[IExecutionService] = None,
    ) -> InferenceRunResult:
        """推論を実行し, 結果と計測情報を返す.

        Args:
            predictor: 推論器.
            val_loader: 推論データローダー.
            execution_service: 実行サービス. 未指定時は内部で生成.

        Returns:
            ランタイム横断で共通利用する推論結果.
        """
        self.logger.info("推論を開始します...")
        runtime_request = self.build_runtime_execution_request(
            predictor=predictor,
            val_loader=val_loader,
        )
        return self.run(
            request=runtime_request,
            execution_service=execution_service,
        )

    def aggregate_and_export(
        self,
        *,
        workspace_dir: Path,
        model_path: Path,
        data_path: Path,
        dataset: PochiImageDataset,
        run_result: InferenceRunResult,
        input_size: Optional[Tuple[int, int, int]],
        model_info: Optional[Dict[str, Any]],
        cm_config: Optional[Dict[str, Any]],
    ) -> None:
        """精度計算・ログ出力・結果エクスポートを実行する.

        Args:
            workspace_dir: 結果出力先ディレクトリ.
            model_path: モデルファイルパス.
            data_path: 推論データパス.
            dataset: 推論データセット.
            run_result: ランタイム横断で共通利用する推論結果.
            input_size: 入力サイズ (C, H, W).
            model_info: モデル情報辞書.
            cm_config: 混同行列可視化設定.
        """
        image_paths = dataset.get_file_paths()
        class_names = dataset.get_classes()

        log_inference_result(
            num_samples=run_result.num_samples,
            correct=run_result.correct,
            avg_time_per_image=run_result.avg_time_per_image,
            total_samples=run_result.total_samples,
            warmup_samples=run_result.warmup_samples,
            avg_total_time_per_image=run_result.avg_total_time_per_image,
            input_size=input_size,
        )

        export_service = ResultExportService(self.logger)
        export_service.export(
            ResultExportRequest(
                output_dir=workspace_dir,
                model_path=model_path,
                data_path=data_path,
                image_paths=image_paths,
                predictions=run_result.predictions,
                true_labels=run_result.true_labels,
                confidences=run_result.confidences,
                class_names=class_names,
                num_samples=run_result.num_samples,
                correct=run_result.correct,
                avg_time_per_image=run_result.avg_time_per_image,
                total_samples=run_result.total_samples,
                warmup_samples=run_result.warmup_samples,
                avg_total_time_per_image=run_result.avg_total_time_per_image,
                input_size=input_size,
                results_filename="pytorch_inference_results.csv",
                summary_filename="pytorch_inference_summary.txt",
                model_info=model_info,
                cm_config=cm_config,
            )
        )
