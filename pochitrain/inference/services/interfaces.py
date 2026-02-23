"""Service 層のインターフェース定義.

IExecutionService は `typing.Protocol` を採用する.
IInferenceOrchestrationService は共通実装を持たせるため `abc.ABC` を使う.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from torch.utils.data import DataLoader

from pochitrain.inference.pipeline_strategy import create_dataset_and_params
from pochitrain.inference.types.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    InferenceResolvedPaths,
    InferenceRunResult,
    InferenceRuntimeOptions,
    RuntimeExecutionRequest,
)
from pochitrain.inference.types.result_export_types import ResultExportRequest
from pochitrain.inference.types.runtime_adapter_protocol import IRuntimeAdapter
from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import (
    PochiImageDataset,
    create_scaled_normalize_tensors,
)
from pochitrain.utils import get_default_output_base_dir, validate_data_path
from pochitrain.utils.directory_manager import InferenceWorkspaceManager
from pochitrain.utils.inference_utils import log_inference_result

from .result_export_service import ResultExportService


class IExecutionService(Protocol):
    """推論実行ループを提供するサービスインターフェース."""

    def run(
        self,
        data_loader: DataLoader[Any],
        runtime: IRuntimeAdapter,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        """推論を実行して集計結果を返す.

        Args:
            data_loader: 推論対象DataLoader.
            runtime: 推論ランタイム差分を吸収するアダプタ.
            request: 実行パラメータ.

        Returns:
            推論集計結果.
        """
        ...


class IInferenceOrchestrationService(ABC):
    """推論サービスの共通基底インターフェース."""

    execution_service_factory: type[IExecutionService]

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """サービスを初期化する.

        Args:
            logger: ロガーインスタンス. 未指定時はモジュールロガーを利用する.
        """
        self.logger = logger or LoggerManager().get_logger(__name__)

    def resolve_paths(
        self,
        request: InferenceCliRequest,
        config: Dict[str, Any],
    ) -> InferenceResolvedPaths:
        """データパスと出力先を解決する.

        Args:
            request: CLI入力を格納した共通リクエスト.
            config: 設定辞書.

        Returns:
            解決済みのパス情報.

        Raises:
            ValueError: データパス解決に失敗した場合.
        """
        if request.data_path is not None:
            data_path = request.data_path
        elif "val_data_root" in config:
            data_path = Path(config["val_data_root"])
        else:
            raise ValueError(
                "--data を指定するか, configにval_data_rootを設定してください"
            )

        validate_data_path(data_path)

        if request.output_dir is not None:
            output_dir = request.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = get_default_output_base_dir(request.model_path)
            workspace_manager = InferenceWorkspaceManager(str(base_dir))
            output_dir = workspace_manager.create_workspace()

        return InferenceResolvedPaths(
            model_path=request.model_path,
            data_path=data_path,
            output_dir=output_dir,
        )

    def resolve_pipeline(self, requested: str, use_gpu: bool) -> str:
        """ランタイム既定のパイプライン名を解決する.

        Args:
            requested: ユーザー指定のパイプライン名.
            use_gpu: GPU が利用可能かどうか.

        Returns:
            解決後のパイプライン名.
        """
        if requested != "auto":
            if requested == "gpu" and not use_gpu:
                return "fast"
            return requested
        return "gpu" if use_gpu else "fast"

    def _resolve_batch_size(self, config: Dict[str, Any]) -> int:
        """ランタイム既定のバッチサイズを解決する.

        Args:
            config: 推論設定辞書.

        Returns:
            解決後のバッチサイズ.
        """
        return int(config.get("batch_size", 1))

    def resolve_runtime_options(
        self,
        config: Dict[str, Any],
        pipeline: str,
        use_gpu: bool = True,
    ) -> InferenceRuntimeOptions:
        """共通実行オプションを構築する.

        Args:
            config: 設定辞書.
            pipeline: 解決済みパイプライン名.
            use_gpu: 推論ランタイムがGPUを使うかどうか.

        Returns:
            推論実行オプション.
        """
        return InferenceRuntimeOptions(
            pipeline=pipeline,
            batch_size=self._resolve_batch_size(config),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", True)),
            use_gpu=use_gpu,
            use_gpu_pipeline=pipeline == "gpu",
        )

    def create_dataloader(
        self,
        config: Dict[str, Any],
        data_path: Path,
        val_transform: Any,
        pipeline: str,
        runtime_options: InferenceRuntimeOptions,
    ) -> tuple[
        DataLoader[Any],
        PochiImageDataset,
        str,
        Optional[List[float]],
        Optional[List[float]],
    ]:
        """推論用 DataLoader とデータセットを生成する.

        Args:
            config: 設定辞書.
            data_path: 推論データのディレクトリパス.
            val_transform: 検証用 transform.
            pipeline: 解決済みパイプライン名.
            runtime_options: 推論実行オプション.

        Returns:
            DataLoader, データセット, 解決後パイプライン名, mean, std.
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
        return loader, dataset, resolved_pipeline, norm_mean, norm_std

    def build_runtime_execution_request(
        self,
        data_loader: DataLoader[Any],
        runtime_adapter: IRuntimeAdapter,
        *,
        use_gpu_pipeline: bool,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        use_cuda_timing: bool = False,
        gpu_non_blocking: bool = True,
        warmup_repeats: int = 10,
        skip_measurement_batches: int = 1,
    ) -> RuntimeExecutionRequest:
        """実行コンテキストを構築する.

        Args:
            data_loader: 推論データローダー.
            runtime_adapter: 推論ランタイムアダプタ.
            use_gpu_pipeline: GPU前処理を使用するかどうか.
            norm_mean: 正規化平均.
            norm_std: 正規化標準偏差.
            use_cuda_timing: CUDA Event 計測を使うかどうか.
            gpu_non_blocking: GPU転送に non_blocking を使うかどうか.
            warmup_repeats: ウォームアップ回数.
            skip_measurement_batches: 計測除外する先頭バッチ数.

        Returns:
            実行コンテキスト.
        """
        mean_255 = None
        std_255 = None
        if use_gpu_pipeline:
            if norm_mean is None or norm_std is None:
                raise ValueError(
                    "gpu パイプラインには normalize パラメータが必要です.",
                )
            adapter_device = getattr(runtime_adapter, "device", "cuda")
            mean_255, std_255 = create_scaled_normalize_tensors(
                norm_mean,
                norm_std,
                device=adapter_device,
            )

        execution_request = ExecutionRequest(
            use_gpu_pipeline=use_gpu_pipeline,
            mean_255=mean_255,
            std_255=std_255,
            warmup_repeats=warmup_repeats,
            skip_measurement_batches=skip_measurement_batches,
            use_cuda_timing=use_cuda_timing,
            gpu_non_blocking=gpu_non_blocking,
        )
        return RuntimeExecutionRequest(
            data_loader=data_loader,
            runtime_adapter=runtime_adapter,
            execution_request=execution_request,
        )

    @abstractmethod
    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        """ランタイム入力形状から入力サイズを解決する."""

    def run(
        self,
        request: RuntimeExecutionRequest,
        execution_service: Optional[IExecutionService] = None,
    ) -> InferenceRunResult:
        """推論を実行し共通結果型へ集約する.

        Args:
            request: 実行コンテキスト.
            execution_service: 実行サービス. 未指定時は既定実装を生成する.

        Returns:
            ランタイム横断で共通利用する推論結果.
        """
        service = execution_service or self.execution_service_factory()
        execution_result = service.run(
            data_loader=request.data_loader,
            runtime=request.runtime_adapter,
            request=request.execution_request,
        )
        return InferenceRunResult.from_execution_result(execution_result)

    def aggregate_and_export(
        self,
        *,
        workspace_dir: Path,
        model_path: Path,
        data_path: Path,
        dataset: PochiImageDataset,
        run_result: InferenceRunResult,
        input_size: Optional[tuple[int, int, int]],
        model_info: Optional[Dict[str, Any]],
        cm_config: Optional[Dict[str, Any]],
        results_filename: str = "inference_results.csv",
        summary_filename: str = "inference_summary.txt",
        extra_info: Optional[Dict[str, Any]] = None,
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
            results_filename: 結果CSVファイル名.
            summary_filename: サマリーファイル名.
            extra_info: サマリー追記情報.
        """
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
                image_paths=dataset.get_file_paths(),
                predictions=run_result.predictions,
                true_labels=run_result.true_labels,
                confidences=run_result.confidences,
                class_names=dataset.get_classes(),
                num_samples=run_result.num_samples,
                correct=run_result.correct,
                avg_time_per_image=run_result.avg_time_per_image,
                total_samples=run_result.total_samples,
                warmup_samples=run_result.warmup_samples,
                avg_total_time_per_image=run_result.avg_total_time_per_image,
                input_size=input_size,
                results_filename=results_filename,
                summary_filename=summary_filename,
                extra_info=extra_info,
                model_info=model_info,
                cm_config=cm_config,
            )
        )
