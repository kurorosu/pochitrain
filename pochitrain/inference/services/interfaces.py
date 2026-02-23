"""Service 層のインターフェース定義.

IExecutionService は `typing.Protocol` を採用する.
IInferenceOrchestrationService は共通実装を持たせるため `abc.ABC` を使う.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from torch.utils.data import DataLoader

from pochitrain.inference.types.execution_types import ExecutionRequest, ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    InferenceResolvedPaths,
    InferenceRunResult,
    InferenceRuntimeOptions,
    RuntimeExecutionRequest,
)
from pochitrain.inference.types.runtime_adapter_protocol import IRuntimeAdapter
from pochitrain.utils import get_default_output_base_dir, validate_data_path
from pochitrain.utils.directory_manager import InferenceWorkspaceManager


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
