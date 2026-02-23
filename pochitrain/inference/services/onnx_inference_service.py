"""ONNX推論CLI向けのオーケストレーション補助サービス."""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from pochitrain.inference.types.execution_types import ExecutionResult
from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    InferenceResolvedPaths,
    InferenceRunResult,
    InferenceRuntimeOptions,
    RuntimeExecutionRequest,
)
from pochitrain.utils import (
    get_default_output_base_dir,
    validate_data_path,
)
from pochitrain.utils.directory_manager import InferenceWorkspaceManager

from .execution_service import ExecutionService


class _ExecutionServiceLike(Protocol):
    def run(self, data_loader: Any, runtime: Any, request: Any) -> ExecutionResult:
        """実行結果を返す."""
        ...


class OnnxInferenceService:
    """ONNX推論CLIで必要な解決処理を提供するサービス."""

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

    def resolve_pipeline(
        self,
        requested: str,
        use_gpu: bool,
    ) -> str:
        """ONNX推論で実際に使うパイプライン名を解決する.

        Args:
            requested: ユーザー指定パイプライン.
            use_gpu: ONNX推論がGPUを使うかどうか.

        Returns:
            解決後パイプライン名.
        """
        if requested != "auto":
            if requested == "gpu" and not use_gpu:
                return "fast"
            return requested

        return "gpu" if use_gpu else "fast"

    def resolve_runtime_options(
        self,
        config: Dict[str, Any],
        pipeline: str,
        use_gpu: bool,
    ) -> InferenceRuntimeOptions:
        """ONNX推論向け実行オプションを構築する.

        Args:
            config: 設定辞書.
            pipeline: 解決済みパイプライン名.
            use_gpu: ONNX推論がGPUを使うかどうか.

        Returns:
            推論実行オプション.
        """
        return InferenceRuntimeOptions(
            pipeline=pipeline,
            batch_size=int(config.get("batch_size", 1)),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", True)),
            use_gpu=use_gpu,
            use_gpu_pipeline=pipeline == "gpu",
        )

    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        """ONNX入力形状から入力サイズを解決する.

        Args:
            shape: ONNXの入力shape.

        Returns:
            入力サイズ (C, H, W). 解決できない場合はNone.
        """
        if len(shape) != 4:
            return None
        c = shape[1] if isinstance(shape[1], int) else 3
        h = shape[2] if isinstance(shape[2], int) else None
        w = shape[3] if isinstance(shape[3], int) else None
        if h and w:
            return (c, h, w)
        return None

    def run(
        self,
        request: RuntimeExecutionRequest,
        execution_service: Optional[_ExecutionServiceLike] = None,
    ) -> InferenceRunResult:
        """推論を実行し共通結果型へ集約する.

        Args:
            request: 実行コンテキスト.
            execution_service: 実行サービス. 未指定時は内部で生成.

        Returns:
            ランタイム横断で共通利用する推論結果.
        """
        service = execution_service or ExecutionService()
        execution_result = service.run(
            data_loader=request.data_loader,
            runtime=request.runtime_adapter,
            request=request.execution_request,
        )
        return InferenceRunResult.from_execution_result(execution_result)
