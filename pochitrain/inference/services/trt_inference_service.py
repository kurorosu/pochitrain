"""TensorRT推論CLI向けのオーケストレーション補助サービス."""

from pathlib import Path
from typing import Any, Dict, Optional

from pochitrain.inference.types.orchestration_types import (
    InferenceCliRequest,
    InferenceResolvedPaths,
    InferenceRuntimeOptions,
)
from pochitrain.utils import (
    get_default_output_base_dir,
    validate_data_path,
)
from pochitrain.utils.directory_manager import InferenceWorkspaceManager


class TensorRTInferenceService:
    """TensorRT推論CLIで必要な解決処理を提供するサービス."""

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

    def resolve_pipeline(self, requested: str) -> str:
        """TensorRT推論で実際に使うパイプライン名を解決する.

        Args:
            requested: ユーザー指定パイプライン.

        Returns:
            解決後パイプライン名.
        """
        if requested == "auto":
            return "gpu"
        return requested

    def resolve_runtime_options(
        self,
        config: Dict[str, Any],
        pipeline: str,
    ) -> InferenceRuntimeOptions:
        """TensorRT推論向け実行オプションを構築する.

        Args:
            config: 設定辞書.
            pipeline: 解決済みパイプライン名.

        Returns:
            推論実行オプション.
        """
        return InferenceRuntimeOptions(
            pipeline=pipeline,
            batch_size=1,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", True)),
            use_gpu=True,
            use_gpu_pipeline=pipeline == "gpu",
        )

    def resolve_input_size(self, shape: Any) -> Optional[tuple[int, int, int]]:
        """TensorRT入力形状から入力サイズを解決する.

        Args:
            shape: TensorRT入力shape.

        Returns:
            入力サイズ (C, H, W). 解決できない場合はNone.
        """
        if len(shape) != 4:
            return None
        return (shape[1], shape[2], shape[3])
