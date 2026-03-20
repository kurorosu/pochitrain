"""INT8 キャリブレーション設定の組み立て."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pochitrain.tensorrt.input_shape_resolver import InputShapeResolver
from pochitrain.utils import ConfigLoader, load_config_auto


@dataclass
class INT8CalibrationConfig:
    """INT8 キャリブレーションに必要な設定."""

    calib_data_root: str
    transform: Any
    input_shape: tuple[int, ...]
    batch_size: int
    max_samples: int
    cache_file: str


class INT8CalibrationConfigurer:
    """INT8 キャリブレーション設定を CLI 引数と config から組み立てる."""

    def __init__(self, logger: logging.Logger) -> None:
        """INT8CalibrationConfigurer を初期化.

        Args:
            logger: ロガー.
        """
        self._logger = logger
        self._shape_resolver = InputShapeResolver(logger)

    def configure(
        self,
        *,
        config_path: Optional[str],
        calib_data: Optional[str],
        input_shape: Optional[tuple[int, ...]],
        onnx_path: Path,
        output_path: Path,
        calib_samples: int,
        calib_batch_size: int,
    ) -> INT8CalibrationConfig:
        """INT8 キャリブレーション設定を組み立てる.

        Args:
            config_path: CLI で指定された設定ファイルパス.
            calib_data: CLI で指定されたキャリブレーションデータパス.
            input_shape: 解決済みの入力形状 (C, H, W). None なら ONNX から取得.
            onnx_path: ONNX モデルファイルパス.
            output_path: 出力エンジンファイルパス (キャッシュファイル名生成用).
            calib_samples: キャリブレーションサンプル数.
            calib_batch_size: キャリブレーションバッチサイズ.

        Returns:
            INT8CalibrationConfig.

        Raises:
            ValueError: 設定が不足している場合.
            FileNotFoundError: データパスが存在しない場合.
            RuntimeError: 設定ファイルの読み込みに失敗した場合.
        """
        config = self._load_config(config_path, onnx_path)
        calib_data_root = self._resolve_calib_data(calib_data, config)
        transform = self._resolve_transform(config)
        calib_input_shape = self._resolve_input_shape(input_shape, onnx_path)

        return INT8CalibrationConfig(
            calib_data_root=calib_data_root,
            transform=transform,
            input_shape=calib_input_shape,
            batch_size=calib_batch_size,
            max_samples=calib_samples,
            cache_file=str(output_path.with_suffix(".cache")),
        )

    def _load_config(
        self, config_path: Optional[str], onnx_path: Path
    ) -> dict[str, Any]:
        """設定ファイルを読み込む."""
        if config_path:
            try:
                config = ConfigLoader.load_config(config_path)
                self._logger.debug(f"設定ファイルを読み込み: {config_path}")
                return config
            except Exception as e:
                raise RuntimeError(f"設定ファイル読み込みエラー: {e}") from e
        else:
            return load_config_auto(onnx_path)

    def _resolve_calib_data(
        self, calib_data: Optional[str], config: dict[str, Any]
    ) -> str:
        """キャリブレーションデータパスを解決する."""
        if calib_data:
            calib_data_root = calib_data
        elif config.get("val_data_root"):
            calib_data_root = config["val_data_root"]
            self._logger.debug(
                f"キャリブレーションデータをconfigから取得: {calib_data_root}"
            )
        else:
            raise ValueError(
                "--calib-data を指定するか, configにval_data_rootを設定してください"
            )

        if not Path(calib_data_root).exists():
            raise FileNotFoundError(
                f"キャリブレーションデータが見つかりません: {calib_data_root}"
            )
        return calib_data_root

    def _resolve_transform(self, config: dict[str, Any]) -> Any:
        """キャリブレーション用 transform を取得する."""
        if "val_transform" not in config:
            raise ValueError(
                "configにval_transformが設定されていません. "
                "INT8キャリブレーションにはval_transformが必要です."
            )
        return config["val_transform"]

    def _resolve_input_shape(
        self,
        input_shape: Optional[tuple[int, ...]],
        onnx_path: Path,
    ) -> tuple[int, ...]:
        """キャリブレーション用入力形状を解決する."""
        if input_shape is not None:
            return input_shape
        return self._shape_resolver.extract_static_shape(onnx_path)
