"""TensorRTエンジンコンバーターモジュール.

ONNXモデルからTensorRTエンジンを生成する.
FP32, FP16, INT8の各精度モードに対応.
"""

import logging
from pathlib import Path
from typing import Optional

from pochitrain.logging import LoggerManager
from pochitrain.tensorrt.inference import check_tensorrt_availability

logger: logging.Logger = LoggerManager().get_logger(__name__)


class TensorRTConverter:
    """ONNXモデルをTensorRTエンジンに変換するクラス.

    TensorRT Builder APIを使用して, ONNXモデルからFP32/FP16/INT8の
    TensorRTエンジンファイルを生成する.

    Attributes:
        onnx_path: ONNXモデルファイルパス
        workspace_size: TensorRTビルダーの最大ワークスペースサイズ (bytes)
    """

    def __init__(
        self,
        onnx_path: Path,
        workspace_size: int = 1 << 30,
    ) -> None:
        """TensorRTConverterを初期化.

        Args:
            onnx_path: ONNXモデルファイルパス
            workspace_size: 最大ワークスペースサイズ (デフォルト: 1GB)

        Raises:
            ImportError: TensorRTがインストールされていない場合
            FileNotFoundError: ONNXファイルが見つからない場合
        """
        if not check_tensorrt_availability():
            raise ImportError(
                "TensorRTがインストールされていません. "
                "TensorRT SDKをインストールしてください."
            )

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNXモデルが見つかりません: {self.onnx_path}")

        self.workspace_size = workspace_size

    def convert(
        self,
        output_path: Path,
        precision: str = "fp32",
        calibrator: Optional[object] = None,
    ) -> Path:
        """ONNXモデルをTensorRTエンジンに変換する.

        Args:
            output_path: 出力エンジンファイルパス (.engine)
            precision: 精度モード ("fp32", "fp16", "int8")
            calibrator: INT8キャリブレータ (precision="int8"の場合に必須)

        Returns:
            生成されたエンジンファイルパス

        Raises:
            ValueError: 無効なprecision指定, またはINT8でcalibratorが未指定の場合
            RuntimeError: エンジンのビルドに失敗した場合
        """
        import tensorrt as trt

        valid_precisions = ("fp32", "fp16", "int8")
        if precision not in valid_precisions:
            raise ValueError(
                f"無効なprecision: {precision}. " f"有効な値: {valid_precisions}"
            )

        if precision == "int8" and calibrator is None:
            raise ValueError(
                "INT8精度にはcalibratorが必須です. "
                "create_int8_calibrator()でキャリブレータを作成してください."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"TensorRTエンジンを生成中... (精度: {precision.upper()})")
        logger.debug(f"ONNX: {self.onnx_path}")
        logger.debug(f"出力: {output_path}")

        # TensorRTロガー・ビルダー・ネットワーク作成
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        # ONNXパーサーでモデルを読み込み
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse_from_file(str(self.onnx_path)):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError(f"ONNXパースエラー: {'; '.join(errors)}")

        logger.debug(
            f"ONNXパース完了: 入力数={network.num_inputs}, "
            f"出力数={network.num_outputs}"
        )

        # ビルド設定
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

        # 動的入力に対するOptimization Profileの設定
        # ONNXエクスポート時にdynamic_axesが設定されている場合に必要
        profile = builder.create_optimization_profile()
        has_dynamic = False
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            # 動的次元 (-1) があるかチェック
            if any(d == -1 for d in shape):
                has_dynamic = True
                # 動的次元をバッチサイズ1に固定
                static_shape = tuple(1 if d == -1 else d for d in shape)
                profile.set_shape(inp.name, static_shape, static_shape, static_shape)
                logger.debug(
                    f"動的入力を検出: {inp.name}, "
                    f"shape={tuple(shape)} -> {static_shape}"
                )
        if has_dynamic:
            config.add_optimization_profile(profile)

        # 精度フラグの設定
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            logger.debug("FP16モードを有効化")
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # INT8非対応層のフォールバック
            config.int8_calibrator = calibrator
            logger.debug("INT8モードを有効化 (FP16フォールバック付き)")

        # エンジンビルド
        logger.info("エンジンをビルド中 (数分かかる場合があります)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("TensorRTエンジンのビルドに失敗しました")

        # エンジンファイルを保存
        with open(output_path, "wb") as f:
            f.write(serialized_engine)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"TensorRTエンジン生成完了: {output_path}")
        logger.info(f"ファイルサイズ: {file_size_mb:.2f} MB")

        return output_path
