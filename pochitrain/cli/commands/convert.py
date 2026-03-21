"""convert サブコマンドの実装."""

import argparse
from pathlib import Path

from pochitrain.cli.cli_commons import setup_logging


def convert_command(args: argparse.Namespace) -> None:
    """TensorRTエンジン変換サブコマンドの実行."""
    logger = setup_logging(debug=args.debug)
    logger.debug("=== pochitrain TensorRT変換モード ===")

    try:
        from pochitrain.tensorrt.converter import TensorRTConverter
        from pochitrain.tensorrt.inference import check_tensorrt_availability
    except ImportError:
        logger.error(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )
        return

    if not check_tensorrt_availability():
        logger.error(
            "TensorRTが利用できません. " "TensorRT SDKをインストールしてください."
        )
        return

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        logger.error(f"ONNXモデルが見つかりません: {onnx_path}")
        return

    if args.int8:
        precision = "int8"
    elif args.fp16:
        precision = "fp16"
    else:
        precision = "fp32"

    if args.output:
        output_path = Path(args.output)
    else:
        stem = onnx_path.stem
        if precision != "fp32":
            stem = f"{stem}_{precision}"
        output_path = onnx_path.with_name(f"{stem}.engine")

    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"精度: {precision.upper()}")
    logger.info(f"出力: {output_path}")

    from pochitrain.tensorrt.input_shape_resolver import InputShapeResolver

    shape_resolver = InputShapeResolver(logger)
    try:
        input_shape = shape_resolver.resolve(args.input_size, onnx_path)
    except ValueError as e:
        logger.error(str(e))
        return
    if input_shape is not None:
        logger.debug(f"入力形状: {input_shape}")

    calibrator = None
    if precision == "int8":
        from pochitrain.tensorrt.calibrator import create_int8_calibrator
        from pochitrain.tensorrt.int8_config import INT8CalibrationConfigurer

        configurer = INT8CalibrationConfigurer(logger)
        try:
            calib_config = configurer.configure(
                config_path=args.config_path,
                calib_data=args.calib_data,
                input_shape=input_shape,
                onnx_path=onnx_path,
                output_path=output_path,
                calib_samples=args.calib_samples,
                calib_batch_size=args.calib_batch_size,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.error(str(e))
            return

        logger.info(
            f"キャリブレーション設定: "
            f"データ={calib_config.calib_data_root}, "
            f"最大サンプル数={calib_config.max_samples}"
        )

        try:
            calibrator = create_int8_calibrator(
                data_root=calib_config.calib_data_root,
                transform=calib_config.transform,
                input_shape=calib_config.input_shape,
                batch_size=calib_config.batch_size,
                max_samples=calib_config.max_samples,
                cache_file=calib_config.cache_file,
            )
        except Exception as e:
            logger.error(f"キャリブレータ作成エラー: {e}")
            return

    try:
        converter = TensorRTConverter(
            onnx_path=onnx_path,
            workspace_size=args.workspace_size,
        )
        engine_path = converter.convert(
            output_path=output_path,
            precision=precision,
            calibrator=calibrator,
            input_shape=input_shape,
        )
        logger.info(f"変換完了: {engine_path}")

    except Exception as e:
        logger.error(f"TensorRT変換エラー: {e}")
        return

    logger.info("推論するには:")
    logger.info(f"  uv run infer-trt {engine_path}")
