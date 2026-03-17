"""convert サブコマンドの実装."""

import argparse
from pathlib import Path

from pochitrain.cli.cli_commons import setup_logging
from pochitrain.utils import (
    ConfigLoader,
    load_config_auto,
)


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

    # 動的シェイプONNXの変換時は全精度で入力サイズ指定が必要.
    input_shape = None
    if args.input_size:
        # チャンネル数は RGB (C=3) 固定. pochitrain は RGB 画像のみ対応.
        input_shape = (3, args.input_size[0], args.input_size[1])
        logger.debug(f"CLI指定の入力形状: {input_shape}")
    else:
        try:
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            input_tensor = onnx_model.graph.input[0]
            input_dims = input_tensor.type.tensor_type.shape.dim

            dynamic_dims = [
                d.dim_param for d in input_dims[1:] if d.dim_value == 0 and d.dim_param
            ]
            if any(d.dim_value == 0 for d in input_dims[1:]):
                dynamic_info = (
                    f" (動的次元: {', '.join(dynamic_dims)})" if dynamic_dims else ""
                )
                logger.error(
                    f"ONNXモデルに動的シェイプが含まれています{dynamic_info}. "
                    "--input-size で入力サイズを明示的に指定してください. "
                    "例: --input-size 224 224"
                )
                return
        except ImportError:
            logger.debug(
                "onnxパッケージが未インストールのため動的シェイプ検出をスキップ"
            )
        except Exception as e:
            logger.debug(f"ONNX動的シェイプ検出中にエラー: {e}")

    calibrator = None
    if precision == "int8":
        from pochitrain.tensorrt.calibrator import create_int8_calibrator

        config = None
        if args.config_path:
            config_path = Path(args.config_path)
            try:
                config = ConfigLoader.load_config(str(config_path))
                logger.debug(f"設定ファイルを読み込み: {config_path}")
            except Exception as e:
                logger.error(f"設定ファイル読み込みエラー: {e}")
                return
        else:
            try:
                config = load_config_auto(onnx_path)
            except (FileNotFoundError, RuntimeError) as e:
                logger.error(str(e))
                return

        if args.calib_data:
            calib_data_root = args.calib_data
        elif config.get("val_data_root"):
            calib_data_root = config["val_data_root"]
            logger.debug(f"キャリブレーションデータをconfigから取得: {calib_data_root}")
        else:
            logger.error(
                "--calib-data を指定するか, " "configにval_data_rootを設定してください"
            )
            return

        calib_data_path = Path(calib_data_root)
        if not calib_data_path.exists():
            logger.error(f"キャリブレーションデータが見つかりません: {calib_data_path}")
            return

        if "val_transform" not in config:
            logger.error(
                "configにval_transformが設定されていません. "
                "INT8キャリブレーションにはval_transformが必要です."
            )
            return
        transform = config["val_transform"]

        if input_shape is not None:
            calib_input_shape = input_shape
        else:
            try:
                import onnx

                onnx_model = onnx.load(str(onnx_path))
                input_tensor = onnx_model.graph.input[0]
                input_dims = input_tensor.type.tensor_type.shape.dim
                calib_input_shape = tuple(d.dim_value for d in input_dims[1:])
                logger.debug(f"ONNX入力形状: {calib_input_shape}")
            except Exception as e:
                logger.error(f"ONNXモデルから入力形状を取得できません: {e}")
                return

        cache_file = str(output_path.with_suffix(".cache"))

        max_calib_samples = args.calib_samples

        logger.info(
            f"キャリブレーション設定: "
            f"データ={calib_data_root}, "
            f"最大サンプル数={max_calib_samples}"
        )

        try:
            calibrator = create_int8_calibrator(
                data_root=calib_data_root,
                transform=transform,
                input_shape=calib_input_shape,
                batch_size=args.calib_batch_size,
                max_samples=max_calib_samples,
                cache_file=cache_file,
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
