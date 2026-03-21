"""ONNX モデルの入力形状解析と動的シェイプ検出."""

import logging
from pathlib import Path
from typing import Optional


class InputShapeResolver:
    """ONNX モデルの入力形状を解析し, 動的シェイプを検出する."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """入力形状リゾルバを初期化.

        Args:
            logger: ロガー. None の場合はモジュールロガーを使用.
        """
        self._logger = logger or logging.getLogger(__name__)

    def resolve(
        self,
        cli_input_size: Optional[list[int]],
        onnx_path: Path,
    ) -> Optional[tuple[int, ...]]:
        """CLI 引数または ONNX モデルから入力形状を解決する.

        Args:
            cli_input_size: CLI で指定された [H, W]. None なら ONNX から検出.
            onnx_path: ONNX モデルファイルパス.

        Returns:
            (C, H, W) 形式の入力形状. 静的シェイプで指定不要な場合は None.

        Raises:
            ValueError: 動的シェイプが検出され, cli_input_size が未指定の場合.
        """
        if cli_input_size is not None:
            # チャンネル数は RGB (C=3) 固定. pochitrain は RGB 画像のみ対応.
            return (3, cli_input_size[0], cli_input_size[1])

        return self._detect_from_onnx(onnx_path)

    def _detect_from_onnx(self, onnx_path: Path) -> Optional[tuple[int, ...]]:
        """ONNX モデルから入力形状を検出する."""
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
                raise ValueError(
                    f"ONNXモデルに動的シェイプが含まれています{dynamic_info}. "
                    "--input-size で入力サイズを明示的に指定してください. "
                    "例: --input-size 224 224"
                )
        except ImportError:
            pass
        except ValueError:
            raise
        except Exception:
            self._logger.debug("ONNX 入力形状の検出中にエラーが発生", exc_info=True)

        return None

    def extract_static_shape(self, onnx_path: Path) -> tuple[int, ...]:
        """ONNX モデルから静的入力形状 (C, H, W) を取得する.

        Args:
            onnx_path: ONNX モデルファイルパス.

        Returns:
            (C, H, W) 形式の入力形状.

        Raises:
            RuntimeError: 入力形状を取得できない場合.
        """
        try:
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            input_tensor = onnx_model.graph.input[0]
            input_dims = input_tensor.type.tensor_type.shape.dim
            return tuple(d.dim_value for d in input_dims[1:])
        except Exception as e:
            raise RuntimeError(f"ONNXモデルから入力形状を取得できません: {e}") from e
