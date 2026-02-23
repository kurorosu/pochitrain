"""推論ランタイムアダプタモジュール."""

from pochitrain.inference.interfaces import IRuntimeAdapter

from .onnx_runtime_adapter import OnnxRuntimeAdapter
from .trt_runtime_adapter import TensorRTRuntimeAdapter

__all__ = [
    "IRuntimeAdapter",
    "OnnxRuntimeAdapter",
    "TensorRTRuntimeAdapter",
]
