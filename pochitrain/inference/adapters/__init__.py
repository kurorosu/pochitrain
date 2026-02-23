"""推論ランタイムアダプタモジュール."""

from pochitrain.inference.interfaces import IRuntimeAdapter

from .onnx_runtime_adapter import OnnxRuntimeAdapter
from .pytorch_runtime_adapter import PyTorchRuntimeAdapter
from .trt_runtime_adapter import TensorRTRuntimeAdapter

__all__ = [
    "IRuntimeAdapter",
    "OnnxRuntimeAdapter",
    "PyTorchRuntimeAdapter",
    "TensorRTRuntimeAdapter",
]
