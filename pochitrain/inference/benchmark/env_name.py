"""ベンチマーク結果の環境名解決."""

import platform
from typing import Optional

import torch


def resolve_env_name(
    *,
    use_gpu: bool,
    configured_env_name: Optional[str],
) -> str:
    """ベンチマーク結果に出力する環境名を解決する.

    Args:
        use_gpu: GPUを利用しているかどうか.
        configured_env_name: configで指定された環境名.

    Returns:
        環境識別文字列.
    """
    if configured_env_name:
        return configured_env_name

    system_name = platform.system()
    if use_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).replace(" ", "")
        return f"{system_name}-{device_name}"
    return f"{system_name}-CPU"
