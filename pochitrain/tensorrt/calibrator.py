"""TensorRT INT8キャリブレーションモジュール.

PochiImageDatasetからキャリブレーションデータを供給し,
TensorRT INT8量子化に必要なスケールファクタを計算する.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Sized, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pochitrain.logging import LoggerManager
from pochitrain.pochi_dataset import PochiImageDataset

logger: logging.Logger = LoggerManager().get_logger(__name__)


def _check_tensorrt_for_calibrator() -> None:
    """キャリブレータ使用時のTensorRT利用可否チェック.

    Raises:
        ImportError: TensorRTがインストールされていない場合
    """
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorRTがインストールされていません. "
            "TensorRT SDKをインストールしてください."
        )


def create_calibration_dataset(
    data_root: str,
    transform: Callable[..., Any],
    max_samples: int = 500,
    seed: int = 42,
) -> Dataset[Any]:
    """キャリブレーション用のサブセットデータセットを作成する.

    検証データセットからランダムにサンプリングして,
    キャリブレーション用の小さなデータセットを作成する.

    Args:
        data_root: データルートディレクトリ
        transform: 前処理Transform (訓練時と同一のval_transformを推奨)
        max_samples: 最大サンプル数 (デフォルト: 500)
        seed: 乱数シード (再現性のため)

    Returns:
        キャリブレーション用サブセットデータセット
    """
    dataset = PochiImageDataset(data_root, transform=transform)
    total = len(dataset)

    if total <= max_samples:
        logger.debug(f"データセット全体をキャリブレーションに使用: {total}枚")
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:max_samples].tolist()
    subset = Subset(dataset, indices)
    logger.debug(
        f"キャリブレーション用サブセットを作成: "
        f"{len(indices)}枚 / {total}枚 (seed={seed})"
    )
    return subset


def _create_calibrator_class() -> type:
    """IInt8EntropyCalibrator2を継承したキャリブレータクラスを動的に生成する.

    TensorRTはオプション依存のため, トップレベルでの継承ができない.
    この関数はTensorRTがインストールされている環境でのみ呼び出される.

    Returns:
        PochiInt8Calibrator クラス (trt.IInt8EntropyCalibrator2 の派生型)

    Raises:
        ImportError: TensorRTがインストールされていない場合
    """
    import tensorrt as trt

    class _PochiInt8Calibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
        """TensorRT INT8キャリブレータ.

        IInt8EntropyCalibrator2を継承し, PochiImageDatasetから
        キャリブレーションデータを供給する.

        PyTorch CUDAテンソルをGPUバッファとして使用するため, pycudaは不要.

        Attributes:
            batch_size: キャリブレーションバッチサイズ
            cache_file: キャリブレーションキャッシュファイルパス
        """

        def __init__(
            self,
            dataset: Dataset[Any],
            input_shape: tuple[int, ...],
            batch_size: int = 1,
            cache_file: str = "calibration.cache",
        ) -> None:
            """PochiInt8Calibratorを初期化.

            Args:
                dataset: キャリブレーション用データセット
                input_shape: モデル入力形状 (channels, height, width)
                batch_size: バッチサイズ
                cache_file: キャリブレーションキャッシュのファイルパス
            """
            super().__init__()

            self.dataset = dataset
            self.batch_size = batch_size
            self.cache_file = cache_file
            self.current_index = 0
            self.input_shape = input_shape

            # DataLoaderで効率的にデータを供給
            self._data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            self._data_iter = iter(self._data_loader)

            # GPUバッファ確保 (PyTorch CUDAテンソル)
            self._d_input = torch.empty(
                (batch_size, *input_shape), dtype=torch.float32, device="cuda"
            )

            logger.debug(
                f"INT8キャリブレータ初期化: "
                f"データ数={len(cast(Sized, dataset))}, "
                f"バッチサイズ={batch_size}, "
                f"入力形状={input_shape}"
            )

        def get_batch_size(self) -> int:
            """キャリブレーションバッチサイズを返す.

            Returns:
                バッチサイズ
            """
            return self.batch_size

        def get_batch(self, names: List[str]) -> Optional[List[int]]:
            """次のキャリブレーションバッチを供給する.

            Args:
                names: テンソル名のリスト (TensorRT APIから渡される)

            Returns:
                GPUバッファポインタのリスト, またはデータ終了時にNone
            """
            try:
                batch_images, _ = next(self._data_iter)
            except StopIteration:
                return None

            actual_batch_size = batch_images.shape[0]

            if actual_batch_size < self.batch_size:
                # バッチサイズに満たない場合はゼロパディング
                self._d_input.zero_()

            self._d_input[:actual_batch_size].copy_(batch_images[:actual_batch_size])
            self.current_index += actual_batch_size

            return [int(self._d_input.data_ptr())]

        def read_calibration_cache(self) -> Optional[bytes]:
            """キャリブレーションキャッシュを読み込む.

            キャッシュが存在する場合は再キャリブレーション不要.

            Returns:
                キャッシュデータ, またはキャッシュが存在しない場合None
            """
            if os.path.exists(self.cache_file):
                logger.debug(
                    f"キャリブレーションキャッシュを読み込み: {self.cache_file}"
                )
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            """キャリブレーション結果をキャッシュファイルに保存する.

            Args:
                cache: キャリブレーションキャッシュデータ
            """
            cache_path = Path(self.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_file, "wb") as f:
                f.write(cache)
            logger.debug(f"キャリブレーションキャッシュを保存: {self.cache_file}")

    return _PochiInt8Calibrator


def create_int8_calibrator(
    data_root: str,
    transform: Callable[..., Any],
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    max_samples: int = 500,
    cache_file: str = "calibration.cache",
    seed: int = 42,
) -> Any:
    """INT8キャリブレータを作成するファクトリ関数.

    データセットの作成からキャリブレータの初期化までを一括で行う.
    TensorRT IInt8EntropyCalibrator2 を継承したインスタンスを返す.

    Args:
        data_root: データルートディレクトリ
        transform: 前処理Transform (val_transform)
        input_shape: モデル入力形状 (channels, height, width)
        batch_size: キャリブレーションバッチサイズ
        max_samples: 最大サンプル数
        cache_file: キャリブレーションキャッシュのファイルパス
        seed: 乱数シード

    Returns:
        初期化済みのINT8キャリブレータ (trt.IInt8EntropyCalibrator2 派生)

    Raises:
        ImportError: TensorRTがインストールされていない場合
    """
    _check_tensorrt_for_calibrator()

    dataset = create_calibration_dataset(
        data_root=data_root,
        transform=transform,
        max_samples=max_samples,
        seed=seed,
    )

    calibrator_cls = _create_calibrator_class()
    return calibrator_cls(
        dataset=dataset,
        input_shape=input_shape,
        batch_size=batch_size,
        cache_file=cache_file,
    )
