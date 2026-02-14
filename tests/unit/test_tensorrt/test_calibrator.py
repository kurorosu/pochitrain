"""TensorRT INT8キャリブレータのユニットテスト.

TensorRT がない環境でも検証できる経路を中心に,
データセット生成とキャリブレータ生成の振る舞いを確認する.
"""

from pathlib import Path
from typing import Any

import pytest
import torchvision.transforms as transforms
from torch.utils.data import Subset

import pochitrain.tensorrt.calibrator as calibrator_module
from pochitrain.tensorrt.calibrator import (
    create_calibration_dataset,
    create_int8_calibrator,
)


def _create_data_dir(
    tmp_path: Path, num_classes: int = 2, images_per_class: int = 5
) -> str:
    """テスト用のフォルダ形式データセットを作成する."""
    from PIL import Image

    for class_index in range(num_classes):
        class_dir = tmp_path / f"class_{class_index}"
        class_dir.mkdir()
        for image_index in range(images_per_class):
            image_path = class_dir / f"img_{image_index}.jpg"
            image = Image.new(
                "RGB",
                (32, 32),
                color=(class_index * 50, image_index * 10, 0),
            )
            image.save(image_path)

    return str(tmp_path)


class TestCreateCalibrationDataset:
    """create_calibration_dataset 関数のテスト."""

    def test_returns_full_dataset_when_under_max(self, tmp_path: Path) -> None:
        """総サンプル数が max_samples 以下なら全件を返すことを確認する."""
        data_root = _create_data_dir(tmp_path, num_classes=2, images_per_class=3)
        transform = transforms.Compose([transforms.ToTensor()])

        result = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=100,
        )

        assert len(result) == 6

    def test_returns_subset_when_over_max(self, tmp_path: Path) -> None:
        """総サンプル数が max_samples を超えると Subset を返すことを確認する."""
        data_root = _create_data_dir(tmp_path, num_classes=4, images_per_class=10)
        transform = transforms.Compose([transforms.ToTensor()])

        result = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=10,
        )

        assert isinstance(result, Subset)
        assert len(result) == 10

    def test_reproducible_with_same_seed(self, tmp_path: Path) -> None:
        """同じ seed なら同じサブセットになることを確認する."""
        data_root = _create_data_dir(tmp_path, num_classes=4, images_per_class=10)
        transform = transforms.Compose([transforms.ToTensor()])

        result1 = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=5,
            seed=123,
        )
        result2 = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=5,
            seed=123,
        )

        assert isinstance(result1, Subset)
        assert isinstance(result2, Subset)
        assert result1.indices == result2.indices

    def test_different_seed_gives_different_subset(self, tmp_path: Path) -> None:
        """異なる seed なら異なるサブセットになることを確認する."""
        data_root = _create_data_dir(tmp_path, num_classes=4, images_per_class=10)
        transform = transforms.Compose([transforms.ToTensor()])

        result1 = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=5,
            seed=42,
        )
        result2 = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=5,
            seed=99,
        )

        assert isinstance(result1, Subset)
        assert isinstance(result2, Subset)
        assert result1.indices != result2.indices


class TestCreateInt8Calibrator:
    """create_int8_calibrator 関数のテスト."""

    def test_raises_import_error_when_tensorrt_missing(self) -> None:
        """TensorRT がない場合は ImportError を送出することを確認する."""

        def _raise_import_error() -> None:
            raise ImportError("TensorRT not installed")

        original = calibrator_module._check_tensorrt_for_calibrator
        calibrator_module._check_tensorrt_for_calibrator = _raise_import_error
        try:
            with pytest.raises(ImportError, match="TensorRT"):
                create_int8_calibrator(
                    data_root="dummy",
                    transform=transforms.Compose([transforms.ToTensor()]),
                    input_shape=(3, 224, 224),
                )
        finally:
            calibrator_module._check_tensorrt_for_calibrator = original

    def test_wires_dataset_and_calibrator_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """内部部品を正しく接続してキャリブレータを構築することを確認する."""
        transform = transforms.Compose([transforms.ToTensor()])
        sentinel_dataset = object()

        called: dict[str, Any] = {}

        def _fake_check() -> None:
            called["check"] = True

        def _fake_create_dataset(
            data_root: str,
            transform: Any,
            max_samples: int,
            seed: int,
        ) -> object:
            called["dataset_args"] = {
                "data_root": data_root,
                "transform": transform,
                "max_samples": max_samples,
                "seed": seed,
            }
            return sentinel_dataset

        class _DummyCalibrator:
            def __init__(
                self,
                dataset: object,
                input_shape: tuple[int, ...],
                batch_size: int,
                cache_file: str,
            ) -> None:
                self.dataset = dataset
                self.input_shape = input_shape
                self.batch_size = batch_size
                self.cache_file = cache_file

        def _fake_create_calibrator_class() -> type[_DummyCalibrator]:
            called["class_factory"] = True
            return _DummyCalibrator

        monkeypatch.setattr(
            calibrator_module, "_check_tensorrt_for_calibrator", _fake_check
        )
        monkeypatch.setattr(
            calibrator_module, "create_calibration_dataset", _fake_create_dataset
        )
        monkeypatch.setattr(
            calibrator_module, "_create_calibrator_class", _fake_create_calibrator_class
        )

        result = create_int8_calibrator(
            data_root="dummy_root",
            transform=transform,
            input_shape=(3, 224, 224),
            batch_size=8,
            max_samples=50,
            cache_file="cache.bin",
            seed=7,
        )

        assert called["check"] is True
        assert called["class_factory"] is True
        assert called["dataset_args"] == {
            "data_root": "dummy_root",
            "transform": transform,
            "max_samples": 50,
            "seed": 7,
        }
        assert isinstance(result, _DummyCalibrator)
        assert result.dataset is sentinel_dataset
        assert result.input_shape == (3, 224, 224)
        assert result.batch_size == 8
        assert result.cache_file == "cache.bin"
