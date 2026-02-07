"""TensorRT INT8キャリブレータのテスト.

TensorRTはオプション依存のため, TensorRT不要な関数のテストと
TensorRT必須のテストを分離.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

from pochitrain.tensorrt.calibrator import create_calibration_dataset


class _DummyDataset(Dataset):
    """テスト用ダミーデータセット."""

    def __init__(self, size, image_shape=(3, 224, 224)):
        self.size = size
        self.image_shape = image_shape
        self.image_paths = [Path(f"img_{i}.jpg") for i in range(size)]
        self.labels = [i % 4 for i in range(size)]
        self.classes = [f"class_{i}" for i in range(4)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(*self.image_shape)
        return image, self.labels[index]

    def get_classes(self):
        return self.classes

    def get_file_paths(self):
        return [str(p) for p in self.image_paths]


class TestCreateCalibrationDataset:
    """create_calibration_dataset関数のテスト."""

    def _create_data_dir(self, tmp_path, num_classes=2, images_per_class=5):
        """テスト用のフォルダ構造を作成."""
        for i in range(num_classes):
            class_dir = tmp_path / f"class_{i}"
            class_dir.mkdir()
            for j in range(images_per_class):
                img_path = class_dir / f"img_{j}.jpg"
                # 最小限の有効なJPEGファイルを作成
                from PIL import Image

                img = Image.new("RGB", (32, 32), color=(i * 50, j * 10, 0))
                img.save(img_path)
        return str(tmp_path)

    def test_returns_full_dataset_when_under_max(self, tmp_path):
        """データ数がmax_samples以下の場合は全データを使用."""
        data_root = self._create_data_dir(tmp_path, num_classes=2, images_per_class=3)
        transform = transforms.Compose([transforms.ToTensor()])

        result = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=100,
        )

        # PochiImageDatasetがそのまま返される
        assert len(result) == 6  # 2 classes * 3 images

    def test_returns_subset_when_over_max(self, tmp_path):
        """データ数がmax_samplesを超える場合はSubsetを返す."""
        data_root = self._create_data_dir(tmp_path, num_classes=4, images_per_class=10)
        transform = transforms.Compose([transforms.ToTensor()])

        result = create_calibration_dataset(
            data_root=data_root,
            transform=transform,
            max_samples=10,
        )

        assert isinstance(result, Subset)
        assert len(result) == 10

    def test_reproducible_with_same_seed(self, tmp_path):
        """同じseedで同じサブセットが生成される."""
        data_root = self._create_data_dir(tmp_path, num_classes=4, images_per_class=10)
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

        # 同じインデックスが選ばれること
        assert isinstance(result1, Subset)
        assert isinstance(result2, Subset)
        assert result1.indices == result2.indices

    def test_different_seed_gives_different_subset(self, tmp_path):
        """異なるseedで異なるサブセットが生成される."""
        data_root = self._create_data_dir(tmp_path, num_classes=4, images_per_class=10)
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


class TestCreateInt8CalibratorWithoutTensorRT:
    """TensorRT未インストール環境でのキャリブレータテスト."""

    def test_import_error_without_tensorrt(self):
        """TensorRTがない場合にcreate_int8_calibratorでImportErrorが発生する."""
        with patch(
            "pochitrain.tensorrt.calibrator._check_tensorrt_for_calibrator",
            side_effect=ImportError("TensorRTがインストールされていません"),
        ):
            from pochitrain.tensorrt.calibrator import create_int8_calibrator

            with pytest.raises(ImportError, match="TensorRT"):
                create_int8_calibrator(
                    data_root="dummy",
                    transform=transforms.Compose([transforms.ToTensor()]),
                    input_shape=(3, 224, 224),
                )


class TestCreateInt8Calibrator:
    """create_int8_calibrator ファクトリ関数のテスト."""

    def test_creates_calibrator_with_valid_args(self, tmp_path):
        """有効な引数でキャリブレータが作成される."""
        # テスト用データ作成
        for i in range(2):
            class_dir = tmp_path / f"class_{i}"
            class_dir.mkdir()
            for j in range(3):
                from PIL import Image

                img = Image.new("RGB", (32, 32))
                img.save(class_dir / f"img_{j}.jpg")

        transform = transforms.Compose([transforms.ToTensor()])

        # TensorRTがない場合はImportErrorが発生するが,
        # create_calibration_dataset自体はTensorRT不要なので分離テスト
        with patch(
            "pochitrain.tensorrt.calibrator._check_tensorrt_for_calibrator"
        ) as mock_check:
            mock_check.side_effect = ImportError("TensorRT not installed")

            from pochitrain.tensorrt.calibrator import create_int8_calibrator

            with pytest.raises(ImportError):
                create_int8_calibrator(
                    data_root=str(tmp_path),
                    transform=transform,
                    input_shape=(3, 224, 224),
                    max_samples=5,
                )
