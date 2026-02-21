"""PochiImageDatasetのテスト."""

import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image

from pochitrain.pochi_dataset import (
    FastInferenceDataset,
    PochiImageDataset,
    convert_transform_for_fast_inference,
    create_data_loaders,
    create_scaled_normalize_tensors,
    gpu_normalize,
)


class TestPochiImageDataset:
    """PochiImageDatasetクラスのテスト."""

    def test_basic_dataset_creation(self, create_dummy_dataset):
        """基本的なデータセット作成のテスト."""
        dataset_path = create_dummy_dataset({"class1": 3, "class2": 2})

        dataset = PochiImageDataset(str(dataset_path))

        assert len(dataset) == 5  # 3 + 2
        assert len(dataset.classes) == 2
        assert "class1" in dataset.classes
        assert "class2" in dataset.classes

    def test_data_loading(self, create_dummy_dataset):
        """データ読み込み機能のテスト."""
        dataset_path = create_dummy_dataset({"cat": 2, "dog": 3})

        dataset = PochiImageDataset(str(dataset_path))

        image, label = dataset[0]

        assert image is not None
        assert isinstance(label, int)
        assert 0 <= label < len(dataset.classes)

    def test_get_classes(self, create_dummy_dataset):
        """クラス情報取得のテスト."""
        dataset_path = create_dummy_dataset({"apple": 1, "banana": 1, "cherry": 1})

        dataset = PochiImageDataset(str(dataset_path))
        classes = dataset.get_classes()

        expected_classes = ["apple", "banana", "cherry"]
        assert classes == expected_classes

    def test_get_class_counts(self, create_dummy_dataset):
        """クラス別画像数取得のテスト."""
        dataset_path = create_dummy_dataset({"class_a": 4, "class_b": 2, "class_c": 6})

        dataset = PochiImageDataset(str(dataset_path))
        counts = dataset.get_class_counts()

        assert counts["class_a"] == 4
        assert counts["class_b"] == 2
        assert counts["class_c"] == 6

    def test_get_file_paths(self, create_dummy_dataset):
        """ファイルパス取得機能のテスト."""
        dataset_path = create_dummy_dataset({"test_class": 3})

        dataset = PochiImageDataset(str(dataset_path))
        file_paths = dataset.get_file_paths()

        assert len(file_paths) == 3

        for path in file_paths:
            assert isinstance(path, str)
            assert "test_class" in path
            assert path.endswith(".jpg")

    def test_different_extensions(self):
        """異なる拡張子のファイル対応テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            class_dir = base_path / "mixed_class"
            class_dir.mkdir()

            extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            for i, ext in enumerate(extensions):
                img = Image.new("RGB", (32, 32), color=(i * 60, 100, 200))
                img_path = class_dir / f"image_{i}{ext}"
                img.save(img_path)

            (class_dir / "text_file.txt").write_text("not an image")

            dataset = PochiImageDataset(str(base_path))

            assert len(dataset) == 4  # .txt は除外される

    def test_custom_extensions(self):
        """カスタム拡張子設定のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            class_dir = base_path / "custom_class"
            class_dir.mkdir()

            img_jpg = Image.new("RGB", (32, 32), color=(255, 0, 0))
            img_jpg.save(class_dir / "image.jpg")

            img_png = Image.new("RGB", (32, 32), color=(0, 255, 0))
            img_png.save(class_dir / "image.png")

            dataset = PochiImageDataset(str(base_path), extensions=(".jpg",))

            assert len(dataset) == 1

    def test_empty_directory_error(self):
        """空ディレクトリのエラーハンドリング."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="クラスフォルダが見つかりません"):
                PochiImageDataset(temp_dir)

    def test_no_images_error(self):
        """画像ファイルなしのエラーハンドリング."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            (base_path / "empty_class").mkdir()

            with pytest.raises(ValueError, match="画像ファイルが見つかりません"):
                PochiImageDataset(str(base_path))

    def test_nonexistent_directory_error(self):
        """存在しないディレクトリのエラーハンドリング."""
        with pytest.raises(
            FileNotFoundError, match="データディレクトリが見つかりません"
        ):
            PochiImageDataset("/nonexistent/path")


class TestCreateDataLoaders:
    """create_data_loaders関数のテスト."""

    def test_create_data_loaders_basic(self, create_dummy_train_val):
        """基本的なデータローダー作成のテスト."""
        train_root, val_root = create_dummy_train_val()

        train_transform = transforms.Compose([transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.ToTensor()])

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(classes) == 2
        assert "cat" in classes
        assert "dog" in classes

        train_batch = next(iter(train_loader))
        assert len(train_batch[0]) <= 2

    def test_create_data_loaders_both_required(self, create_dummy_train_val):
        """訓練・検証データ両方必須のテスト."""
        train_root, val_root = create_dummy_train_val()

        train_transform = transforms.Compose([transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.ToTensor()])

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(classes) == 2

    def test_create_data_loaders_custom_params(self, create_dummy_train_val):
        """カスタムパラメータでのデータローダー作成テスト."""
        train_root, val_root = create_dummy_train_val()

        train_transform = transforms.Compose([transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.ToTensor()])

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        assert train_loader.batch_size == 1
        assert train_loader.pin_memory is False


class TestDataLoaderCreation:
    """データローダー作成のテスト（バリデーションは別モジュールで実施）."""

    def test_data_loaders_creation_with_valid_transforms(self, create_dummy_train_val):
        """有効なtransformでデータローダーが正常作成されることをテスト."""
        train_root, val_root = create_dummy_train_val(
            train_per_class=2, val_per_class=1
        )

        train_transform = transforms.Compose([transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.ToTensor()])

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(classes) == 2

    def test_complex_transforms_work(self, create_dummy_train_val):
        """複雑なtransformで正常動作するテスト."""
        train_root, val_root = create_dummy_train_val(
            train_per_class=2, val_per_class=1
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_loader, val_loader, classes = create_data_loaders(
            train_root=train_root,
            val_root=val_root,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(classes) == 2


class TestFastInferenceDataset:
    """FastInferenceDatasetクラスのテスト."""

    def test_returns_tensor(self, create_dummy_dataset):
        """decode_imageによりテンソルが直接返されることを確認."""
        dataset_path = create_dummy_dataset({"class1": 2, "class2": 1})

        transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = FastInferenceDataset(str(dataset_path), transform=transform)

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
        assert image.shape[0] == 3  # RGB channels
        assert isinstance(label, int)

    def test_without_transform(self, create_dummy_dataset):
        """transform無しでuint8テンソルが返されることを確認."""
        dataset_path = create_dummy_dataset({"class1": 1})

        dataset = FastInferenceDataset(str(dataset_path))

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.uint8
        assert image.shape[0] == 3

    def test_inherits_pochi_image_dataset(self, create_dummy_dataset):
        """PochiImageDatasetの機能を継承していることを確認."""
        dataset_path = create_dummy_dataset({"cat": 3, "dog": 2})

        dataset = FastInferenceDataset(str(dataset_path))

        assert isinstance(dataset, PochiImageDataset)
        assert len(dataset) == 5
        assert dataset.get_classes() == ["cat", "dog"]
        assert dataset.get_class_counts() == {"cat": 3, "dog": 2}

    def test_output_matches_pochi_image_dataset(self, create_dummy_dataset):
        """PochiImageDatasetとFastInferenceDatasetで同等の出力を確認."""
        dataset_path = create_dummy_dataset({"class1": 2})

        pochi_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        fast_transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        pochi_ds = PochiImageDataset(str(dataset_path), transform=pochi_transform)
        fast_ds = FastInferenceDataset(str(dataset_path), transform=fast_transform)

        pochi_img, pochi_label = pochi_ds[0]
        fast_img, fast_label = fast_ds[0]

        assert pochi_label == fast_label
        assert pochi_img.shape == fast_img.shape
        assert torch.allclose(pochi_img, fast_img, atol=1e-5)


class TestConvertTransformForFastInference:
    """convert_transform_for_fast_inference関数のテスト."""

    def test_replaces_to_tensor(self):
        """ToTensorがConvertImageDtypeに置き換えられることを確認."""
        original = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        result = convert_transform_for_fast_inference(original)

        assert result is not None
        assert len(result.transforms) == 3
        assert isinstance(result.transforms[0], transforms.Resize)
        assert isinstance(result.transforms[1], transforms.ConvertImageDtype)
        assert isinstance(result.transforms[2], transforms.Normalize)

    def test_preserves_non_to_tensor_transforms(self):
        """ToTensor以外のtransformが維持されることを確認."""
        original = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        result = convert_transform_for_fast_inference(original)

        assert result is not None
        assert len(result.transforms) == 4
        assert isinstance(result.transforms[0], transforms.Resize)
        assert isinstance(result.transforms[1], transforms.CenterCrop)
        assert isinstance(result.transforms[2], transforms.ConvertImageDtype)
        assert isinstance(result.transforms[3], transforms.Normalize)

    def test_returns_none_for_pil_only_transform(self):
        """PIL専用transformが含まれる場合にNoneを返すことを確認."""
        original = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )

        result = convert_transform_for_fast_inference(original)

        assert result is None

    def test_no_compose_preserves_original(self):
        """Composeでない単体入力の場合に元のtransformが保持されることを確認."""
        result = convert_transform_for_fast_inference(transforms.Resize(224))

        assert result is not None
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], transforms.Resize)
        assert isinstance(result.transforms[1], transforms.ConvertImageDtype)

    def test_no_compose_to_tensor(self):
        """単体ToTensorがConvertImageDtypeに置き換えられることを確認."""
        result = convert_transform_for_fast_inference(transforms.ToTensor())

        assert result is not None
        assert len(result.transforms) == 1
        assert isinstance(result.transforms[0], transforms.ConvertImageDtype)

    def test_no_compose_pil_only_returns_none(self):
        """単体PIL専用transformの場合にNoneを返すことを確認."""
        result = convert_transform_for_fast_inference(transforms.ToPILImage())

        assert result is None

    def test_convert_dtype_is_float32(self):
        """置き換えられたConvertImageDtypeがfloat32であることを確認."""
        original = transforms.Compose([transforms.ToTensor()])

        result = convert_transform_for_fast_inference(original)

        assert result is not None
        convert_dtype = result.transforms[0]
        assert isinstance(convert_dtype, transforms.ConvertImageDtype)
        assert convert_dtype.dtype == torch.float32


class TestGpuNormalize:
    """gpu_normalize と事前計算テンソル作成のテスト."""

    def test_create_scaled_normalize_tensors_cpu(self):
        """255倍済みテンソルが想定形状/値で作成されることを確認."""
        mean = [0.5, 0.25, 0.125]
        std = [0.1, 0.2, 0.4]

        mean_255, std_255 = create_scaled_normalize_tensors(mean, std, device="cpu")

        assert mean_255.shape == (1, 3, 1, 1)
        assert std_255.shape == (1, 3, 1, 1)
        assert mean_255.dtype == torch.float32
        assert std_255.dtype == torch.float32
        assert torch.allclose(
            mean_255.view(-1), torch.tensor(mean, dtype=torch.float32) * 255.0
        )
        assert torch.allclose(
            std_255.view(-1), torch.tensor(std, dtype=torch.float32) * 255.0
        )

    def test_gpu_normalize_accepts_precomputed_tensors(self):
        """事前計算テンソルを使って正規化できることを確認."""
        images = torch.tensor(
            [
                [[0, 255], [128, 64]],
                [[10, 20], [30, 40]],
                [[50, 60], [70, 80]],
            ],
            dtype=torch.uint8,
        )
        mean = [0.5, 0.25, 0.125]
        std = [0.1, 0.2, 0.4]

        mean_255, std_255 = create_scaled_normalize_tensors(mean, std, device="cpu")
        normalized = gpu_normalize(images, mean_255, std_255)

        expected = (images.unsqueeze(0).to(torch.float32) - mean_255) / std_255
        assert normalized.shape == (1, 3, 2, 2)
        assert normalized.dtype == torch.float32
        assert torch.allclose(normalized, expected, atol=1e-6)

    def test_gpu_normalize_non_blocking_flag_keeps_numerical_result(self):
        """non_blocking の有無で正規化結果が一致することを確認."""
        images = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
        mean_255, std_255 = create_scaled_normalize_tensors(
            [0.5, 0.25, 0.125],
            [0.1, 0.2, 0.4],
            device="cpu",
        )

        out_non_block = gpu_normalize(images, mean_255, std_255, non_blocking=True)
        out_block = gpu_normalize(images, mean_255, std_255, non_blocking=False)

        assert torch.allclose(out_non_block, out_block, atol=1e-6)
