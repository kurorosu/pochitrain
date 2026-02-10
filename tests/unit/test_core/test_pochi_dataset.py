"""
PochiImageDatasetのテスト
"""

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
)


class TestPochiImageDataset:
    """PochiImageDatasetクラスのテスト"""

    def create_test_dataset(self, temp_dir: str, structure: dict) -> Path:
        """テスト用データセットを作成するヘルパーメソッド"""
        base_path = Path(temp_dir)

        for class_name, num_images in structure.items():
            class_dir = base_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_images):
                # 小さなテスト画像を作成
                img = Image.new("RGB", (32, 32), color=(i * 50 % 255, 100, 150))
                img_path = class_dir / f"image_{i}.jpg"
                img.save(img_path)

        return base_path

    def test_basic_dataset_creation(self):
        """基本的なデータセット作成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テストデータの作成
            dataset_path = self.create_test_dataset(
                temp_dir, {"class1": 3, "class2": 2}
            )

            # データセットの作成
            dataset = PochiImageDataset(str(dataset_path))

            # 基本的な属性の確認
            assert len(dataset) == 5  # 3 + 2
            assert len(dataset.classes) == 2
            assert "class1" in dataset.classes
            assert "class2" in dataset.classes

    def test_data_loading(self):
        """データ読み込み機能のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(temp_dir, {"cat": 2, "dog": 3})

            dataset = PochiImageDataset(str(dataset_path))

            # データの取得テスト
            image, label = dataset[0]

            # 画像がテンソルまたはPIL画像であることを確認
            assert image is not None
            assert isinstance(label, int)
            assert 0 <= label < len(dataset.classes)

    def test_get_classes(self):
        """クラス情報取得のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(
                temp_dir, {"apple": 1, "banana": 1, "cherry": 1}
            )

            dataset = PochiImageDataset(str(dataset_path))
            classes = dataset.get_classes()

            # クラス名がアルファベット順にソートされていることを確認
            expected_classes = ["apple", "banana", "cherry"]
            assert classes == expected_classes

    def test_get_class_counts(self):
        """クラス別画像数取得のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(
                temp_dir, {"class_a": 4, "class_b": 2, "class_c": 6}
            )

            dataset = PochiImageDataset(str(dataset_path))
            counts = dataset.get_class_counts()

            assert counts["class_a"] == 4
            assert counts["class_b"] == 2
            assert counts["class_c"] == 6

    def test_get_file_paths(self):
        """ファイルパス取得機能のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(temp_dir, {"test_class": 3})

            dataset = PochiImageDataset(str(dataset_path))
            file_paths = dataset.get_file_paths()

            # 全ファイルパスが取得されることを確認
            assert len(file_paths) == 3

            # パスが文字列であることを確認
            for path in file_paths:
                assert isinstance(path, str)
                assert "test_class" in path
                assert path.endswith(".jpg")

    def test_different_extensions(self):
        """異なる拡張子のファイル対応テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            class_dir = base_path / "mixed_class"
            class_dir.mkdir()

            # 様々な拡張子のファイルを作成
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            for i, ext in enumerate(extensions):
                img = Image.new("RGB", (32, 32), color=(i * 60, 100, 200))
                img_path = class_dir / f"image_{i}{ext}"
                img.save(img_path)

            # サポートされていない拡張子も作成
            (class_dir / "text_file.txt").write_text("not an image")

            dataset = PochiImageDataset(str(base_path))

            # サポートされている画像ファイルのみが読み込まれることを確認
            assert len(dataset) == 4  # .txt は除外される

    def test_custom_extensions(self):
        """カスタム拡張子設定のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            class_dir = base_path / "custom_class"
            class_dir.mkdir()

            # .jpg と .png ファイルを作成
            img_jpg = Image.new("RGB", (32, 32), color=(255, 0, 0))
            img_jpg.save(class_dir / "image.jpg")

            img_png = Image.new("RGB", (32, 32), color=(0, 255, 0))
            img_png.save(class_dir / "image.png")

            # .jpg のみを許可する設定
            dataset = PochiImageDataset(str(base_path), extensions=(".jpg",))

            # .jpg ファイルのみが読み込まれることを確認
            assert len(dataset) == 1

    def test_empty_directory_error(self):
        """空ディレクトリのエラーハンドリング"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 空のディレクトリでデータセット作成を試行
            with pytest.raises(ValueError, match="クラスフォルダが見つかりません"):
                PochiImageDataset(temp_dir)

    def test_no_images_error(self):
        """画像ファイルなしのエラーハンドリング"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # クラスフォルダは作るが画像ファイルは作らない
            (base_path / "empty_class").mkdir()

            with pytest.raises(ValueError, match="画像ファイルが見つかりません"):
                PochiImageDataset(str(base_path))

    def test_nonexistent_directory_error(self):
        """存在しないディレクトリのエラーハンドリング"""
        with pytest.raises(
            FileNotFoundError, match="データディレクトリが見つかりません"
        ):
            PochiImageDataset("/nonexistent/path")


class TestCreateDataLoaders:
    """create_data_loaders関数のテスト"""

    def create_test_datasets(self, temp_dir: str):
        """train/valディレクトリ構造を作成"""
        base_path = Path(temp_dir)

        # trainディレクトリ
        train_dir = base_path / "train"
        for class_name in ["cat", "dog"]:
            class_dir = train_dir / class_name
            class_dir.mkdir(parents=True)
            for i in range(3):
                img = Image.new("RGB", (64, 64), color=(i * 80, 100, 200))
                img.save(class_dir / f"train_{i}.jpg")

        # valディレクトリ
        val_dir = base_path / "val"
        for class_name in ["cat", "dog"]:
            class_dir = val_dir / class_name
            class_dir.mkdir(parents=True)
            for i in range(2):
                img = Image.new("RGB", (64, 64), color=(i * 80, 150, 100))
                img.save(class_dir / f"val_{i}.jpg")

        return str(train_dir), str(val_dir)

    def test_create_data_loaders_basic(self):
        """基本的なデータローダー作成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            # 必要なtransformを追加
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            train_loader, val_loader, classes = create_data_loaders(
                train_root=train_root,
                val_root=val_root,
                batch_size=2,
                num_workers=0,  # テスト用に0に設定
                train_transform=train_transform,
                val_transform=val_transform,
            )

            # データローダーが正しく作成されることを確認
            assert train_loader is not None
            assert val_loader is not None
            assert len(classes) == 2
            assert "cat" in classes
            assert "dog" in classes

            # バッチサイズの確認
            train_batch = next(iter(train_loader))
            assert len(train_batch[0]) <= 2  # バッチサイズ以下

    def test_create_data_loaders_both_required(self):
        """訓練・検証データ両方必須のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            # 必要なtransformを追加
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            train_loader, val_loader, classes = create_data_loaders(
                train_root=train_root,
                val_root=val_root,  # 検証データ必須
                batch_size=1,
                num_workers=0,
                train_transform=train_transform,
                val_transform=val_transform,
            )

            assert train_loader is not None
            assert val_loader is not None
            assert len(classes) == 2

    def test_create_data_loaders_custom_params(self):
        """カスタムパラメータでのデータローダー作成テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            # 必要なtransformを追加
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

            # カスタムパラメータが適用されることを確認
            assert train_loader.batch_size == 1
            assert train_loader.pin_memory is False


class TestDataLoaderCreation:
    """データローダー作成のテスト（バリデーションは別モジュールで実施）"""

    def create_test_datasets(self, temp_dir: str):
        """train/valディレクトリ構造を作成"""
        base_path = Path(temp_dir)

        # trainディレクトリ
        train_dir = base_path / "train"
        for class_name in ["cat", "dog"]:
            class_dir = train_dir / class_name
            class_dir.mkdir(parents=True)
            for i in range(2):
                img = Image.new("RGB", (64, 64), color=(i * 80, 100, 200))
                img.save(class_dir / f"train_{i}.jpg")

        # valディレクトリ
        val_dir = base_path / "val"
        for class_name in ["cat", "dog"]:
            class_dir = val_dir / class_name
            class_dir.mkdir(parents=True)
            for i in range(1):
                img = Image.new("RGB", (64, 64), color=(i * 80, 150, 100))
                img.save(class_dir / f"val_{i}.jpg")

        return str(train_dir), str(val_dir)

    def test_data_loaders_creation_with_valid_transforms(self):
        """有効なtransformでデータローダーが正常作成されることをテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            # 有効なtransformで正常にデータローダーが作成される
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

    def test_complex_transforms_work(self):
        """複雑なtransformで正常動作するテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

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

            # 正常にデータローダーが作成される
            assert train_loader is not None
            assert val_loader is not None
            assert len(classes) == 2


class TestFastInferenceDataset:
    """FastInferenceDatasetクラスのテスト"""

    def create_test_dataset(self, temp_dir: str, structure: dict) -> Path:
        """テスト用データセットを作成するヘルパーメソッド"""
        base_path = Path(temp_dir)

        for class_name, num_images in structure.items():
            class_dir = base_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_images):
                img = Image.new("RGB", (32, 32), color=(i * 50 % 255, 100, 150))
                img_path = class_dir / f"image_{i}.jpg"
                img.save(img_path)

        return base_path

    def test_returns_tensor(self):
        """decode_imageによりテンソルが直接返されることを確認"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(
                temp_dir, {"class1": 2, "class2": 1}
            )

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

    def test_without_transform(self):
        """transform無しでuint8テンソルが返されることを確認"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(temp_dir, {"class1": 1})

            dataset = FastInferenceDataset(str(dataset_path))

            image, label = dataset[0]

            assert isinstance(image, torch.Tensor)
            assert image.dtype == torch.uint8
            assert image.shape[0] == 3

    def test_inherits_pochi_image_dataset(self):
        """PochiImageDatasetの機能を継承していることを確認"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(temp_dir, {"cat": 3, "dog": 2})

            dataset = FastInferenceDataset(str(dataset_path))

            assert isinstance(dataset, PochiImageDataset)
            assert len(dataset) == 5
            assert dataset.get_classes() == ["cat", "dog"]
            assert dataset.get_class_counts() == {"cat": 3, "dog": 2}

    def test_output_matches_pochi_image_dataset(self):
        """PochiImageDatasetとFastInferenceDatasetで同等の出力を確認"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = self.create_test_dataset(temp_dir, {"class1": 2})

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
    """convert_transform_for_fast_inference関数のテスト"""

    def test_replaces_to_tensor(self):
        """ToTensorがConvertImageDtypeに置き換えられることを確認"""
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
        """ToTensor以外のtransformが維持されることを確認"""
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
        """PIL専用transformが含まれる場合にNoneを返すことを確認"""
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
        """Composeでない単体入力の場合に元のtransformが保持されることを確認"""
        result = convert_transform_for_fast_inference(transforms.Resize(224))

        assert result is not None
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], transforms.Resize)
        assert isinstance(result.transforms[1], transforms.ConvertImageDtype)

    def test_no_compose_to_tensor(self):
        """単体ToTensorがConvertImageDtypeに置き換えられることを確認"""
        result = convert_transform_for_fast_inference(transforms.ToTensor())

        assert result is not None
        assert len(result.transforms) == 1
        assert isinstance(result.transforms[0], transforms.ConvertImageDtype)

    def test_no_compose_pil_only_returns_none(self):
        """単体PIL専用transformの場合にNoneを返すことを確認"""
        result = convert_transform_for_fast_inference(transforms.ToPILImage())

        assert result is None

    def test_convert_dtype_is_float32(self):
        """置き換えられたConvertImageDtypeがfloat32であることを確認"""
        original = transforms.Compose([transforms.ToTensor()])

        result = convert_transform_for_fast_inference(original)

        assert result is not None
        convert_dtype = result.transforms[0]
        assert isinstance(convert_dtype, transforms.ConvertImageDtype)
        assert convert_dtype.dtype == torch.float32
