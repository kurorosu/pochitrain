"""
PochiImageDatasetのテスト
"""

import tempfile
from pathlib import Path

import pytest
import torchvision.transforms as transforms
from PIL import Image

from pochitrain.pochi_dataset import PochiImageDataset, create_data_loaders


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


class TestTransformValidation:
    """Transform必須バリデーションのテスト"""

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

    def test_train_transform_required(self):
        """train_transform必須のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            # train_transformがNoneの場合はエラー
            with pytest.raises(ValueError, match="train_transform が必須です"):
                create_data_loaders(
                    train_root=train_root,
                    val_root=val_root,  # 検証データも必須
                    batch_size=1,
                    num_workers=0,
                    train_transform=None,  # None → エラー
                    val_transform=transforms.Compose([transforms.ToTensor()]),
                )

    def test_val_transform_required_when_val_data_exists(self):
        """検証データありでval_transform必須のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            # 検証データがあるのにval_transformがNoneの場合はエラー
            with pytest.raises(ValueError, match="val_transform が必須です"):
                create_data_loaders(
                    train_root=train_root,
                    val_root=val_root,  # 検証データあり
                    batch_size=1,
                    num_workers=0,
                    train_transform=train_transform,
                    val_transform=None,  # None → エラー
                )

    def test_val_data_required(self):
        """検証データ必須のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            val_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            # 検証データとval_transformが両方必須
            train_loader, val_loader, classes = create_data_loaders(
                train_root=train_root,
                val_root=val_root,  # 検証データ必須
                batch_size=1,
                num_workers=0,
                train_transform=train_transform,
                val_transform=val_transform,  # 必須
            )

            assert train_loader is not None
            assert val_loader is not None
            assert len(classes) == 2

    def test_valid_transforms_work(self):
        """正しいtransformが設定されている場合は正常動作のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                ]
            )

            # 正しい設定では正常に動作
            train_loader, val_loader, classes = create_data_loaders(
                train_root=train_root,
                val_root=val_root,
                batch_size=1,
                num_workers=0,
                train_transform=train_transform,
                val_transform=val_transform,
            )

            assert train_loader is not None
            assert val_loader is not None
            assert len(classes) == 2

            # データローダーが実際に動作することを確認
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))

            assert train_batch[0].shape[0] == 1  # バッチサイズ
            assert val_batch[0].shape[0] == 1  # バッチサイズ

    def test_transform_validation_error_messages(self):
        """エラーメッセージの内容テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_datasets(temp_dir)

            # train_transform未設定のエラーメッセージ
            with pytest.raises(ValueError) as exc_info:
                create_data_loaders(
                    train_root=train_root,
                    val_root=val_root,
                    batch_size=1,
                    num_workers=0,
                    train_transform=None,
                    val_transform=transforms.Compose([transforms.ToTensor()]),
                )

            assert "train_transform が必須です" in str(exc_info.value)
            assert "configs/pochi_config.py" in str(exc_info.value)

            # val_transform未設定のエラーメッセージ
            train_transform = transforms.Compose([transforms.ToTensor()])
            with pytest.raises(ValueError) as exc_info:
                create_data_loaders(
                    train_root=train_root,
                    val_root=val_root,
                    batch_size=1,
                    num_workers=0,
                    train_transform=train_transform,
                    val_transform=None,
                )

            assert "val_transform が必須です" in str(exc_info.value)
            assert "configs/pochi_config.py" in str(exc_info.value)
