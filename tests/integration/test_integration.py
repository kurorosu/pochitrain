"""
統合テスト
"""

import tempfile
from pathlib import Path

import pytest
import torchvision.transforms as transforms
from PIL import Image

from pochitrain.pochi_dataset import create_data_loaders


class TestConfigLoading:
    """設定ファイル読み込みのテスト"""

    def create_test_config(self, temp_dir: str, config_content: str) -> Path:
        """テスト用設定ファイルを作成"""
        config_path = Path(temp_dir) / "test_config.py"
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_basic_config_loading(self):
        """基本的な設定ファイル読み込みテスト"""
        from pochi import load_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_content = """
# テスト設定ファイル
model_name = "resnet18"
num_classes = 3
pretrained = True
train_data_root = "data/train"
val_data_root = "data/val"
batch_size = 16
num_workers = 2
epochs = 10
learning_rate = 0.001
optimizer = "Adam"
work_dir = "work_dirs"
device = "cpu"
"""
            config_path = self.create_test_config(temp_dir, config_content)

            # 設定ファイルの読み込み
            config = load_config(str(config_path))

            # 設定値の確認
            assert config["model_name"] == "resnet18"
            assert config["num_classes"] == 3
            assert config["pretrained"] is True
            assert config["batch_size"] == 16
            assert config["learning_rate"] == 0.001

    def test_config_loading_with_scheduler(self):
        """スケジューラー設定を含む設定ファイルテスト"""
        from pochi import load_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_content = """
model_name = "resnet34"
num_classes = 5
scheduler = "StepLR"
scheduler_params = {"step_size": 30, "gamma": 0.1}
"""
            config_path = self.create_test_config(temp_dir, config_content)

            config = load_config(str(config_path))

            assert config["scheduler"] == "StepLR"
            assert config["scheduler_params"]["step_size"] == 30
            assert config["scheduler_params"]["gamma"] == 0.1

    def test_config_loading_missing_file(self):
        """存在しない設定ファイルのエラーテスト"""
        from pochi import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.py")


class TestMainWorkflow:
    """メインワークフローの統合テスト"""

    def create_test_data_structure(self, temp_dir: str):
        """テスト用のデータ構造を作成"""
        base_path = Path(temp_dir)

        # train/valディレクトリ構造を作成
        for split in ["train", "val"]:
            for class_name in ["cat", "dog"]:
                class_dir = base_path / split / class_name
                class_dir.mkdir(parents=True)

                # 各クラスに小さな画像ファイルを作成
                num_images = 2 if split == "val" else 3
                for i in range(num_images):
                    img = Image.new("RGB", (64, 64), color=(i * 100, 150, 200))
                    img.save(class_dir / f"{split}_{i}.jpg")

        return str(base_path / "train"), str(base_path / "val")

    def create_test_config_file(
        self, temp_dir: str, train_root: str, val_root: str
    ) -> Path:
        """テスト用設定ファイルを作成"""
        config_content = f"""
# 統合テスト用設定
model_name = "resnet18"
num_classes = 2
pretrained = False  # テスト高速化のため
train_data_root = "{train_root.replace(chr(92), '/')}"
val_data_root = "{val_root.replace(chr(92), '/')}"
batch_size = 1
num_workers = 0
epochs = 1  # テスト用に短く
learning_rate = 0.1
optimizer = "SGD"
work_dir = "{temp_dir.replace(chr(92), '/')}/work_dirs"
device = "cpu"
"""
        config_path = Path(temp_dir) / "test_config.py"
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_data_loader_creation_workflow(self):
        """データローダー作成ワークフローのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テストデータの作成
            train_root, val_root = self.create_test_data_structure(temp_dir)

            # 必要なtransformを定義
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            # データローダーの作成
            train_loader, val_loader, classes = create_data_loaders(
                train_root=train_root,
                val_root=val_root,
                batch_size=1,
                num_workers=0,
                train_transform=train_transform,
                val_transform=val_transform,
            )

            # 基本的な検証
            assert train_loader is not None
            assert val_loader is not None
            assert len(classes) == 2
            assert "cat" in classes
            assert "dog" in classes

            # データの取得テスト
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))

            assert len(train_batch) == 2  # (images, labels)
            assert len(val_batch) == 2

    def test_trainer_creation_workflow(self, mocker):
        """トレーナー作成ワークフローのテスト"""
        mock_trainer_class = mocker.patch("pochitrain.PochiTrainer")
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_data_structure(temp_dir)
            config_path = self.create_test_config_file(temp_dir, train_root, val_root)

            # 設定の読み込み
            from pochi import load_config

            config = load_config(str(config_path))

            # 必要なtransformを定義
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            # クラス数の動的設定テスト
            train_loader, val_loader, classes = create_data_loaders(
                train_root=config["train_data_root"],
                val_root=config["val_data_root"],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                train_transform=train_transform,
                val_transform=val_transform,
            )

            config["num_classes"] = len(classes)

            # トレーナーが正しい設定で呼び出されることを確認
            from pochitrain import PochiTrainer

            # トレーナーのインスタンス化をテスト
            PochiTrainer(
                model_name=config["model_name"],
                num_classes=config["num_classes"],
                pretrained=config["pretrained"],
                device=config["device"],
                work_dir=config["work_dir"],
            )

            # モックが正しい引数で呼び出されたことを確認
            mock_trainer_class.assert_called_once_with(
                model_name="resnet18",
                num_classes=2,
                pretrained=False,
                device="cpu",
                work_dir=f"{temp_dir}/work_dirs".replace(chr(92), "/"),
            )

    def test_workspace_and_paths_workflow(self):
        """ワークスペースとパス保存のワークフローテスト"""
        from pochitrain.utils.directory_manager import PochiWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = self.create_test_data_structure(temp_dir)

            # ワークスペースマネージャーでワークスペース作成
            workspace_manager = PochiWorkspaceManager(temp_dir)
            workspace_manager.create_workspace()

            # 必要なtransformを定義
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            # データローダー作成
            train_loader, val_loader, classes = create_data_loaders(
                train_root=train_root,
                val_root=val_root,
                batch_size=1,
                num_workers=0,
                train_transform=train_transform,
                val_transform=val_transform,
            )

            # データセットからパスを取得
            train_paths = train_loader.dataset.get_file_paths()
            val_paths = val_loader.dataset.get_file_paths()

            # パスの保存
            train_file, val_file = workspace_manager.save_dataset_paths(
                train_paths, val_paths
            )

            # 保存されたファイルの確認
            assert train_file.exists()
            assert val_file.exists()
            assert train_file.parent.name == "paths"

            # ファイル内容の確認
            saved_train_paths = (
                train_file.read_text(encoding="utf-8").strip().split("\n")
            )
            saved_val_paths = val_file.read_text(encoding="utf-8").strip().split("\n")

            assert len(saved_train_paths) == len(train_paths)
            assert len(saved_val_paths) == len(val_paths)

    def test_logger_integration(self):
        """ログシステム統合テスト"""
        from pochitrain.logging import LoggerManager

        # ロガーマネージャーの作成
        logger_manager = LoggerManager()
        logger = logger_manager.get_logger("integration_test")

        # ログ出力のテスト（エラーが発生しないことを確認）
        logger.info("統合テストでのログメッセージ")
        logger.warning("統合テストでの警告メッセージ")

        # ロガーが適切に管理されていることを確認
        available_loggers = logger_manager.get_available_loggers()
        assert "integration_test" in available_loggers

    def test_error_handling_integration(self):
        """エラーハンドリングの統合テスト"""
        with tempfile.TemporaryDirectory():
            # 必要なtransformを定義
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            # 存在しないデータディレクトリでの処理
            with pytest.raises(Exception):  # 具体的なエラー型は実装による
                create_data_loaders(
                    train_root="/nonexistent/train",
                    val_root="/nonexistent/val",
                    batch_size=1,
                    train_transform=train_transform,
                    val_transform=val_transform,
                )
