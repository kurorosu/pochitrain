"""統合テスト.

ユニットテストで担保済みの単体挙動は除き,
モジュールの結合点のみを検証する.
"""

import tempfile
from pathlib import Path

import pytest
import torchvision.transforms as transforms

from pochitrain.pochi_dataset import create_data_loaders
from pochitrain.utils import ConfigLoader
from pochitrain.utils.directory_manager import PochiWorkspaceManager


class TestMainWorkflow:
    """メインワークフローの統合テスト."""

    @staticmethod
    def _create_config_file(config_dir: Path, train_root: str, val_root: str) -> Path:
        """テスト用設定ファイルを作成."""
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
work_dir = "{str(config_dir).replace(chr(92), '/')}/work_dirs"
device = "cpu"
"""
        config_path = config_dir / "test_config.py"
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_config_loader_to_data_loader_workflow(
        self, tmp_path, create_dummy_train_val
    ):
        """ConfigLoaderとDataLoaderの結合経路を検証."""
        train_root, val_root = create_dummy_train_val()
        config_path = self._create_config_file(tmp_path, train_root, val_root)
        config = ConfigLoader.load_config(str(config_path))

        train_transform = transforms.Compose([transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.ToTensor()])

        train_loader, val_loader, classes = create_data_loaders(
            train_root=config["train_data_root"],
            val_root=config["val_data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
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
        val_batch = next(iter(val_loader))
        assert len(train_batch) == 2
        assert len(val_batch) == 2

    def test_workspace_and_paths_workflow(self, create_dummy_train_val):
        """ワークスペース作成とパス保存の結合経路を検証."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_root, val_root = create_dummy_train_val()

            workspace_manager = PochiWorkspaceManager(temp_dir)
            workspace_manager.create_workspace()

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

            train_paths = train_loader.dataset.get_file_paths()
            val_paths = val_loader.dataset.get_file_paths()

            train_file, val_file = workspace_manager.save_dataset_paths(
                train_paths, val_paths
            )

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

    def test_error_handling_integration(self):
        """データルート不存在時にFileNotFoundErrorが送出されることを検証."""
        with tempfile.TemporaryDirectory():
            train_transform = transforms.Compose([transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToTensor()])

            with pytest.raises(FileNotFoundError):
                create_data_loaders(
                    train_root="/nonexistent/train",
                    val_root="/nonexistent/val",
                    batch_size=1,
                    pin_memory=False,
                    train_transform=train_transform,
                    val_transform=val_transform,
                )
