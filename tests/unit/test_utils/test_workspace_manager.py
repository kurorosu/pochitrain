"""
PochiWorkspaceManagerのテスト
"""

from pathlib import Path

import pytest

from pochitrain.utils.directory_manager import (
    InferenceWorkspaceManager,
    PochiWorkspaceManager,
)


class TestPochiWorkspaceManager:
    """PochiWorkspaceManagerクラスのテスト"""

    def test_workspace_creation(self, tmp_path):
        """ワークスペース作成の基本テスト"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        workspace = manager.create_workspace()

        assert workspace.exists()
        assert workspace.is_dir()

        workspace_name = workspace.name
        assert len(workspace_name) == 12
        assert "_" in workspace_name

    def test_subdirectories_creation(self, tmp_path):
        """サブディレクトリ（models, paths）の作成テスト"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        workspace = manager.create_workspace()

        models_dir = workspace / "models"
        assert models_dir.exists()
        assert models_dir.is_dir()

        paths_dir = workspace / "paths"
        assert paths_dir.exists()
        assert paths_dir.is_dir()

    def test_get_methods(self, tmp_path):
        """各ディレクトリ取得メソッドのテスト"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        workspace = manager.create_workspace()

        models_dir = manager.get_models_dir()
        assert models_dir == workspace / "models"
        assert models_dir.exists()

    def test_save_config(self, tmp_path):
        """設定ファイル保存機能のテスト"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        workspace = manager.create_workspace()

        test_config_path = Path(temp_dir) / "test_config.py"
        test_config_content = "# Test configuration\ntest_param = 'test_value'\n"
        test_config_path.write_text(test_config_content, encoding="utf-8")

        saved_path = manager.save_config(test_config_path, "saved_config.py")

        expected_path = workspace / "saved_config.py"
        assert saved_path == expected_path
        assert saved_path.exists()

        saved_content = saved_path.read_text(encoding="utf-8")
        assert saved_content == test_config_content

    def test_save_dataset_paths(self, tmp_path):
        """データセットパス保存機能のテスト"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        workspace = manager.create_workspace()

        train_paths = [
            "/data/train/class1/img1.jpg",
            "/data/train/class1/img2.jpg",
            "/data/train/class2/img3.jpg",
        ]
        val_paths = ["/data/val/class1/img4.jpg", "/data/val/class2/img5.jpg"]

        train_file, val_file = manager.save_dataset_paths(train_paths, val_paths)

        expected_train_file = workspace / "paths" / "train.txt"
        expected_val_file = workspace / "paths" / "val.txt"

        assert train_file == expected_train_file
        assert val_file == expected_val_file
        assert train_file.exists()
        assert val_file.exists()

        train_content = train_file.read_text(encoding="utf-8").strip().split("\n")
        val_content = val_file.read_text(encoding="utf-8").strip().split("\n")

        assert train_content == train_paths
        assert val_content == val_paths

    def test_save_dataset_paths_train_only(self, tmp_path):
        """訓練データのみのパス保存テスト"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        manager.create_workspace()

        train_paths = ["/data/train/img1.jpg", "/data/train/img2.jpg"]

        train_file, val_file = manager.save_dataset_paths(train_paths, None)

        assert train_file.exists()
        assert val_file is None

    def test_multiple_workspace_creation(self, tmp_path):
        """複数ワークスペース作成のテスト"""
        temp_dir = str(tmp_path)
        manager1 = PochiWorkspaceManager(temp_dir)
        manager2 = PochiWorkspaceManager(temp_dir)

        workspace1 = manager1.create_workspace()
        workspace2 = manager2.create_workspace()

        assert workspace1 != workspace2
        assert workspace1.exists()
        assert workspace2.exists()

        if workspace1.name[:8] == workspace2.name[:8]:
            index1 = int(workspace1.name.split("_")[1])
            index2 = int(workspace2.name.split("_")[1])
            assert index2 > index1

    def test_error_handling_no_workspace(self, tmp_path):
        """ワークスペース未作成時のエラーハンドリング"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)

        with pytest.raises(RuntimeError, match="ワークスペースが作成されていません"):
            manager.get_models_dir()

    def test_error_handling_missing_config(self, tmp_path):
        """存在しない設定ファイルのエラーハンドリング"""
        temp_dir = str(tmp_path)
        manager = PochiWorkspaceManager(temp_dir)
        manager.create_workspace()

        non_existent_path = Path(temp_dir) / "non_existent.py"

        with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
            manager.save_config(non_existent_path)


class TestInferenceWorkspaceManager:
    """InferenceWorkspaceManagerクラスのテスト."""

    def test_create_workspace_creates_only_workspace_directory(self, tmp_path):
        """推論ワークスペースでは本体ディレクトリのみ生成される."""
        temp_dir = str(tmp_path)
        manager = InferenceWorkspaceManager(temp_dir)
        workspace = manager.create_workspace()

        assert workspace.exists()
        assert workspace.is_dir()
        assert not (workspace / "models").exists()
        assert not (workspace / "paths").exists()
        assert not (workspace / "visualization").exists()
