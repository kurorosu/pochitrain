"""
PochiWorkspaceManagerのテスト
"""

import tempfile
from pathlib import Path

import pytest

from pochitrain.utils.directory_manager import PochiWorkspaceManager


class TestPochiWorkspaceManager:
    """PochiWorkspaceManagerクラスのテスト"""

    def test_workspace_creation(self):
        """ワークスペース作成の基本テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # ワークスペースが作成されることを確認
            assert workspace.exists()
            assert workspace.is_dir()

            # ワークスペース名の形式確認（yyyymmdd_xxx）
            workspace_name = workspace.name
            assert len(workspace_name) == 12  # yyyymmdd_xxx
            assert "_" in workspace_name

    def test_subdirectories_creation(self):
        """サブディレクトリ（models, paths）の作成テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # modelsディレクトリの確認
            models_dir = workspace / "models"
            assert models_dir.exists()
            assert models_dir.is_dir()

            # pathsディレクトリの確認
            paths_dir = workspace / "paths"
            assert paths_dir.exists()
            assert paths_dir.is_dir()

    def test_get_methods(self):
        """各ディレクトリ取得メソッドのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # get_current_workspace
            current_workspace = manager.get_current_workspace()
            assert current_workspace == workspace

            # get_models_dir
            models_dir = manager.get_models_dir()
            assert models_dir == workspace / "models"
            assert models_dir.exists()

            # get_paths_dir
            paths_dir = manager.get_paths_dir()
            assert paths_dir == workspace / "paths"
            assert paths_dir.exists()

    def test_save_config(self):
        """設定ファイル保存機能のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # テスト用設定ファイルを作成
            test_config_path = Path(temp_dir) / "test_config.py"
            test_config_content = "# Test configuration\ntest_param = 'test_value'\n"
            test_config_path.write_text(test_config_content, encoding="utf-8")

            # 設定ファイルを保存
            saved_path = manager.save_config(test_config_path, "saved_config.py")

            # 保存先の確認
            expected_path = workspace / "saved_config.py"
            assert saved_path == expected_path
            assert saved_path.exists()

            # 内容の確認
            saved_content = saved_path.read_text(encoding="utf-8")
            assert saved_content == test_config_content

    def test_save_dataset_paths(self):
        """データセットパス保存機能のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # テスト用パスリスト
            train_paths = [
                "/data/train/class1/img1.jpg",
                "/data/train/class1/img2.jpg",
                "/data/train/class2/img3.jpg",
            ]
            val_paths = ["/data/val/class1/img4.jpg", "/data/val/class2/img5.jpg"]

            # パスを保存
            train_file, val_file = manager.save_dataset_paths(train_paths, val_paths)

            # ファイルパスの確認
            expected_train_file = workspace / "paths" / "train.txt"
            expected_val_file = workspace / "paths" / "val.txt"

            assert train_file == expected_train_file
            assert val_file == expected_val_file
            assert train_file.exists()
            assert val_file.exists()

            # ファイル内容の確認
            train_content = train_file.read_text(encoding="utf-8").strip().split("\n")
            val_content = val_file.read_text(encoding="utf-8").strip().split("\n")

            assert train_content == train_paths
            assert val_content == val_paths

    def test_save_dataset_paths_train_only(self):
        """訓練データのみのパス保存テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            manager.create_workspace()

            train_paths = ["/data/train/img1.jpg", "/data/train/img2.jpg"]

            # 検証データなしで保存
            train_file, val_file = manager.save_dataset_paths(train_paths, None)

            # 訓練ファイルのみ作成されることを確認
            assert train_file.exists()
            assert val_file is None

    def test_multiple_workspace_creation(self):
        """複数ワークスペース作成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager1 = PochiWorkspaceManager(temp_dir)
            manager2 = PochiWorkspaceManager(temp_dir)

            workspace1 = manager1.create_workspace()
            workspace2 = manager2.create_workspace()

            # 異なるワークスペースが作成されることを確認
            assert workspace1 != workspace2
            assert workspace1.exists()
            assert workspace2.exists()

            # インデックスが増加することを確認（同じ日の場合）
            if workspace1.name[:8] == workspace2.name[:8]:  # 同じ日付
                index1 = int(workspace1.name.split("_")[1])
                index2 = int(workspace2.name.split("_")[1])
                assert index2 > index1

    def test_error_handling_no_workspace(self):
        """ワークスペース未作成時のエラーハンドリング"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            # ワークスペース作成前にメソッドを呼び出すとエラー
            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.get_models_dir()

            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.get_paths_dir()

    def test_error_handling_missing_config(self):
        """存在しない設定ファイルのエラーハンドリング"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            manager.create_workspace()

            # 存在しないファイルを指定
            non_existent_path = Path(temp_dir) / "non_existent.py"

            with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
                manager.save_config(non_existent_path)
