"""ConfigLoaderクラスのテスト.

実際のPython設定ファイルを作成してロードする古典的テスト.
"""

from pathlib import Path

import pytest

from pochitrain.utils.config_loader import ConfigLoader


class TestConfigLoaderLoadConfig:
    """ConfigLoader.load_config()のテスト."""

    def test_load_basic_config(self, tmp_path):
        """基本的な設定ファイルを読み込める."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            'model_name = "resnet18"\nnum_classes = 10\nbatch_size = 32\n',
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)

        assert config["model_name"] == "resnet18"
        assert config["num_classes"] == 10
        assert config["batch_size"] == 32

    def test_load_config_with_string_path(self, tmp_path):
        """文字列パスでも読み込める."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            'device = "cpu"\n',
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(str(config_file))
        assert config["device"] == "cpu"

    def test_excludes_private_variables(self, tmp_path):
        """アンダースコアで始まる変数は除外される."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            '_private = "hidden"\npublic = "visible"\n',
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)
        assert "_private" not in config
        assert config["public"] == "visible"

    def test_excludes_callable(self, tmp_path):
        """関数は除外される."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "value = 42\ndef my_func():\n    return 1\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)
        assert config["value"] == 42
        assert "my_func" not in config

    def test_includes_transform_objects(self, tmp_path):
        """transformsオブジェクト（callableだがtransforms属性を持つ）は含まれる."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "import torchvision.transforms as transforms\n"
            "train_transform = transforms.Compose([\n"
            "    transforms.Resize((224, 224)),\n"
            "    transforms.ToTensor(),\n"
            "])\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)
        assert "train_transform" in config
        assert hasattr(config["train_transform"], "transforms")

    def test_nonexistent_file_raises(self, tmp_path):
        """存在しないファイルでFileNotFoundErrorが発生する."""
        with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
            ConfigLoader.load_config(tmp_path / "nonexistent.py")

    def test_load_config_with_list(self, tmp_path):
        """リスト型の設定値を読み込める."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "class_weights = [1.0, 2.0, 0.5]\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)
        assert config["class_weights"] == [1.0, 2.0, 0.5]

    def test_load_config_with_dict(self, tmp_path):
        """辞書型の設定値を読み込める."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            'scheduler_params = {"step_size": 10, "gamma": 0.1}\n',
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)
        assert config["scheduler_params"] == {"step_size": 10, "gamma": 0.1}

    def test_load_config_with_bool(self, tmp_path):
        """ブール型の設定値を読み込める."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "pretrained = True\nenable_layer_wise_lr = False\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(config_file)
        assert config["pretrained"] is True
        assert config["enable_layer_wise_lr"] is False

    def test_returns_dict(self, tmp_path):
        """戻り値がdict型."""
        config_file = tmp_path / "config.py"
        config_file.write_text("x = 1\n", encoding="utf-8")

        config = ConfigLoader.load_config(config_file)
        assert isinstance(config, dict)
