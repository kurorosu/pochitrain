"""設定ファイル読み込みユーティリティ."""

import importlib.util
from pathlib import Path
from typing import Any, Dict, Union


class ConfigLoader:
    """設定ファイルを読み込むクラス."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """設定ファイル(Python形式)を読み込む.

        Args:
            config_path: 設定ファイルのパス

        Returns:
            設定辞書

        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合
            RuntimeError: 設定ファイルの読み込みに失敗した場合
        """
        config_path_obj = Path(config_path)

        if not config_path_obj.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        spec = importlib.util.spec_from_file_location("config", config_path_obj)
        if spec is None:
            raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {config_path}")

        config_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"設定ファイルのローダーが見つかりません: {config_path}")

        spec.loader.exec_module(config_module)

        config: Dict[str, Any] = {}
        for key in dir(config_module):
            if not key.startswith("_"):
                value = getattr(config_module, key)
                # 関数やメソッドは除外するが, transformsオブジェクトは含める
                if not callable(value) or hasattr(value, "transforms"):
                    config[key] = value

        return config
