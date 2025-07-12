"""
pochitrain.core.config: 設定管理システム

シンプルなPython辞書ベースの設定管理システム
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Config:
    """
    設定管理クラス

    Python辞書ベースの設定管理を行う

    Args:
        config_dict (dict): 設定辞書
        filename (str, optional): 設定ファイル名

    Examples:
        >>> # ファイルから読み込み
        >>> config = Config.from_file('configs/resnet/resnet18_cifar10.py')
        >>> 
        >>> # 辞書から作成
        >>> config = Config({'model': {'type': 'ResNet', 'depth': 18}})
        >>> 
        >>> # 設定値の取得
        >>> model_config = config.model
        >>> model_type = config.model.type
    """

    def __init__(self, config_dict: Dict[str, Any], filename: Optional[str] = None):
        self._config_dict = config_dict
        self._filename = filename

        # 設定辞書を属性として設定
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config(filename={self._filename})"

    def __getitem__(self, key: str) -> Any:
        """辞書のようにアクセス"""
        return self._config_dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """辞書のように設定"""
        self._config_dict[key] = value
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """in演算子をサポート"""
        return key in self._config_dict

    def keys(self):
        """設定のキー一覧を取得"""
        return self._config_dict.keys()

    def values(self):
        """設定の値一覧を取得"""
        return self._config_dict.values()

    def items(self):
        """設定のキー・値ペア一覧を取得"""
        return self._config_dict.items()

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得（デフォルト値付き）

        Args:
            key (str): 設定キー
            default (Any): デフォルト値

        Returns:
            Any: 設定値またはデフォルト値
        """
        return self._config_dict.get(key, default)

    def update(self, other: Union[Dict[str, Any], 'Config']) -> None:
        """
        設定を更新

        Args:
            other (dict or Config): 更新する設定
        """
        if isinstance(other, Config):
            other = other._config_dict

        self._config_dict.update(other)

        # 属性も更新
        for key, value in other.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書として取得

        Returns:
            dict: 設定辞書
        """
        return self._config_dict.copy()

    @property
    def filename(self) -> Optional[str]:
        """設定ファイル名を取得"""
        return self._filename

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'Config':
        """
        設定ファイルから読み込み

        Args:
            filename (str or Path): 設定ファイルパス

        Returns:
            Config: 設定オブジェクト

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ImportError: ファイルの読み込みに失敗した場合
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {filename}")

        if not filename.suffix == '.py':
            raise ValueError(f"設定ファイルは .py 拡張子である必要があります: {filename}")

        # モジュールとして読み込み
        spec = importlib.util.spec_from_file_location("config", filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"設定ファイルの読み込みに失敗しました: {filename}")

        config_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(config_module)
        except Exception as e:
            raise ImportError(f"設定ファイルの実行に失敗しました: {filename}. エラー: {e}")

        # 設定辞書を構築
        config_dict = {}
        for key in dir(config_module):
            if not key.startswith('_'):  # プライベート変数を除外
                value = getattr(config_module, key)
                if not callable(value):  # 関数を除外
                    config_dict[key] = value

        return cls(config_dict, str(filename))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        辞書から設定を作成

        Args:
            config_dict (dict): 設定辞書

        Returns:
            Config: 設定オブジェクト
        """
        return cls(config_dict)

    def save(self, filename: Union[str, Path]) -> None:
        """
        設定をファイルに保存

        Args:
            filename (str or Path): 保存先ファイルパス
        """
        filename = Path(filename)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# pochitrain 設定ファイル\n")
            f.write(f"# 生成日時: {self._get_timestamp()}\n")
            if self._filename:
                f.write(f"# 元ファイル: {self._filename}\n")
            f.write("\n")

            for key, value in self._config_dict.items():
                f.write(f"{key} = {repr(value)}\n")

    def _get_timestamp(self) -> str:
        """現在の日時を取得"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def pretty_print(self) -> None:
        """設定を見やすく表示"""
        print(f"Config(filename={self._filename})")
        print("=" * 50)
        self._print_dict(self._config_dict)

    def _print_dict(self, d: Dict[str, Any], indent: int = 0) -> None:
        """辞書を見やすく表示"""
        for key, value in d.items():
            print("  " * indent + f"{key}: ", end="")
            if isinstance(value, dict):
                print()
                self._print_dict(value, indent + 1)
            else:
                print(f"{value}")

    def validate(self) -> None:
        """
        設定の妥当性を検証

        基本的な設定項目の存在確認を行う

        Raises:
            ValueError: 必須設定が不足している場合
        """
        required_keys = ['model', 'dataset', 'training']

        for key in required_keys:
            if key not in self._config_dict:
                raise ValueError(f"必須設定項目 '{key}' が見つかりません。")

        # モデル設定の検証
        if 'type' not in self._config_dict['model']:
            raise ValueError("model 設定に 'type' が指定されていません。")

        # データセット設定の検証
        if 'type' not in self._config_dict['dataset']:
            raise ValueError("dataset 設定に 'type' が指定されていません。")

        # 訓練設定の検証
        training_required = ['batch_size', 'epochs', 'learning_rate']
        for key in training_required:
            if key not in self._config_dict['training']:
                raise ValueError(f"training 設定に '{key}' が指定されていません。")

    def merge(self, other: Union[Dict[str, Any], 'Config']) -> 'Config':
        """
        他の設定とマージした新しい設定を作成

        Args:
            other (dict or Config): マージする設定

        Returns:
            Config: マージされた新しい設定
        """
        if isinstance(other, Config):
            other = other._config_dict

        merged_dict = self._config_dict.copy()
        merged_dict.update(other)

        return Config(merged_dict)
