"""
pochitrain.core.registry: レジストリシステム

モデル、データセット、変換処理の登録と管理を行うレジストリシステム
"""

import inspect
from typing import Dict, Type, Any, Optional, Union


class Registry:
    """
    レジストリクラス

    モデル、データセット、変換処理などのクラスを登録・管理する

    Args:
        name (str): レジストリの名前

    Examples:
        >>> MODELS = Registry('models')
        >>> 
        >>> @MODELS.register_module()
        >>> class ResNet:
        ...     pass
        >>> 
        >>> # または
        >>> MODELS.register_module(ResNet)
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})"

    @property
    def name(self) -> str:
        """レジストリの名前を取得"""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        """登録されたモジュールの辞書を取得"""
        return self._module_dict

    def get(self, key: str) -> Type:
        """
        登録されたモジュールを取得

        Args:
            key (str): モジュール名

        Returns:
            Type: 登録されたクラス

        Raises:
            KeyError: 指定されたキーが見つからない場合
        """
        if key not in self._module_dict:
            raise KeyError(f"'{key}' は {self._name} レジストリに登録されていません。"
                           f"利用可能なモジュール: {list(self._module_dict.keys())}")
        return self._module_dict[key]

    def register_module(self,
                        name: Optional[str] = None,
                        module: Optional[Type] = None,
                        force: bool = False) -> Union[Type, callable]:
        """
        モジュールを登録

        Args:
            name (str, optional): 登録名。Noneの場合はクラス名を使用
            module (Type, optional): 登録するクラス
            force (bool): 既存の登録を上書きするかどうか

        Returns:
            デコレータとして使用される場合は関数、直接呼び出しの場合はクラス

        Examples:
            >>> # デコレータとして使用
            >>> @MODELS.register_module()
            >>> class ResNet:
            ...     pass
            >>> 
            >>> # 直接呼び出し
            >>> MODELS.register_module(name='ResNet', module=ResNet)
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name は str 型である必要があります。受け取った型: {type(name)}")

        # 直接呼び出しの場合
        if module is not None:
            self._register_module(module_class=module,
                                  module_name=name,
                                  force=force)
            return module

        # デコレータとして使用される場合
        def _register(cls):
            self._register_module(module_class=cls,
                                  module_name=name,
                                  force=force)
            return cls

        return _register

    def _register_module(self,
                         module_class: Type,
                         module_name: Optional[str] = None,
                         force: bool = False) -> None:
        """
        モジュールを内部的に登録

        Args:
            module_class (Type): 登録するクラス
            module_name (str, optional): 登録名
            force (bool): 既存の登録を上書きするかどうか
        """
        if not inspect.isclass(module_class):
            raise TypeError(f"module_class はクラスである必要があります。受け取った型: {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(f"'{module_name}' は既に {self._name} レジストリに登録されています。"
                           f"上書きする場合は force=True を指定してください。")

        self._module_dict[module_name] = module_class

    def build(self, cfg: Union[dict, str], *args, **kwargs) -> Any:
        """
        設定からモジュールのインスタンスを構築

        Args:
            cfg (dict or str): 設定辞書またはモジュール名
            *args: 位置引数
            **kwargs: キーワード引数

        Returns:
            Any: 構築されたインスタンス

        Examples:
            >>> config = {'type': 'ResNet', 'depth': 18, 'num_classes': 10}
            >>> model = MODELS.build(config)
        """
        if isinstance(cfg, str):
            # 文字列の場合は単純にモジュール名として扱う
            module_class = self.get(cfg)
            return module_class(*args, **kwargs)

        if not isinstance(cfg, dict):
            raise TypeError(f"cfg は dict または str 型である必要があります。受け取った型: {type(cfg)}")

        if 'type' not in cfg:
            raise KeyError("設定に 'type' キーが見つかりません。")

        cfg = cfg.copy()
        module_name = cfg.pop('type')
        module_class = self.get(module_name)

        # 設定の引数とキーワード引数をマージ
        for key, value in cfg.items():
            if key not in kwargs:
                kwargs[key] = value

        return module_class(*args, **kwargs)


# 標準的なレジストリを作成
MODELS = Registry('models')
DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
HOOKS = Registry('hooks')
LOSSES = Registry('losses')
METRICS = Registry('metrics')
