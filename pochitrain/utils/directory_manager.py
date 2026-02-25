"""
pochitrain.utils.directory_manager: ワークスペース管理クラス.

タイムスタンプ付きワークスペースの作成と管理機能
"""

import shutil
from pathlib import Path
from typing import Optional, Tuple

from .timestamp_utils import (
    find_next_index,
    format_workspace_name,
    get_current_date_str,
)


class PochiWorkspaceManager:
    """
    タイムスタンプ付きワークスペース管理クラス.

    yyyymmdd_xxx 形式のディレクトリ構造を管理し、
    モデル保存、設定ファイル、画像リスト、学習グラフの保存先を提供

    Args:
        base_dir (str): ベースディレクトリのパス (デフォルト: "work_dirs")
    """

    _WORKSPACE_NOT_CREATED_ERROR = (
        "ワークスペースが作成されていません。"
        "create_workspace() を先に呼び出してください."
    )

    def __init__(self, base_dir: str = "work_dirs"):
        """PochiWorkspaceManagerを初期化."""
        self.base_dir = Path(base_dir)
        self.current_workspace: Optional[Path] = None

    def _require_workspace(self) -> Path:
        """現在のワークスペースを返す.

        Returns:
            現在のワークスペース.

        Raises:
            RuntimeError: ワークスペースが未作成の場合.
        """
        if self.current_workspace is None:
            raise RuntimeError(self._WORKSPACE_NOT_CREATED_ERROR)
        return self.current_workspace

    def _create_workspace_path(self) -> Path:
        """ワークスペース本体ディレクトリを生成して返す."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        date_str = get_current_date_str()
        next_index = find_next_index(self.base_dir, date_str)
        workspace_name = format_workspace_name(date_str, next_index)
        workspace_path = self.base_dir / workspace_name

        workspace_path.mkdir(parents=True, exist_ok=True)
        self.current_workspace = workspace_path
        return workspace_path

    def create_workspace(self) -> Path:
        """
        新しいワークスペースを作成.

        yyyymmdd_xxx 形式のディレクトリを作成し、
        必要なサブディレクトリ (models/) も作成

        Returns:
            Path: 作成されたワークスペースのパス

        Examples:
            work_dirs/20241220_001/
            work_dirs/20241220_001/models/
        """
        workspace_path = self._create_workspace_path()

        models_dir = workspace_path / "models"
        models_dir.mkdir(exist_ok=True)

        paths_dir = workspace_path / "paths"
        paths_dir.mkdir(exist_ok=True)

        visualization_dir = workspace_path / "visualization"
        visualization_dir.mkdir(exist_ok=True)

        return workspace_path

    def get_models_dir(self) -> Path:
        """
        モデル保存用ディレクトリのパスを取得.

        Returns:
            Path: モデル保存用ディレクトリのパス

        Raises:
            RuntimeError: ワークスペースが作成されていない場合
        """
        return self._require_workspace() / "models"

    def _get_paths_dir(self) -> Path:
        """
        パス保存用ディレクトリのパスを取得.

        Returns:
            Path: パス保存用ディレクトリのパス

        Raises:
            RuntimeError: ワークスペースが作成されていない場合
        """
        return self._require_workspace() / "paths"

    def get_visualization_dir(self) -> Path:
        """
        可視化ファイル保存用ディレクトリのパスを取得.

        Returns:
            Path: 可視化ファイル保存用ディレクトリのパス

        Raises:
            RuntimeError: ワークスペースが作成されていない場合
        """
        return self._require_workspace() / "visualization"

    def save_config(self, config_path: Path, target_name: str = "config.py") -> Path:
        """
        設定ファイルをワークスペースにコピー.

        Args:
            config_path (Path): コピー元の設定ファイルパス
            target_name (str): コピー先のファイル名

        Returns:
            Path: コピー先のファイルパス

        Raises:
            RuntimeError: ワークスペースが作成されていない場合
            FileNotFoundError: コピー元ファイルが存在しない場合
        """
        workspace = self._require_workspace()

        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        target_path = workspace / target_name
        shutil.copy2(config_path, target_path)

        return target_path

    def save_dataset_paths(
        self, train_paths: list, val_paths: Optional[list] = None
    ) -> Tuple[Path, Optional[Path]]:
        """
        訓練・検証データのパスリストを保存.

        Args:
            train_paths (list): 訓練データのパスリスト
            val_paths (list, optional): 検証データのパスリスト

        Returns:
            Tuple[Path, Optional[Path]]: 保存されたファイルのパス (train.txt, val.txt)

        Raises:
            RuntimeError: ワークスペースが作成されていない場合
        """
        self._require_workspace()

        paths_dir = self._get_paths_dir()
        train_file_path = paths_dir / "train.txt"
        with open(train_file_path, "w", encoding="utf-8") as f:
            for path in train_paths:
                f.write(f"{path}\n")

        val_file_path = None
        if val_paths is not None:
            val_file_path = paths_dir / "val.txt"
            with open(val_file_path, "w", encoding="utf-8") as f:
                for path in val_paths:
                    f.write(f"{path}\n")

        return train_file_path, val_file_path


class InferenceWorkspaceManager(PochiWorkspaceManager):
    """
    推論専用ワークスペース管理クラス.

    PochiWorkspaceManagerを継承し、推論に特化したワークスペース管理を提供.
    訓練用とは異なり、推論結果とメタデータのみを保存.

    Args:
        base_dir (str): ベースディレクトリのパス (デフォルト: "inference_results")
    """

    def __init__(self, base_dir: str = "inference_results"):
        """InferenceWorkspaceManagerを初期化."""
        super().__init__(base_dir)

    def create_workspace(self) -> Path:
        """
        推論専用ワークスペースを作成.

        親クラスのワークスペース作成機能を使用しますが、
        推論専用なのでmodelsディレクトリは作成しません.

        Returns:
            Path: 作成されたワークスペースのパス
        """
        return self._create_workspace_path()
