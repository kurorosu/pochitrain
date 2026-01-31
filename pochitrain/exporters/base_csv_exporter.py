"""
CSV出力系クラスの共通基底クラス.

InferenceCSVExporter と TrainingMetricsExporter の
共通処理(出力ディレクトリ管理, ロガー設定, ファイル名生成)を提供します.
"""

import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class BaseCSVExporter(ABC):
    """
    CSV出力クラスの共通基底クラス.

    出力ディレクトリの管理, ロガーの設定, タイムスタンプ付きファイル名の生成
    といった共通処理を提供します.

    Args:
        output_dir (str | Path): 出力ディレクトリ
        logger (logging.Logger, optional): ロガーインスタンス
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        logger: Optional[logging.Logger] = None,
    ):
        """BaseCSVExporterを初期化."""
        self.output_dir = Path(output_dir)

        if logger is None:
            self.logger = logging.getLogger(type(self).__qualname__)
        else:
            self.logger = logger

    def _generate_filename(self, prefix: str, filename: Optional[str] = None) -> str:
        """
        タイムスタンプ付きCSVファイル名を生成.

        filename が指定されている場合はそのまま使用し, .csv 拡張子を保証します.
        filename が None の場合は ``{prefix}_{YYYYMMDD_HHMMSS}.csv`` 形式で生成します.

        Args:
            prefix (str): ファイル名のプレフィックス
            filename (str, optional): 明示的なファイル名

        Returns:
            str: .csv 拡張子付きのファイル名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.csv"

        if not filename.endswith(".csv"):
            filename += ".csv"

        return filename

    def _build_output_path(self, filename: str) -> Path:
        """
        出力ファイルのフルパスを構築.

        Args:
            filename (str): ファイル名

        Returns:
            Path: output_dir / filename のパス
        """
        return self.output_dir / filename
