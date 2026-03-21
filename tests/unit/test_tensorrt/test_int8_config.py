"""INT8CalibrationConfigurer のテスト.

実際のファイルシステムと設定ファイルを使用して INT8 設定組み立てを検証する古典派テスト.
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pochitrain.tensorrt.int8_config import (
    INT8CalibrationConfig,
    INT8CalibrationConfigurer,
)


def _write_config(path: Path, content: dict[str, Any]) -> Path:
    """テスト用の Python 設定ファイルを書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in content.items():
        lines.append(f"{key} = {value!r}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _make_calib_data_dir(tmp_path: Path) -> Path:
    """ダミーのキャリブレーションデータディレクトリを作成する."""
    data_dir = tmp_path / "calib_data"
    data_dir.mkdir()
    return data_dir


def _make_onnx_stub(tmp_path: Path) -> Path:
    """ダミーの ONNX ファイルパスを作成する (中身は使わない)."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"stub")
    return onnx_path


class TestConfigureWithExplicitPaths:
    """CLI 引数で明示的にパスが指定された場合のテスト."""

    def test_configure_with_config_path_and_calib_data(self, tmp_path: Path):
        """config_path と calib_data を明示指定で設定が組み立てられる."""
        calib_dir = _make_calib_data_dir(tmp_path)
        config_path = _write_config(
            tmp_path / "config.py",
            {"val_transform": "dummy_transform", "val_data_root": str(calib_dir)},
        )
        onnx_path = _make_onnx_stub(tmp_path)
        output_path = tmp_path / "model.engine"
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        result = configurer.configure(
            config_path=str(config_path),
            calib_data=str(calib_dir),
            input_shape=(3, 224, 224),
            onnx_path=onnx_path,
            output_path=output_path,
            calib_samples=100,
            calib_batch_size=4,
        )

        assert isinstance(result, INT8CalibrationConfig)
        assert result.calib_data_root == str(calib_dir)
        assert result.transform == "dummy_transform"
        assert result.input_shape == (3, 224, 224)
        assert result.batch_size == 4
        assert result.max_samples == 100
        assert result.cache_file == str(output_path.with_suffix(".cache"))

    def test_calib_data_from_config(self, tmp_path: Path):
        """calib_data=None の場合, config の val_data_root を使用する."""
        calib_dir = _make_calib_data_dir(tmp_path)
        config_path = _write_config(
            tmp_path / "config.py",
            {"val_transform": "transform", "val_data_root": str(calib_dir)},
        )
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        result = configurer.configure(
            config_path=str(config_path),
            calib_data=None,
            input_shape=(3, 224, 224),
            onnx_path=_make_onnx_stub(tmp_path),
            output_path=tmp_path / "out.engine",
            calib_samples=50,
            calib_batch_size=1,
        )

        assert result.calib_data_root == str(calib_dir)


class TestConfigureErrors:
    """異常系テスト."""

    def test_missing_calib_data_raises_value_error(self, tmp_path: Path):
        """calib_data も val_data_root もない場合に ValueError."""
        config_path = _write_config(
            tmp_path / "config.py",
            {"val_transform": "transform"},
        )
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        with pytest.raises(ValueError, match="calib-data"):
            configurer.configure(
                config_path=str(config_path),
                calib_data=None,
                input_shape=(3, 224, 224),
                onnx_path=_make_onnx_stub(tmp_path),
                output_path=tmp_path / "out.engine",
                calib_samples=50,
                calib_batch_size=1,
            )

    def test_nonexistent_calib_data_raises_file_not_found(self, tmp_path: Path):
        """キャリブレーションデータが存在しない場合に FileNotFoundError."""
        config_path = _write_config(
            tmp_path / "config.py",
            {"val_transform": "transform"},
        )
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        with pytest.raises(FileNotFoundError, match="見つかりません"):
            configurer.configure(
                config_path=str(config_path),
                calib_data=str(tmp_path / "nonexistent"),
                input_shape=(3, 224, 224),
                onnx_path=_make_onnx_stub(tmp_path),
                output_path=tmp_path / "out.engine",
                calib_samples=50,
                calib_batch_size=1,
            )

    def test_missing_val_transform_raises_value_error(self, tmp_path: Path):
        """val_transform が config にない場合に ValueError."""
        calib_dir = _make_calib_data_dir(tmp_path)
        config_path = _write_config(
            tmp_path / "config.py",
            {"val_data_root": str(calib_dir)},
        )
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        with pytest.raises(ValueError, match="val_transform"):
            configurer.configure(
                config_path=str(config_path),
                calib_data=str(calib_dir),
                input_shape=(3, 224, 224),
                onnx_path=_make_onnx_stub(tmp_path),
                output_path=tmp_path / "out.engine",
                calib_samples=50,
                calib_batch_size=1,
            )

    def test_invalid_config_path_raises_runtime_error(self, tmp_path: Path):
        """存在しない設定ファイルで RuntimeError."""
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        with pytest.raises(RuntimeError, match="設定ファイル読み込みエラー"):
            configurer.configure(
                config_path=str(tmp_path / "nonexistent.py"),
                calib_data=None,
                input_shape=(3, 224, 224),
                onnx_path=_make_onnx_stub(tmp_path),
                output_path=tmp_path / "out.engine",
                calib_samples=50,
                calib_batch_size=1,
            )


class TestConfigureAutoConfig:
    """config_path=None で自動検出する場合のテスト."""

    def test_auto_config_with_valid_workspace(self, tmp_path: Path):
        """ワークスペース構造がある場合に自動検出で設定を読み込む."""
        # work_dirs/xxx/models/model.onnx の構造を作成
        models_dir = tmp_path / "work_dirs" / "20260101_001" / "models"
        models_dir.mkdir(parents=True)
        onnx_path = models_dir / "model.onnx"
        onnx_path.write_bytes(b"stub")

        calib_dir = _make_calib_data_dir(tmp_path)
        # config.py を work_dirs/xxx/ に配置
        _write_config(
            models_dir.parent / "config.py",
            {"val_transform": "auto_transform", "val_data_root": str(calib_dir)},
        )

        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        result = configurer.configure(
            config_path=None,
            calib_data=str(calib_dir),
            input_shape=(3, 224, 224),
            onnx_path=onnx_path,
            output_path=tmp_path / "out.engine",
            calib_samples=50,
            calib_batch_size=1,
        )

        assert result.transform == "auto_transform"


class TestCacheFilePath:
    """キャッシュファイルパスの生成テスト."""

    def test_cache_file_uses_engine_path_suffix(self, tmp_path: Path):
        """キャッシュファイルは output_path の拡張子を .cache に変えたもの."""
        calib_dir = _make_calib_data_dir(tmp_path)
        config_path = _write_config(
            tmp_path / "config.py",
            {"val_transform": "t", "val_data_root": str(calib_dir)},
        )
        output_path = tmp_path / "model_int8.engine"
        logger = logging.getLogger("test_int8")
        configurer = INT8CalibrationConfigurer(logger)

        result = configurer.configure(
            config_path=str(config_path),
            calib_data=str(calib_dir),
            input_shape=(3, 224, 224),
            onnx_path=_make_onnx_stub(tmp_path),
            output_path=output_path,
            calib_samples=50,
            calib_batch_size=1,
        )

        assert result.cache_file.endswith(".cache")
        assert "model_int8" in result.cache_file
