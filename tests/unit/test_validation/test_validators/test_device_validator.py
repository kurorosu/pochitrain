"""
DeviceValidatorのテスト.
"""

import pytest

from pochitrain.validation.validators.device_validator import DeviceValidator


@pytest.fixture
def validator():
    """DeviceValidatorのfixture."""
    return DeviceValidator()


def test_device_none_validation_fails(validator, mocker):
    """device設定がNoneの場合はバリデーションが失敗することをテスト."""
    mock_logger = mocker.Mock()
    config = {"device": None}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is False
    # エラーメッセージが出力されることを確認
    assert mock_logger.error.call_count == 2
    mock_logger.error.assert_any_call(
        "device設定が必須です。configs/pochi_train_config.pyでdeviceを'cuda'または'cpu'に設定してください。"
    )
    mock_logger.error.assert_any_call("例: device = 'cuda' または device = 'cpu'")


def test_device_cpu_shows_warning(validator, mocker):
    """device設定が'cpu'の場合は警告メッセージを表示することをテスト."""
    mock_logger = mocker.Mock()
    config = {"device": "cpu"}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    # 警告メッセージが出力されることを確認
    assert mock_logger.warning.call_count == 3
    mock_logger.warning.assert_any_call("⚠️  CPU使用モードで実行中です")
    mock_logger.warning.assert_any_call(
        "⚠️  GPU使用を推奨します（大幅な性能向上が期待できます）"
    )
    mock_logger.warning.assert_any_call(
        "⚠️  GPU使用時: device = 'cuda' に設定してください"
    )


def test_device_cuda_no_warning(validator, mocker):
    """device設定が'cuda'の場合は警告メッセージを表示しないことをテスト."""
    mock_logger = mocker.Mock()
    config = {"device": "cuda"}

    result = validator.validate(config, mock_logger)

    # アサーション
    assert result is True
    # 警告メッセージが出力されないことを確認
    mock_logger.warning.assert_not_called()


def test_device_missing_from_config(validator, mocker):
    """設定辞書にdeviceキーがない場合のテスト."""
    mock_logger = mocker.Mock()
    config = {}  # deviceキーなし

    result = validator.validate(config, mock_logger)

    # アサーション（device=Noneと同じ扱い）
    assert result is False
    assert mock_logger.error.call_count == 2


class TestCudnnBenchmarkValidator:
    """cudnn_benchmark設定のバリデーションテスト."""

    def test_cudnn_benchmark_true_with_cuda(self, validator, mocker):
        """cudnn_benchmark=TrueでCUDA使用時は成功することをテスト."""
        mock_logger = mocker.Mock()
        config = {"device": "cuda", "cudnn_benchmark": True}

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.error.assert_not_called()

    def test_cudnn_benchmark_false_with_cuda(self, validator, mocker):
        """cudnn_benchmark=FalseでCUDA使用時は成功することをテスト."""
        mock_logger = mocker.Mock()
        config = {"device": "cuda", "cudnn_benchmark": False}

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.error.assert_not_called()

    def test_cudnn_benchmark_none_skipped(self, validator, mocker):
        """cudnn_benchmark未設定の場合はスキップされることをテスト."""
        mock_logger = mocker.Mock()
        config = {"device": "cuda"}  # cudnn_benchmarkなし

        result = validator.validate(config, mock_logger)

        assert result is True
        mock_logger.error.assert_not_called()

    def test_cudnn_benchmark_invalid_type_fails(self, validator, mocker):
        """cudnn_benchmarkがbool型以外の場合はエラーになることをテスト."""
        mock_logger = mocker.Mock()
        config = {"device": "cuda", "cudnn_benchmark": "True"}  # 文字列

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "cudnn_benchmark" in error_msg
        assert "bool型" in error_msg

    def test_cudnn_benchmark_int_type_fails(self, validator, mocker):
        """cudnn_benchmarkがint型の場合はエラーになることをテスト."""
        mock_logger = mocker.Mock()
        config = {"device": "cuda", "cudnn_benchmark": 1}  # int

        result = validator.validate(config, mock_logger)

        assert result is False
        mock_logger.error.assert_called_once()

    def test_cudnn_benchmark_true_with_cpu_warns(self, validator, mocker):
        """cudnn_benchmark=TrueでCPU使用時は警告が出ることをテスト."""
        mock_logger = mocker.Mock()
        config = {"device": "cpu", "cudnn_benchmark": True}

        result = validator.validate(config, mock_logger)

        assert result is True
        # CPU使用の警告3回 + cudnn_benchmark警告1回 = 4回
        assert mock_logger.warning.call_count == 4
        warning_msgs = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("cudnn_benchmark=True" in msg for msg in warning_msgs)
