"""test_validationパッケージ共通のテストヘルパー."""

from unittest.mock import MagicMock


def assert_info_or_debug_called_with(mock_logger: MagicMock, message: str) -> None:
    """INFO/DEBUGのどちらかでメッセージが出ていることを確認する.

    Args:
        mock_logger: モックロガー
        message: 期待するログメッセージ
    """
    info_calls = [call.args[0] for call in mock_logger.info.call_args_list if call.args]
    debug_calls = [
        call.args[0] for call in mock_logger.debug.call_args_list if call.args
    ]
    assert message in info_calls or message in debug_calls, (
        f"Expected message '{message}' in INFO or DEBUG calls.\n"
        f"INFO calls: {info_calls}\n"
        f"DEBUG calls: {debug_calls}"
    )
