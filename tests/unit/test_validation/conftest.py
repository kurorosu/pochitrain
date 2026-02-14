"""test_validationパッケージ共通のテストヘルパー."""

from unittest.mock import MagicMock


def assert_info_or_debug_called_with(mock_logger: MagicMock, message: str) -> None:
    """INFO/DEBUGのどちらかでメッセージが出ていることを確認する.

    Args:
        mock_logger: モックロガー
        message: 期待するログメッセージ（部分一致）
    """
    info_calls = [call.args[0] for call in mock_logger.info.call_args_list if call.args]
    debug_calls = [
        call.args[0] for call in mock_logger.debug.call_args_list if call.args
    ]
    all_calls = info_calls + debug_calls
    assert any(message in logged_message for logged_message in all_calls), (
        f"Expected substring '{message}' in INFO or DEBUG calls.\n"
        f"INFO calls: {info_calls}\n"
        f"DEBUG calls: {debug_calls}"
    )


def assert_error_called_with_substring(
    mock_logger: MagicMock,
    substring: str,
    *,
    expected_calls: int | None = 1,
) -> None:
    """ERRORログの呼び出し回数と内容（部分一致）を確認."""
    error_calls = [
        str(call.args[0]) for call in mock_logger.error.call_args_list if call.args
    ]
    if expected_calls is not None:
        assert len(error_calls) == expected_calls, (
            f"Expected {expected_calls} ERROR call(s), but got {len(error_calls)}.\n"
            f"ERROR calls: {error_calls}"
        )
    assert any(substring in message for message in error_calls), (
        f"Expected substring '{substring}' in ERROR calls.\n"
        f"ERROR calls: {error_calls}"
    )
