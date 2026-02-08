"""argparse用のカスタム型バリデーション関数."""

import argparse


def positive_int(value: str) -> int:
    """argparse用の正の整数バリデーション.

    Args:
        value (str): コマンドライン引数の文字列値

    Returns:
        int: 変換された正の整数

    Raises:
        argparse.ArgumentTypeError: 値が1未満の場合
    """
    int_value = int(value)
    if int_value < 1:
        raise argparse.ArgumentTypeError(f"1以上の整数を指定してください: {value}")
    return int_value
