"""Немедленно вызывает ошибку, если pyspark отсутствует в среде выполнения."""

from __future__ import annotations

import pytest


def test_pyspark_is_installed() -> None:
    """Базовая проверка кластера/CI: отсутствие зависимости Spark не скрывается пропусками и подменами."""
    try:
        import pyspark
    except ImportError:
        message = (
            "pyspark is not installed. Feature-selection Spark paths will not work. "
            "Install the spark optional dependency group."
        )
        pytest.fail(message)
    assert pyspark.__version__
