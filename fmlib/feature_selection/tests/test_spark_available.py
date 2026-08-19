"""Fail fast when pyspark is missing from the runtime."""

from __future__ import annotations

import pytest


def test_pyspark_is_installed() -> None:
    """Cluster/CI smoke: do not hide a missing Spark extra behind skips and fakes."""
    try:
        import pyspark
    except ImportError:
        message = (
            "pyspark is not installed. Feature-selection Spark paths will not work. "
            "Install the spark optional dependency group."
        )
        pytest.fail(message)
    assert pyspark.__version__
