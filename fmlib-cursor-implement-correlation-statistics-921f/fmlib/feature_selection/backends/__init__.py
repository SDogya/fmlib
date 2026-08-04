"""Backend adapters package."""

from fmlib.feature_selection.backends.base import BackendCapabilities
from fmlib.feature_selection.backends.spark import (
    SPARK_CAPABILITIES,
    ensure_dataframe,
    ensure_spark_session,
    get_columns,
    project_columns,
)

__all__ = [
    "SPARK_CAPABILITIES",
    "BackendCapabilities",
    "ensure_dataframe",
    "ensure_spark_session",
    "get_columns",
    "project_columns",
]
