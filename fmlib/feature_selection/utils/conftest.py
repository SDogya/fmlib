"""Shared test helpers for feature selection.

Pytest picks this up for ``utils/tests``. Other test packages import the
session ``spark`` fixture from here via their own ``tests/conftest.py``.
Spark tests use a real local ``SparkSession``; if pyspark or the JVM is
missing the run fails immediately instead of substituting a fake session.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd
import pytest

_SPARK_SESSION: Any = None
_PYSPARK_MISSING = (
    "pyspark is not installed. Feature-selection Spark paths will not work. "
    "Install the spark optional dependency group."
)


def require_spark_session() -> Any:
    """Return a real local SparkSession, or fail the test if Spark cannot start."""
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        return _SPARK_SESSION
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        SparkSession = None  # type: ignore[misc, assignment]
    if SparkSession is None:
        pytest.fail(_PYSPARK_MISSING)
    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("fmlib-feature-selection-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")
    except Exception as exc:  # noqa: BLE001 - surface JVM/Spark startup as a test failure
        message = f"Could not start a local SparkSession: {exc}"
        pytest.fail(message)
    _SPARK_SESSION = session
    return session


def _shutdown_spark_session() -> None:
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        _SPARK_SESSION.stop()
        _SPARK_SESSION = None


@pytest.fixture(scope="session", autouse=True)
def spark() -> Iterator[Any]:
    """Start one real local SparkSession for the whole test run.

    Autouse so pandas-path tests still get a real session in ``StageContext``,
    and so later pyspark stubs cannot prevent Spark from starting.
    Missing pyspark or JVM fails the run immediately.
    """
    session = require_spark_session()
    try:
        yield session
    finally:
        _shutdown_spark_session()


def make_wide_schema_columns(n_features: int = 40) -> tuple[list[str], list[str], list[str]]:
    """Build categorical/continuous names and full column list for tests."""
    categorical = [f"cat_{i}" for i in range(n_features // 4)]
    continuous = [f"num_{i}" for i in range(n_features - len(categorical))]
    service = ["response", "event_date", "dataset_split", "client_id"]
    columns = categorical + continuous + service
    return categorical, continuous, columns


def make_pandas_frame(
    columns: Sequence[str],
    *,
    n_rows: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small pandas DataFrame with random float data.

    Categorical columns (prefixed ``cat_``) get integer codes; all others get
    standard-normal floats. Service columns get constant sentinel values.
    The result is suitable for integration tests that go through the real
    CorrelationSelector implementation.

    Args:
        columns: Column names to populate.
        n_rows: Number of rows.
        seed: Random seed for reproducibility.

    Returns:
        pandas DataFrame with ``n_rows`` rows and one column per name.
    """
    rng = random.Random(seed)  # noqa: S311 - test data generation, not cryptographic use
    data: dict[str, list] = {}
    for col in columns:
        if col.startswith("cat_"):
            data[col] = [rng.randint(0, 9) for _ in range(n_rows)]
        elif col in ("response", "event_date", "dataset_split", "client_id"):
            data[col] = [0] * n_rows
        else:
            data[col] = [rng.gauss(0.0, 1.0) for _ in range(n_rows)]
    return pd.DataFrame(data)


def pytest_report_header(config: object) -> list[str]:
    """Show pyspark presence in the pytest header before any test runs."""
    del config
    try:
        import pyspark
    except ImportError:
        return [
            "pyspark: NOT INSTALLED — tests that need Spark will FAIL. "
            "Install the spark optional extra.",
        ]
    return [f"pyspark: {pyspark.__version__}"]
