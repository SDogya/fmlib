"""Shared test doubles for feature selection tests."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


@dataclass
class FakeDataFrame:
    """Minimal DataFrame double with columns and select.

    Used for schema/result tests that only need column names without real data.
    Pipeline tests that go through CorrelationSelector should use pandas DataFrames
    (see ``make_pandas_frame``).
    """

    columns: list[str] = field(default_factory=list)

    def select(self: FakeDataFrame, *cols: str) -> FakeDataFrame:
        return FakeDataFrame(columns=list(cols))


@dataclass
class FakeSparkSession:
    """Minimal SparkSession double accepted by duck-typed checks."""

    version: str = "3.5.0-fake"
    spark_context: object = field(default_factory=object, metadata={"alias": "sparkContext"})

    def __post_init__(self: FakeSparkSession) -> None:
        # Duck-typed SparkSession check looks for sparkContext / version.
        object.__setattr__(self, "sparkContext", self.spark_context)


def make_wide_schema_columns(n_features: int = 40) -> tuple[list[str], list[str], list[str]]:
    """Build categorical/continuous names and full column list for tests."""
    categorical = [f"cat_{i}" for i in range(n_features // 4)]
    continuous = [f"num_{i}" for i in range(n_features - len(categorical))]
    service = ["response", "event_date", "dataset_split", "client_id"]
    columns = categorical + continuous + service
    return categorical, continuous, columns


def make_frame(columns: Sequence[str]) -> FakeDataFrame:
    return FakeDataFrame(columns=list(columns))


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
