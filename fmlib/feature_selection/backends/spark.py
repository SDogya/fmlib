"""Lightweight Spark-oriented adapter without a hard pyspark dependency."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from fmlib.feature_selection.backends.base import BackendCapabilities
from fmlib.feature_selection.exceptions import BackendError, SchemaError

logger = logging.getLogger(__name__)

SPARK_CAPABILITIES = BackendCapabilities(
    name="spark",
    supports_distributed=True,
    requires_local_materialization=False,
)


def ensure_spark_session(spark: Any, *, required: bool = True) -> Any:
    """Validate that ``spark`` looks like an active SparkSession.

    The module uses duck typing so core imports do not require pyspark.
    No selector reads the session: every stage works off the DataFrames in
    ``StageContext.datasets``, and a Spark DataFrame already carries its own
    session. ``required=False`` therefore lets a fully local pandas run
    proceed without one.

    Args:
        spark: Candidate Spark session, or ``None``.
        required: When False, ``None`` is accepted and returned unchanged.

    Returns:
        The same ``spark`` object.

    Raises:
        BackendError: If the object does not look like a Spark session, or if
            it is ``None`` while ``required`` is set.
    """
    if spark is None:
        if not required:
            return None
        msg = "An active SparkSession is required. Pass spark=spark to fit_select."
        raise BackendError(
            msg,
        )
    module_name = type(spark).__module__
    has_context = hasattr(spark, "sparkContext") or hasattr(spark, "version")
    looks_like_spark = module_name.startswith("pyspark") or has_context
    if not looks_like_spark:
        msg = f"Expected a SparkSession-like object, got {type(spark)!r}. Pass an active pyspark.sql.SparkSession."
        raise BackendError(
            msg,
        )
    return spark


def ensure_dataframe(data: Any, *, name: str = "data") -> Any:
    """Validate that ``data`` looks like a Spark DataFrame or test double.

    Args:
        data: Candidate DataFrame.
        name: Argument name used in error messages.

    Returns:
        The same ``data`` object.

    Raises:
        BackendError: If the object has no usable columns interface.
    """
    if data is None:
        msg = f"{name} must be a Spark DataFrame, got None."
        raise BackendError(msg)
    if not hasattr(data, "columns"):
        msg = f"{name} must expose a columns attribute (Spark DataFrame or compatible). Got {type(data)!r}."
        raise BackendError(
            msg,
        )
    return data


def persist_unless_cached(frame: Any) -> tuple[Any, Callable[[], None]]:
    """Persist ``frame`` unless it already is, and return it with a releaser.

    Selectors that aggregate the same frame once per column batch would
    otherwise recompute its whole lineage on every batch. Persisting is
    best-effort: a cluster that refuses the storage level should still produce
    numbers, just more slowly.

    Args:
        frame: Spark DataFrame (or any object exposing ``persist``).

    Returns:
        Tuple of the frame to use and a no-argument release callable, safe to
        call in a ``finally`` block whether or not persisting happened.
    """
    if getattr(frame, "is_cached", False):
        return frame, lambda: None
    try:
        persisted = frame.persist()
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        logger.debug("Could not persist a frame: %s", exc)
        return frame, lambda: None

    def release() -> None:
        try:
            persisted.unpersist()
        except Exception:  # noqa: BLE001 - release is best-effort
            logger.debug("Could not unpersist a frame.")

    return persisted, release


def get_columns(data: Any) -> list[str]:
    """Return column names from a DataFrame-like object.

    Args:
        data: DataFrame-like object.

    Returns:
        List of column names.
    """
    ensure_dataframe(data)
    return list(data.columns)


def project_columns(data: Any, columns: Sequence[str]) -> Any:
    """Project columns from a DataFrame-like object.

    Args:
        data: Input DataFrame-like object.
        columns: Columns to keep.

    Returns:
        Projected object via ``select`` when available, otherwise a shallow copy
        with updated ``columns`` for test doubles.

    Raises:
        SchemaError: If requested columns are missing.
    """
    ensure_dataframe(data)
    available = set(get_columns(data))
    missing = [name for name in columns if name not in available]
    if missing:
        msg = f"Cannot project missing columns: {missing}."
        raise SchemaError(msg)

    if hasattr(data, "select") and callable(data.select):
        return data.select(*columns)

    if hasattr(data, "columns"):
        try:
            return type(data)(columns=list(columns))
        except TypeError:
            data.columns = list(columns)
            return data
    return list(columns)


def estimate_local_materialization(
    *,
    n_rows: Optional[int],
    n_columns: int,
    max_local_rows: int,
    local_memory_limit_gb: float,
) -> dict[str, Any]:
    """Skeleton capacity check before a Spark → local transition.

    Args:
        n_rows: Estimated row count, if known.
        n_columns: Number of columns to materialize.
        max_local_rows: Configured row limit.
        local_memory_limit_gb: Configured memory limit in GB.

    Returns:
        Diagnostic dictionary describing the planned transition.

    Raises:
        NotImplementedError: Full materialization is out of scope for the skeleton.
    """
    del n_columns, local_memory_limit_gb
    if n_rows is not None and n_rows > max_local_rows:
        from fmlib.feature_selection.exceptions import CapacityError

        msg = (
            f"Local materialization would load {n_rows} rows, exceeding "
            f"max_local_rows={max_local_rows}. Increase sampling or disable the local stage."
        )
        raise CapacityError(
            msg,
        )
    msg = (
        "Controlled Spark → local materialization is not implemented in the skeleton. "
        "Stub selectors operate on candidate name lists only."
    )
    raise NotImplementedError(
        msg,
    )
