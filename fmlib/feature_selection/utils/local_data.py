"""Shared bounded Spark/pandas materialization for local ML selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fmlib.feature_selection.exceptions import BackendError, ExecutionError

_NUMERIC_SPARK_TYPE_NAMES = frozenset(
    {
        "ByteType",
        "ShortType",
        "IntegerType",
        "LongType",
        "FloatType",
        "DoubleType",
        "DecimalType",
    },
)


@dataclass
class LocalNumericSample:
    """Driver-local numeric sample shared by LightGBM and BorutaSHAP."""

    frame: pd.DataFrame
    target_col: str
    max_rows: int
    seed: int
    sample_fraction: float | None


def _canonical_row_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Order rows by content so the frame does not depend on partitioning.

    ``toPandas`` concatenates partitions in index order, so a materialized
    frame carries whatever row order the input happened to be split into --
    and Spark chooses that split from the cores available at read time, which
    varies between runs. That order then reaches the tuning hold-out split,
    LightGBM's binning and Boruta's shadow shuffles, so the same data selects
    different features on a rerun even with every seed pinned.

    Sorting by a row hash makes the order a property of the data instead.
    Rows that collide are byte-identical for the selectors, so a stable sort
    leaves nothing order-dependent behind.

    Args:
        frame: Materialized driver-local frame.

    Returns:
        The same rows in a partitioning-independent order.
    """
    keys = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return frame.iloc[np.argsort(keys, kind="stable")].reset_index(drop=True)


def _stratified_from_context(context: Any | None) -> bool:
    """Stratify local samples unless the schema task is regression."""
    if context is None:
        return True
    schema = getattr(context, "schema", None)
    if schema is None:
        return True
    return getattr(schema, "task_type", None) != "regression"


def prepare_numeric_frame(
    data: Any,
    *,
    target_col: str,
    feature_cols: list[str],
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    context: Any | None = None,
) -> pd.DataFrame:
    """Build a bounded local numeric frame.

    Numeric missing values stay as NaN so LightGBM and BorutaSHAP can use
    native missing-value splits. Sampling is stratified by the target unless
    ``context.schema.task_type`` is ``regression``. When ``context`` is given,
    a compatible sample prepared by an earlier driver method (same seed /
    max_rows / target) is reused instead of a second Spark ``toPandas``.
    """
    reused = _reuse_local_numeric_sample(
        context,
        target_col=target_col,
        feature_cols=feature_cols,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    if reused is not None:
        return reused
    stratified = _stratified_from_context(context)
    if is_spark_dataframe(data):
        local = _prepare_spark_frame(
            data,
            target_col=target_col,
            feature_cols=feature_cols,
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            method_name=method_name,
            stratified=stratified,
        )
    elif isinstance(data, pd.DataFrame):
        local = _prepare_pandas_frame(
            data,
            target_col=target_col,
            feature_cols=feature_cols,
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            method_name=method_name,
            stratified=stratified,
        )
    else:
        msg = (
            f"{method_name}: unsupported train split type {type(data)!r}. "
            "Expected a Spark DataFrame or pandas DataFrame."
        )
        raise ExecutionError(msg)

    if local.empty:
        msg = f"{method_name}: the bounded training sample is empty."
        raise ExecutionError(msg)
    if local[target_col].isna().any():
        msg = f"{method_name}: target column contains missing values."
        raise ExecutionError(msg)

    prepared = local.loc[:, feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    conversion_failures = [
        column
        for column in feature_cols
        if local[column].notna().any() and prepared[column].isna().all()
    ]
    if conversion_failures:
        msg = (
            f"{method_name}: FeatureSchema.continuous columns could not be "
            f"converted to a numeric matrix: {conversion_failures}."
        )
        raise ExecutionError(msg)

    result = prepared.copy()
    result[target_col] = local[target_col].to_numpy()
    result = _canonical_row_order(result)
    _store_local_numeric_sample(
        context,
        frame=result,
        target_col=target_col,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    return result


def prepare_mixed_frame(
    data: Any,
    *,
    target_col: str,
    feature_cols: list[str],
    categorical_cols: Sequence[str] = (),
    extra_cols: Sequence[str] = (),
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    context: Any | None = None,
) -> pd.DataFrame:
    """Build a bounded local frame that preserves categorical features.

    Categorical candidates stay as strings; numeric missing values stay as
    NaN. Gradient boosting libraries with native categorical and NaN support
    handle both themselves. Sampling is stratified by the target unless
    ``context.schema.task_type`` is ``regression``.
    """
    categorical_set = set(categorical_cols)
    categorical = [column for column in feature_cols if column in categorical_set]
    numeric = [column for column in feature_cols if column not in categorical_set]
    extras = [
        column
        for column in extra_cols
        if column not in feature_cols and column != target_col
    ]
    columns = [*feature_cols, *extras, target_col]
    stratified = _stratified_from_context(context)

    if is_spark_dataframe(data):
        local = _materialize_spark(
            data,
            columns=columns,
            target_col=target_col,
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            method_name=method_name,
            stratified=stratified,
        )
    elif isinstance(data, pd.DataFrame):
        local = _materialize_pandas(
            data,
            columns=columns,
            target_col=target_col,
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            method_name=method_name,
            stratified=stratified,
        )
    else:
        msg = (
            f"{method_name}: unsupported train split type {type(data)!r}. "
            "Expected a Spark DataFrame or pandas DataFrame."
        )
        raise ExecutionError(msg)

    if local.empty:
        msg = f"{method_name}: the bounded training sample is empty."
        raise ExecutionError(msg)
    if local[target_col].isna().any():
        msg = f"{method_name}: target column contains missing values."
        raise ExecutionError(msg)

    for column in categorical:
        local[column] = local[column].fillna("None").astype(str)

    if numeric:
        converted = local.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
        conversion_failures = [
            column
            for column in numeric
            if local[column].notna().any() and converted[column].isna().all()
        ]
        if conversion_failures:
            msg = (
                f"{method_name}: columns declared as continuous could not be "
                f"converted to numeric: {conversion_failures}. Declare them as "
                "categorical or fix the source types."
            )
            raise ExecutionError(msg)
        for column in numeric:
            local[column] = converted[column]

    return _canonical_row_order(local)


def _materialize_spark(
    frame: Any,
    *,
    columns: list[str],
    target_col: str,
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    stratified: bool = True,
) -> pd.DataFrame:
    """Project, sample, and materialize a Spark input without type coercion."""
    fields = {field.name: field.dataType for field in frame.schema.fields}
    missing = [column for column in columns if column not in fields]
    if missing:
        msg = f"{method_name}: columns missing from train schema: {missing}."
        raise ExecutionError(msg)

    projected = frame.select(
        *[_quoted_col(column).alias(column) for column in columns],
    )
    sampled = _sample_spark(
        projected,
        target_col=target_col,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
        method_name=method_name,
        stratified=stratified,
    )
    try:
        return sampled.toPandas()
    except Exception as exc:  # noqa: BLE001 - Spark/Py4J exception hierarchy
        msg = (
            f"{method_name}: failed to materialize the bounded Spark sample "
            f"as pandas. Root cause: {root_cause(exc)}."
        )
        raise ExecutionError(msg) from exc


def _materialize_pandas(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    target_col: str,
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    stratified: bool = True,
) -> pd.DataFrame:
    """Project and sample an already-local pandas input without type coercion."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"{method_name}: columns missing from train DataFrame: {missing}."
        raise ExecutionError(msg)
    projected = frame.loc[:, columns]
    sampled = _sample_pandas(
        projected,
        target_col=target_col,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
        method_name=method_name,
        stratified=stratified,
    )
    return sampled.reset_index(drop=True)


def _reuse_local_numeric_sample(
    context: Any | None,
    *,
    target_col: str,
    feature_cols: list[str],
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
) -> pd.DataFrame | None:
    """Return a column subset of a compatible cached sample, if any."""
    if context is None:
        return None
    cached = getattr(context, "local_numeric_sample", None)
    if not isinstance(cached, LocalNumericSample):
        return None
    if cached.target_col != target_col:
        return None
    if cached.seed != seed:
        return None
    if cached.max_rows != max_rows:
        return None
    if cached.sample_fraction != sample_fraction:
        return None
    missing = [
        name for name in [*feature_cols, target_col] if name not in cached.frame.columns
    ]
    if missing:
        return None
    return cached.frame.loc[:, [*feature_cols, target_col]].copy()


def _store_local_numeric_sample(
    context: Any | None,
    *,
    frame: pd.DataFrame,
    target_col: str,
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
) -> None:
    """Keep the widest sample so a later method can subset columns."""
    if context is None:
        return
    cached = getattr(context, "local_numeric_sample", None)
    if isinstance(cached, LocalNumericSample) and len(cached.frame.columns) >= len(
        frame.columns,
    ):
        return
    context.local_numeric_sample = LocalNumericSample(
        frame=frame,
        target_col=target_col,
        max_rows=max_rows,
        seed=seed,
        sample_fraction=sample_fraction,
    )


def sample_size(
    total_rows: int,
    max_rows: int,
    sample_fraction: float | None,
) -> int:
    """Resolve a positive sample size bounded by the execution row limit."""
    if total_rows <= 0:
        return 0
    fraction_rows = (
        total_rows
        if sample_fraction is None
        else max(1, int(total_rows * sample_fraction))
    )
    return min(total_rows, max_rows, fraction_rows)


def sample_frame_rows(
    data: Any,
    *,
    target_col: str,
    max_rows: int,
    stratified: bool,
    seed: int,
    method_name: str,
) -> tuple[Any, int, int]:
    """Return a bounded test-run sample and before/after row counts."""
    if is_spark_dataframe(data):
        if stratified and target_col not in data.columns:
            msg = (
                f"{method_name}: target column {target_col!r} is required "
                "for stratified sampling."
            )
            raise ExecutionError(msg)
        total_rows = int(data.count())
        if total_rows <= max_rows:
            return data, total_rows, total_rows
        if stratified:
            sampled = _sample_spark(
                data,
                target_col=target_col,
                max_rows=max_rows,
                sample_fraction=None,
                seed=seed,
                method_name=method_name,
            )
        else:
            try:
                from pyspark.sql import functions
            except ImportError as exc:
                msg = f"{method_name}: pyspark is required for Spark sampling."
                raise BackendError(msg) from exc
            sampled = data.orderBy(functions.rand(seed)).limit(max_rows)
        sampled_rows = int(sampled.count())
        return sampled, total_rows, sampled_rows
    if isinstance(data, pd.DataFrame):
        if stratified and target_col not in data.columns:
            msg = (
                f"{method_name}: target column {target_col!r} is required "
                "for stratified sampling."
            )
            raise ExecutionError(msg)
        total_rows = len(data)
        if total_rows <= max_rows:
            return data.copy(), total_rows, total_rows
        if stratified:
            sampled = _sample_pandas(
                data,
                target_col=target_col,
                max_rows=max_rows,
                sample_fraction=None,
                seed=seed,
                method_name=method_name,
            )
        else:
            sampled = data.sample(n=max_rows, random_state=seed)
        return sampled.reset_index(drop=True), total_rows, len(sampled)
    msg = (
        f"{method_name}: unsupported data type {type(data)!r}. Expected a "
        "Spark DataFrame or pandas DataFrame."
    )
    raise ExecutionError(msg)


def is_spark_dataframe(data: Any) -> bool:
    """Return whether data looks like a pyspark DataFrame."""
    module_name = type(data).__module__
    return (
        module_name.startswith("pyspark")
        and hasattr(data, "select")
        and hasattr(data, "groupBy")
    )


def root_cause(exc: BaseException) -> str:
    """Extract a concise message from nested Spark and model exceptions."""
    java_exc = getattr(exc, "java_exception", None)
    if java_exc is not None:
        return str(java_exc).splitlines()[0]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        return str(cause).splitlines()[0]
    text = str(exc).splitlines()
    return text[0] if text else type(exc).__name__


def _prepare_spark_frame(
    frame: Any,
    *,
    target_col: str,
    feature_cols: list[str],
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    stratified: bool = True,
) -> pd.DataFrame:
    """Project, validate, sample, and materialize a Spark input."""
    fields = {field.name: field.dataType for field in frame.schema.fields}
    _validate_spark_columns(
        fields,
        feature_cols=feature_cols,
        target_col=target_col,
        method_name=method_name,
    )
    projected = frame.select(
        *[
            _quoted_col(column).alias(column)
            for column in [*feature_cols, target_col]
        ],
    )
    sampled = _sample_spark(
        projected,
        target_col=target_col,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
        method_name=method_name,
        stratified=stratified,
    )
    try:
        return sampled.toPandas()
    except Exception as exc:  # noqa: BLE001 - Spark/Py4J exception hierarchy
        msg = (
            f"{method_name}: failed to materialize the bounded Spark sample "
            f"as pandas. Root cause: {root_cause(exc)}."
        )
        raise ExecutionError(msg) from exc


def _prepare_pandas_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    stratified: bool = True,
) -> pd.DataFrame:
    """Validate and sample an already-local pandas input."""
    missing = [
        column
        for column in [*feature_cols, target_col]
        if column not in frame.columns
    ]
    if missing:
        msg = f"{method_name}: columns missing from train DataFrame: {missing}."
        raise ExecutionError(msg)
    projected = frame.loc[:, [*feature_cols, target_col]]
    return _sample_pandas(
        projected,
        target_col=target_col,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
        method_name=method_name,
        stratified=stratified,
    )


def _sample_spark(
    frame: Any,
    *,
    target_col: str,
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    stratified: bool = True,
) -> Any:
    """Apply bounded Spark sampling, stratified by the target when requested."""
    try:
        from pyspark.sql import functions
    except ImportError as exc:
        msg = (
            f"{method_name}: pyspark is required for Spark DataFrames. "
            "Install the spark optional dependency group."
        )
        raise BackendError(msg) from exc

    if not stratified:
        total_rows = int(frame.count())
        target_rows = sample_size(total_rows, max_rows, sample_fraction)
        if target_rows >= total_rows:
            return frame
        return frame.orderBy(functions.rand(seed)).limit(target_rows)

    escaped_target = target_col.replace("`", "")
    with_stratum = frame.withColumn(
        "__fmlib_stratum__",
        functions.col(f"`{escaped_target}`").cast("string"),
    )
    try:
        counts = with_stratum.groupBy("__fmlib_stratum__").count().collect()
        if any(row["__fmlib_stratum__"] is None for row in counts):
            msg = f"{method_name}: target column contains missing values."
            raise ExecutionError(msg)
        total_rows = sum(int(row["count"]) for row in counts)
        target_rows = sample_size(total_rows, max_rows, sample_fraction)
        if target_rows < len(counts):
            msg = (
                f"{method_name}: row limit is too small to retain every target "
                "class during stratified sampling."
            )
            raise ExecutionError(msg)
        if target_rows >= total_rows:
            return with_stratum.drop("__fmlib_stratum__")

        fraction = target_rows / total_rows
        fractions = {row["__fmlib_stratum__"]: fraction for row in counts}
        sampled = with_stratum.sampleBy(
            "__fmlib_stratum__",
            fractions=fractions,
            seed=seed,
        ).drop("__fmlib_stratum__")
        return sampled.limit(target_rows)
    except ExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - Spark/Py4J exception hierarchy
        msg = (
            f"{method_name}: Spark stratified sampling failed. "
            f"Root cause: {root_cause(exc)}."
        )
        raise ExecutionError(msg) from exc


def _sample_pandas(
    frame: pd.DataFrame,
    *,
    target_col: str,
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    stratified: bool = True,
) -> pd.DataFrame:
    """Apply bounded sampling to a pandas input."""
    if frame[target_col].isna().any():
        msg = f"{method_name}: target column contains missing values."
        raise ExecutionError(msg)

    target_rows = sample_size(len(frame), max_rows, sample_fraction)
    if target_rows >= len(frame):
        return frame.copy()
    if not stratified:
        return frame.sample(n=target_rows, random_state=seed).reset_index(drop=True)

    counts = frame[target_col].value_counts(sort=False)
    if target_rows < len(counts):
        msg = (
            f"{method_name}: row limit is too small to retain every target "
            "class during stratified sampling."
        )
        raise ExecutionError(msg)

    sizes = _allocate_strata(counts, target_rows)
    sampled = [
        group.sample(n=int(sizes[class_value]), random_state=seed)
        for class_value, group in frame.groupby(target_col, sort=False)
    ]
    return pd.concat(sampled, ignore_index=True)


def _allocate_strata(counts: pd.Series, target_rows: int) -> pd.Series:
    """Allocate an exact bounded sample proportionally across target classes."""
    ideal = counts.astype(float) * (target_rows / int(counts.sum()))
    sizes = np.floor(ideal).astype(int).clip(lower=1)
    sizes = sizes.combine(counts, min)
    remainder = target_rows - int(sizes.sum())

    if remainder > 0:
        priorities = (ideal - np.floor(ideal)).sort_values(ascending=False)
        while remainder:
            changed = False
            for class_value in priorities.index:
                if sizes[class_value] < counts[class_value]:
                    sizes[class_value] += 1
                    remainder -= 1
                    changed = True
                    if remainder == 0:
                        break
            if not changed:
                break
    elif remainder < 0:
        priorities = sizes.sort_values(ascending=False)
        while remainder:
            changed = False
            for class_value in priorities.index:
                if sizes[class_value] > 1:
                    sizes[class_value] -= 1
                    remainder += 1
                    changed = True
                    if remainder == 0:
                        break
            if not changed:
                break
    return sizes


def _validate_spark_columns(
    fields: dict[str, Any],
    *,
    feature_cols: list[str],
    target_col: str,
    method_name: str,
) -> None:
    """Validate required Spark columns and continuous physical types."""
    required = [*feature_cols, target_col]
    missing = [column for column in required if column not in fields]
    if missing:
        msg = f"{method_name}: columns missing from train schema: {missing}."
        raise ExecutionError(msg)
    non_numeric = [
        f"{column}:{type(fields[column]).__name__}"
        for column in feature_cols
        if type(fields[column]).__name__ not in _NUMERIC_SPARK_TYPE_NAMES
    ]
    if non_numeric:
        msg = (
            f"{method_name}: FeatureSchema.continuous columns must have numeric "
            f"Spark types. Invalid columns: {non_numeric}."
        )
        raise ExecutionError(msg)


def _quoted_col(name: str) -> Any:
    """Build a Spark column reference that tolerates dots and spaces."""
    from pyspark.sql import functions as F  # noqa: N812

    escaped = name.replace("`", "")
    return F.col(f"`{escaped}`")
