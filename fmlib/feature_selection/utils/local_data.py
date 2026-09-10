"""Общая загрузка Spark/pandas в локальную память с ограничением размера для методов отбора на основе моделей."""

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
    """Числовая выборка в памяти драйвера, общая для LightGBM и BorutaSHAP."""

    frame: pd.DataFrame
    target_col: str
    max_rows: int
    seed: int
    sample_fraction: float | None


def _canonical_row_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Упорядочивает строки по содержимому, чтобы DataFrame не зависел от разбиения на партиции.

    ``toPandas`` объединяет партиции в порядке их индексов, поэтому загруженный
    DataFrame сохраняет порядок строк, возникший при разбиении входных данных, —
    а Spark выбирает это разбиение по числу ядер, доступных при чтении, которое
    меняется между запусками. Затем этот порядок влияет на отложенное разбиение для подбора параметров,
    разбиение на интервалы в LightGBM и перемешивание теневых признаков Boruta, поэтому на тех же данных
    повторный запуск отбирает другие признаки даже при фиксированных значениях seed всех генераторов.

    Сортировка по хэшу строки делает порядок свойством самих данных.
    Строки с совпавшими хэшами побитово идентичны для методов отбора, поэтому стабильная сортировка
    устраняет зависимость от исходного порядка.

    Args:
        frame: DataFrame, загруженный в память драйвера.

    Returns:
        Те же строки в порядке, не зависящем от разбиения на партиции.
    """
    keys = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return frame.iloc[np.argsort(keys, kind="stable")].reset_index(drop=True)


def _stratified_from_context(context: Any | None) -> bool:
    """Стратифицирует локальные выборки, если задача в схеме не является регрессией."""
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
    """Создаёт локальный числовой DataFrame ограниченного размера.

    Пропуски в числовых данных остаются NaN, чтобы LightGBM и BorutaSHAP могли использовать
    встроенную обработку пропусков при разбиениях. Выборка стратифицируется по целевой переменной, если
    ``context.schema.task_type`` не равен ``regression``. Если передан ``context``,
    совместимая выборка от предыдущего метода на драйвере (те же seed /
    max_rows / target) используется повторно вместо второго вызова Spark ``toPandas``.
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
    """Создаёт локальный DataFrame ограниченного размера с сохранением категориальных признаков.

    Категориальные кандидаты остаются строками; пропуски в числовых данных остаются
    NaN. Библиотеки градиентного бустинга со встроенной поддержкой категорий и NaN
    обрабатывают их самостоятельно. Выборка стратифицируется по целевой переменной, если
    ``context.schema.task_type`` не равен ``regression``.
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
    """Выбирает столбцы и строки входных данных Spark и загружает их в память без приведения типов."""
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
    """Выбирает столбцы и строки уже локальных данных pandas без приведения типов."""
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
    """Возвращает подмножество столбцов совместимой кэшированной выборки, если она есть."""
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
    """Сохраняет выборку с наибольшим числом столбцов, чтобы следующий метод мог выбрать нужные."""
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
    """Определяет положительный размер выборки в пределах лимита строк выполнения."""
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
    """Возвращает выборку ограниченного размера для тестового запуска и число строк до и после отбора."""
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
    """Возвращает, соответствуют ли данные интерфейсу pyspark DataFrame."""
    module_name = type(data).__module__
    return (
        module_name.startswith("pyspark")
        and hasattr(data, "select")
        and hasattr(data, "groupBy")
    )


def root_cause(exc: BaseException) -> str:
    """Извлекает краткое сообщение из вложенных исключений Spark и моделей."""
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
    """Выбирает столбцы, проверяет данные, формирует выборку Spark и загружает её в память."""
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
    """Проверяет уже локальные данные pandas и формирует выборку."""
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
    """Формирует выборку Spark ограниченного размера со стратификацией по целевой переменной при необходимости."""
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
    """Формирует выборку ограниченного размера из входных данных pandas."""
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
    """Распределяет выборку точно заданного ограниченного размера пропорционально между целевыми классами."""
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
    """Проверяет обязательные столбцы Spark и физические типы непрерывных признаков."""
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
    """Создаёт ссылку на столбец Spark с поддержкой точек и пробелов."""
    from pyspark.sql import functions as F  # noqa: N812

    escaped = name.replace("`", "")
    return F.col(f"`{escaped}`")
