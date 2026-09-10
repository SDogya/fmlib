"""Легковесный адаптер для Spark без обязательной зависимости от pyspark."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from fmlib.feature_selection.backends.base import BackendCapabilities
from fmlib.feature_selection.exceptions import BackendError, SchemaError

SPARK_CAPABILITIES = BackendCapabilities(
    name="spark",
    supports_distributed=True,
    requires_local_materialization=False,
)


def ensure_spark_session(spark: Any) -> Any:
    """Проверяет, что ``spark`` соответствует интерфейсу активной SparkSession.

    В каркасе используется утиная типизация, поэтому импорт основных модулей не требует pyspark.

    Args:
        spark: Проверяемая сессия Spark.

    Returns:
        Тот же объект ``spark``.

    Raises:
        BackendError: Если объект не соответствует интерфейсу сессии Spark.
    """
    if spark is None:
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
    """Проверяет, что ``data`` соответствует интерфейсу Spark DataFrame или его тестового аналога.

    Args:
        data: Проверяемый DataFrame.
        name: Имя аргумента для сообщений об ошибках.

    Returns:
        Тот же объект ``data``.

    Raises:
        BackendError: Если объект не предоставляет пригодный для работы интерфейс столбцов.
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


def get_columns(data: Any) -> list[str]:
    """Возвращает имена столбцов объекта с интерфейсом DataFrame.

    Args:
        data: Объект с интерфейсом DataFrame.

    Returns:
        Список имён столбцов.
    """
    ensure_dataframe(data)
    return list(data.columns)


def project_columns(data: Any, columns: Sequence[str]) -> Any:
    """Выбирает столбцы из объекта с интерфейсом DataFrame.

    Args:
        data: Входной объект с интерфейсом DataFrame.
        columns: Столбцы, которые нужно оставить.

    Returns:
        Объект с выбранными столбцами, полученный через ``select``, если метод доступен; иначе — поверхностная копия
        с обновлённым атрибутом ``columns`` для тестовых аналогов.

    Raises:
        SchemaError: Если запрошенные столбцы отсутствуют.
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
    """Выполняет предварительную проверку ресурсов перед переносом данных из Spark в локальную память.

    Args:
        n_rows: Оценка числа строк, если известна.
        n_columns: Число столбцов для загрузки в память.
        max_local_rows: Заданный лимит числа строк.
        local_memory_limit_gb: Заданный лимит памяти в ГБ.

    Returns:
        Словарь с диагностикой планируемого переноса данных.

    Raises:
        NotImplementedError: Полная загрузка данных в память выходит за рамки каркаса.
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
