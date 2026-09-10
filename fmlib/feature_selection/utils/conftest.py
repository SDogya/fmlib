"""Общие вспомогательные функции тестирования отбора признаков.

Pytest автоматически использует этот модуль для ``utils/tests``. Другие тестовые пакеты импортируют
отсюда фикстуру ``spark`` с областью session через собственный ``tests/conftest.py``.
Тесты Spark используют реальную локальную ``SparkSession``; если pyspark или JVM
отсутствуют, запуск немедленно завершается ошибкой, а не подменяет сессию заглушкой.
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
    """Возвращает реальную локальную SparkSession или завершает тест ошибкой, если Spark не запускается."""
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
    """Запускает одну реальную локальную SparkSession на весь тестовый запуск.

    Автоматически используется через autouse, чтобы тесты варианта pandas также получали реальную сессию в ``StageContext``,
    а последующие заглушки pyspark не могли помешать запуску Spark.
    Отсутствие pyspark или JVM немедленно завершает запуск ошибкой.
    """
    session = require_spark_session()
    try:
        yield session
    finally:
        _shutdown_spark_session()


def make_wide_schema_columns(n_features: int = 40) -> tuple[list[str], list[str], list[str]]:
    """Формирует имена категориальных и непрерывных признаков и полный список столбцов для тестов."""
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
    """Создаёт небольшой pandas DataFrame со случайными числами с плавающей точкой.

    Категориальные столбцы (с префиксом ``cat_``) получают целочисленные коды; остальные —
    числа из стандартного нормального распределения. Служебные столбцы получают постоянные значения-маркеры.
    Результат подходит для интеграционных тестов, использующих реальную
    реализацию CorrelationSelector.

    Args:
        columns: Имена заполняемых столбцов.
        n_rows: Число строк.
        seed: Seed для воспроизводимости.

    Returns:
        pandas DataFrame с ``n_rows`` строками и одним столбцом на каждое имя.
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
    """Показывает наличие pyspark в заголовке pytest до запуска тестов."""
    del config
    try:
        import pyspark
    except ImportError:
        return [
            "pyspark: NOT INSTALLED — tests that need Spark will FAIL. "
            "Install the spark optional extra.",
        ]
    return [f"pyspark: {pyspark.__version__}"]
