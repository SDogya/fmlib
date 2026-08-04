import math
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def drop_constant_features(
    df: DataFrame,
    tol: float = 0.98,
    exclude_cols: Optional[List[str]] = None,
    chunk_size: int = 1000,
    target_col: Optional[str] = None,
) -> DataFrame:
    """
    Удаляет константные и квази-константные признаки с батчингом колонок (chunking)
    для предотвращения OOM и Catalyst Codegen Overflow на широких датасетах.

    Колонка удаляется, если наиболее частое НЕ-NULL значение занимает
    не менее tol среди всех НЕ-NULL значений.

    :param df: Исходный DataFrame.
    :param tol: Порог квази-константности [0.0, 1.0].
    :param exclude_cols: Колонки, которые нельзя удалять.
    :param chunk_size: Количество колонок в одном батче агрегации (по умолчанию 100).
    :param target_col: Целевая колонка, которая должна быть сохранена.
    :return: DataFrame без квази-константных колонок.
    """
    if not 0.0 <= tol <= 1.0:
        raise ValueError("tol must be in range [0.0, 1.0]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    exclude = set(exclude_cols or [])

    # Гарантированно добавляем target_col в exclude, если он есть в данных
    if target_col and target_col in df.columns and target_col not in exclude:
        exclude.add(target_col)

    cols_to_check = [c for c in df.columns if c not in exclude]

    if not cols_to_check:
        return df

    # Разбиваем список проверяемых колонок на чанки
    col_chunks = [
        cols_to_check[i : i + chunk_size]
        for i in range(0, len(cols_to_check), chunk_size)
    ]

    modes: dict = {}

    # Шаг 1. Вычисляем моду по батчам
    for chunk in col_chunks:
        mode_exprs = [F.mode(F.col(c)).alias(c) for c in chunk]
        modes_row = df.agg(*mode_exprs).first()
        if modes_row:
            modes.update(modes_row.asDict())

    if not modes:
        return df

    cols_to_drop = set()

    # Шаг 2. Подсчитываем non_null и mode_count по батчам
    for chunk in col_chunks:
        agg_exprs = []
        for c in chunk:
            mode_val = modes.get(c)

            agg_exprs.append(F.count(F.col(c)).alias(f"{c}__non_null"))

            if mode_val is None:
                agg_exprs.append(F.lit(0).alias(f"{c}__mode_count"))
                continue

            # Робастная проверка на NaN для float/double/numpy.float64
            try:
                is_nan = math.isnan(mode_val)
            except (TypeError, ValueError):
                is_nan = False

            if is_nan:
                agg_exprs.append(
                    F.sum(F.isnan(F.col(c)).cast("int")).alias(f"{c}__mode_count")
                )
            else:
                agg_exprs.append(
                    F.sum((F.col(c) == F.lit(mode_val)).cast("int")).alias(
                        f"{c}__mode_count"
                    )
                )

        stats_row = df.agg(*agg_exprs).first()
        if not stats_row:
            continue

        stats = stats_row.asDict()

        for c in chunk:
            non_null = stats.get(f"{c}__non_null") or 0
            mode_count = stats.get(f"{c}__mode_count") or 0

            if non_null and (mode_count / non_null) >= tol:
                cols_to_drop.add(c)

    if not cols_to_drop:
        return df

    cols_to_keep = [c for c in df.columns if c not in cols_to_drop]

    # Гарантированно возвращаем target_col в результат, если он был в исходном df
    if target_col and target_col in df.columns and target_col not in cols_to_keep:
        cols_to_keep.append(target_col)

    return df.select(*cols_to_keep)
