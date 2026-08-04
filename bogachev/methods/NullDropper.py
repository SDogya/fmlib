from typing import List, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def drop_null_features(
    df: DataFrame,
    max_null_fraction: float = 0.5,
    exclude_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> DataFrame:
    """
    Удаляет колонки, в которых доля NULL (и NaN для float/double)
    превышает max_null_fraction.
    :param df: Исходный DataFrame.
    :param max_null_fraction: Максимально допустимая доля пропусков.
    :param exclude_cols: Колонки, которые нельзя удалять.
    :param target_col: Целевая колонка, которая должна быть сохранена.
    :return: DataFrame без колонок с высокой долей пропусков.
    """
    if not 0.0 <= max_null_fraction <= 1.0:
        raise ValueError("max_null_fraction must be in range [0.0, 1.0]")
    exclude = set(exclude_cols or [])

    # Гарантированно добавляем target_col в exclude, если он есть в данных
    if target_col and target_col in df.columns and target_col not in exclude:
        exclude.add(target_col)

    cols_to_check = [c for c in df.columns if c not in exclude]
    if not cols_to_check:
        return df
    dtypes = dict(df.dtypes)
    agg_exprs = []
    for c in cols_to_check:
        dtype = dtypes[c]
        if dtype in ("double", "float"):
            agg_exprs.append(
                F.sum(
                    (F.col(c).isNull() | F.isnan(F.col(c))).cast("int")
                ).alias(c)
            )
        else:
            agg_exprs.append(
                F.count(F.col(c)).alias(c)
            )
    agg_exprs.append(F.count("*").alias("__total__"))
    metrics = df.agg(*agg_exprs).first().asDict()
    total_rows = metrics.pop("__total__")
    if total_rows == 0:
        return df
    max_nulls = total_rows * max_null_fraction
    cols_to_drop = []
    for c in cols_to_check:
        if dtypes[c] in ("double", "float"):
            null_count = metrics[c] or 0
        else:
            null_count = total_rows - metrics[c]
        if null_count > max_nulls:
            cols_to_drop.append(c)
    if not cols_to_drop:
        return df
    cols_to_keep = [c for c in df.columns if c not in cols_to_drop]

    # Гарантированно возвращаем target_col в результат, если он был в исходном df
    if target_col and target_col in df.columns and target_col not in cols_to_keep:
        cols_to_keep.append(target_col)

    return df.select(*cols_to_keep)
