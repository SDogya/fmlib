from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.stat import Correlation
import numpy as np


def drop_correlated_features(
    df: DataFrame,
    exclude_cols: Optional[List[str]] = None,
    corr_threshold: float = 0.95,
    target_col: Optional[str] = None,
) -> DataFrame:
    """
    Удаляет коррелированные признаки на основе матрицы корреляций Пирсона.

    Для каждой пары признаков с корреляцией > corr_threshold удаляется один
    (более поздний по порядку в списке колонок).

    :param df: Исходный DataFrame.
    :param exclude_cols: Колонки, которые нельзя удалять.
    :param corr_threshold: Порог корреляции для удаления [0.0, 1.0].
    :param target_col: Целевая колонка, которая должна быть сохранена.
    :return: DataFrame без коррелированных колонок.
    """
    exclude = set(exclude_cols or [])

    # Добавляем target_col в exclude если он не указан и есть в данных
    if target_col and target_col not in exclude and target_col in df.columns:
        exclude.add(target_col)

    # Оставляем только числовые колонки (без exclude и target)
    num_types = ("int", "bigint", "double", "float", "decimal", "smallint", "tinyint")
    all_cols = [c for c in df.columns if c not in exclude]

    # Фильтруем только числовые признаки
    feature_cols = [c for c in all_cols if df.schema[c].dataType.simpleString() in num_types]

    if not feature_cols:
        return df.select(*all_cols)

    # Шаг 0: Отфильтровать колонки с только null значениями
    # Spark Imputer не может работать с колонками где все значения null
    agg_exprs = [F.count(F.col(c)).alias(c) for c in feature_cols]
    counts = df.agg(*agg_exprs).first().asDict()
    feature_cols_valid = [c for c in feature_cols if counts.get(c, 0) > 0]

    if not feature_cols_valid:
        return df.select(*all_cols)

    # Шаг 1: Заполнение пропусков медианой (как в correlation_selection.ipynb)
    imputer = Imputer(strategy="median", inputCols=feature_cols_valid, outputCols=feature_cols_valid)
    df_filled = imputer.fit(df).transform(df)

    # Шаг 2: Преобразование в векторный вид для расчета корреляций
    assembler = VectorAssembler(inputCols=feature_cols_valid, outputCol="features")
    vector_df = assembler.transform(df_filled)

    # Шаг 3: Расчет матрицы корреляций
    matrix_row = Correlation.corr(vector_df, "features", method="pearson").head()
    corr_matrix = matrix_row[0].toArray()

    # numpy матрица верхнетреугольного вида (без диагонали)
    upper_tri = np.triu(corr_matrix, k=1)

    # Шаг 4: Удаление по одному из скоррелированных пар
    features_to_drop = set()

    for i in range(len(feature_cols_valid)):
        if feature_cols_valid[i] in features_to_drop:
            continue
        for j in range(len(feature_cols_valid)):
            if abs(upper_tri[i, j]) > corr_threshold:
                # Для простоты удаляем более поздний признак (j)
                features_to_drop.add(feature_cols_valid[j])

    # Пересечение с original feature_cols чтобы сохранить правильный порядок
    features_to_drop = set(c for c in feature_cols if c in features_to_drop)

    if not features_to_drop:
        return df.select(*all_cols)

    # Удаляем коррелированные колонки
    cols_to_keep = [c for c in all_cols if c not in features_to_drop]
    return df.select(*cols_to_keep)
