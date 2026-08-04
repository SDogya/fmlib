import logging
from typing import List, Optional

import lightgbm as lgb
import optuna
import pandas as pd
from BorutaShap import BorutaShap
from pyspark.sql import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def select_features_boruta_shap(
    df: DataFrame,
    target_col: str = "target",
    exclude_cols: Optional[List[str]] = None,
    model_type: str = "lgbm",  # "lgbm" или "rf"
    max_rows_limit: int = 100_000,
    sample_fraction: Optional[float] = None,
    optuna_trials: int = 20,
    boruta_trials: int = 50,
    tentative_fix_method: Optional[str] = "rough",  # "rough" или None
) -> List[str]:
    """
    Агрессивный и сверхбыстрый отбор признаков через библиотеку Boruta-SHAP.

    Алгоритм:
    1. Нативный Pushdown данных в Pandas (без full scan).
    2. Быстрая оптимизация гиперпараметров выбранной модели через Optuna (Hold-out).
    3. Запуск готовой библиотеки BorutaShap с оптимизированной моделью (до 50 итераций).
    4. Опциональное разрешение Tentative-признаков (TentativeRoughFix).

    :param df: Исходный PySpark DataFrame.
    :param target_col: Целевая переменная (бинарная).
    :param exclude_cols: Список колонок для исключения (id, даты и т.д.).
    :param model_type: "lgbm" (LightGBM) или "rf" (scikit-learn Random Forest).
    :param max_rows_limit: Жесткий лимит строк для выгрузки на драйвер.
    :param sample_fraction: Доля датасета для сэмплирования перед лимитом.
    :param optuna_trials: Количество итераций подбора параметров.
    :param boruta_trials: Количество итераций генерации теневых признаков в BorutaShap.
    :param tentative_fix_method: Метод обработки спорных признаков ("rough" вызывает .TentativeRoughFix()).
    :return: Список отобранных признаков (accepted).
    """
    valid_models = {"lgbm", "rf"}
    if model_type not in valid_models:
        raise ValueError(f"model_type должен быть одним из {valid_models}")

    exclude = set(exclude_cols or []) | {target_col}
    dtypes = dict(df.dtypes)

    # Оставляем только числовые фичи для BorutaShap
    num_types = ("int", "bigint", "double", "float", "decimal", "smallint", "tinyint")
    feature_cols = [
        c for c in df.columns if c not in exclude and dtypes.get(c, "").startswith(num_types)
    ]

    if not feature_cols:
        raise ValueError("Нет числовых колонок для анализа.")

    data_df = df.select(*feature_cols, target_col)

    # 1. Catalyst Pushdown: Сэмплирование -> Лимитирование
    if sample_fraction and 0.0 < sample_fraction < 1.0:
        data_df = data_df.sample(fraction=sample_fraction, seed=42)

    pdf = data_df.limit(max_rows_limit).toPandas()
    if pdf.empty:
        raise ValueError("Выборка пуста после применения фильтров.")

    # Быстрое векторизованное заполнение нуллов медианой (Boruta/Scikit не переваривают NaN)
    pdf[feature_cols] = pdf[feature_cols].fillna(pdf[feature_cols].median())

    # BorutaShap требует DataFrame (для названий фичей) и Series для таргета
    X = pdf[feature_cols]
    y = pdf[target_col]

    logger.info(
        f"Выборка собрана: {X.shape[0]} строк, {X.shape[1]} признаков. "
        f"Запуск Optuna для модели {model_type.upper()}..."
    )

    # 2. Optuna: Быстрый поиск гиперпараметров на Hold-out сплите
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        if model_type == "lgbm":
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'n_jobs': -1,
                'random_state': 42
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            preds = model.predict_proba(X_val)[:, 1]

        else:  # 'rf'
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'n_jobs': -1,
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]

        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=optuna_trials)

    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42

    if model_type == "lgbm":
        best_params['verbosity'] = -1
        final_model = lgb.LGBMClassifier(**best_params)
    else:
        final_model = RandomForestClassifier(**best_params)

    logger.info(
        f"Optuna завершена. Лучший AUC: {study.best_value:.4f}. Запуск Boruta-SHAP..."
    )

    # 3. Boruta-SHAP отбор
    feature_selector = BorutaShap(
        model=final_model,
        importance_measure='shap',
        classification=True
    )

    # Запуск 50 итераций с теневыми признаками
    feature_selector.fit(
        X=X,
        y=y,
        n_trials=boruta_trials,
        random_state=42,
        verbose=False
    )

    # 4. Обработка "Tentative" (неопределенных) признаков
    if tentative_fix_method == "rough":
        logger.info("Применение TentativeRoughFix...")
        feature_selector.TentativeRoughFix()

    accepted_features = list(feature_selector.accepted)
    logger.info(f"Boruta-SHAP оставил {len(accepted_features)} признаков из {len(feature_cols)}.")

    return accepted_features
