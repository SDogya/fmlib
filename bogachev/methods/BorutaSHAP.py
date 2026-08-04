import logging
from typing import Any, Dict, List, Optional, Union

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

# Default grid for LightGBM (mimics the provided config)
DEFAULT_LGBM_PARAMETERS = {
    'n_estimators': {'type': 'int', 'min': 100, 'max': 1400},
    'num_leaves': {'type': 'int', 'min': 8, 'max': 64},
    'max_depth': {'type': 'int', 'min': 4, 'max': 7},
    'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.1, 'log': True},
    'min_child_samples': {'type': 'int', 'min': 16, 'max': 64},
    'subsample': {'type': 'float', 'min': 0.7, 'max': 1.0},
    'colsample_bytree': {'type': 'float', 'min': 0.7, 'max': 1.0},
    'reg_alpha': {'type': 'float', 'min': 0.01, 'max': 1.0, 'log': True},
    'reg_lambda': {'type': 'float', 'min': 0.01, 'max': 1.0, 'log': True},
    'boosting_type': {'type': 'categorical', 'values': ['gbdt', 'dart', 'goss']},
    'bootstrap_type': {'type': 'categorical', 'values': ['MVS', 'Bernoulli', 'Poisson']},
}

# Default Optuna settings
DEFAULT_OPTUNA_PARAMS = {
    'n_trials': 100,
    'n_startup_trials': 10,
    'sampler': 'TPE',
}


def select_features_boruta_shap(
    df: DataFrame,
    target_col: str = "target",
    exclude_cols: Optional[List[str]] = None,
    model_type: str = "lgbm",  # "lgbm" или "rf"
    max_rows_limit: int = 700_000,
    sample_fraction: Optional[float] = None,
    optuna_trials: int = 20,
    boruta_trials: int = 50,
    tentative_fix_method: Optional[str] = "rough",  # "rough" или None
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    optuna_params: Optional[Dict[str, Any]] = None,
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
    :param optuna_trials: Количество итераций подбора параметров (deprecated, используйте optuna_params['n_trials']).
    :param boruta_trials: Количество итераций генерации теневых признаков в BorutaShap.
    :param tentative_fix_method: Метод обработки спорных признаков ("rough" вызывает .TentativeRoughFix()).
    :param parameters: Сетка гиперпараметров для Optuna. Пример: {'n_estimators': {'type': 'int', 'min': 100, 'max': 1400}, 'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.1, 'log': True}}.
    :param optuna_params: Настройки Optuna: n_trials, n_startup_trials, sampler. Пример: {'n_trials': 100, 'sampler': 'TPE'}.
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

    # --- Optuna configuration with backward compatibility ---
    # Merge user-provided params with defaults
    user_parameters = parameters if parameters else {}
    user_optuna_params = optuna_params if optuna_params else {}

    # Determine effective n_trials (backward compatibility)
    n_trials = user_optuna_params.get('n_trials', user_optuna_params.get('niter', optuna_trials))

    # Build parameter grid from defaults + user overrides
    if model_type == "lgbm":
        base_params = DEFAULT_LGBM_PARAMETERS.copy()
        # Remove LightGBM-specific params not supported by sklearn API
        base_params.pop('bootstrap_type', None)  # LightGBM-only
    else:
        # Default for Random Forest
        base_params = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 200},
            'max_depth': {'type': 'int', 'min': 3, 'max': 10},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 10},
        }

    # Merge user parameters (override defaults)
    for key, value in user_parameters.items():
        base_params[key] = value

    # Create sampler based on config
    sampler_name = user_optuna_params.get('sampler', 'TPE').upper()
    if sampler_name == 'TPE':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=user_optuna_params.get('n_startup_trials', 10),
            seed=42
        )
    elif sampler_name == 'GRID':
        sampler = optuna.samplers.GridSampler(search_space={})
    elif sampler_name == 'RANDOM':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)

    def get_param_value(trial: optuna.Trial, param_name: str, config: Dict[str, Any]) -> Any:
        """Generate parameter suggestion based on config type."""
        param_type = config.get('type', 'float')
        
        if param_type == 'int':
            return trial.suggest_int(param_name, config['min'], config['max'])
        elif param_type == 'float':
            log = config.get('log', False)
            return trial.suggest_float(param_name, config['min'], config['max'], log=log)
        elif param_type == 'categorical':
            return trial.suggest_categorical(param_name, config['values'])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        # Add grid-based parameters
        for param_name, param_config in base_params.items():
            params[param_name] = get_param_value(trial, param_name, param_config)

        # Add model-specific fixed params
        if model_type == "lgbm":
            params['objective'] = 'binary'
            params['metric'] = 'auc'
            # LightGBM parameter name mappings
            if 'n_estimators' in params:
                params['num_iterations'] = params.pop('n_estimators')
            if 'max_depth' in params and params['max_depth'] > 0:
                params['max_depth'] = params['max_depth']
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            preds = model.predict_proba(X_val)[:, 1]
        else:  # 'rf'
            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]

        return roc_auc_score(y_val, preds)

    # Create study and run optimization
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

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