import logging
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split

# Отключаем лишний спам от Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _extract_and_prep_data(
    df: "pyspark.sql.DataFrame",
    target_col: str,
    exclude_cols: List[str],
    max_rows: int,
    sample_fraction: Optional[float],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Нативный pushdown выборки данных в Pandas без UDF и full scan."""
    exclude = set(exclude_cols) | {target_col}
    dtypes = dict(df.dtypes)
    
    # Оставляем только числа
    num_types = ("int", "bigint", "double", "float", "decimal", "smallint", "tinyint")
    features = [c for c in df.columns if c not in exclude and dtypes.get(c, "").startswith(num_types)]
    
    if not features:
        raise ValueError("Нет числовых признаков для анализа.")

    data_df = df.select(*features, target_col)
    
    # Catalyst pushdown: сначала sample, потом жесткий limit для защиты драйвера
    if sample_fraction and 0.0 < sample_fraction < 1.0:
        data_df = data_df.sample(fraction=sample_fraction, seed=42)
        
    pdf = data_df.limit(max_rows).toPandas()
    
    if pdf.empty:
        raise ValueError("Выборка пуста.")

    # Векторизованное заполнение медианой (быстрее, чем apply)
    pdf[features] = pdf[features].fillna(pdf[features].median())
    
    return pdf[features].to_numpy(), pdf[target_col].to_numpy(), features


def select_robust_features(
    df: "pyspark.sql.DataFrame",
    target_col: str = "target",
    exclude_cols: Optional[List[str]] = None,
    n_trials: int = 30,
    n_folds: int = 5,
    max_rows_limit: int = 100_000,
    sample_fraction: Optional[float] = None,
    lgbm_threshold: float = 0.85,
    shap_threshold: float = 0.85,
) -> List[str]:
    """
    Выжимает максимум из данных: подбирает параметры LGBM через Optuna,
    строит K-Fold модель ОДИН РАЗ, снимает с нее встроенный feature_importance и SHAP,
    возвращая пересечение топовых признаков.
    """
    exclude_cols = exclude_cols or []
    X, y, feature_cols = _extract_and_prep_data(df, target_col, exclude_cols, max_rows_limit, sample_fraction)
    
    logger.info(f"Данные собраны: {X.shape[0]} строк, {X.shape[1]} признаков. Запуск Optuna...")

    # 1. СВЕРХБЫСТРАЯ ОПТИМИЗАЦИЯ (Optuna на Hold-out)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': -1, # Отдаем всю многопоточность C++ бекенду LGBM
            'random_state': 42
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        # Оценка на валидационной выборке
        preds = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42
    best_params['verbosity'] = -1

    logger.info(f"Optuna завершена. Лучший AUC: {study.best_value:.4f}. Обучение K-Fold...")

    # 2. ФИНАЛЬНАЯ МОДЕЛЬ И ЭКСТРАКЦИЯ ВАЖНОСТИ (Единый проход K-Fold)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    lgbm_importances = np.zeros(len(feature_cols))
    shap_importances = np.zeros(len(feature_cols))

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid = X[val_idx]
        
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)

        # А. Встроенная важность (split)
        lgbm_importances += model.booster_.feature_importance(importance_type='split') / n_folds

        # Б. SHAP важность (снимаем с той же самой обученной модели!)
        # TreeExplainer работает мгновенно на C++ уровне
        explainer = shap.TreeExplainer(model)
        
        # Ограничиваем размер выборки для SHAP, чтобы не зависнуть на больших данных
        shap_sample = X_valid if len(X_valid) <= 5000 else X_valid[np.random.choice(len(X_valid), 5000, replace=False)]
        shap_vals = explainer.shap_values(shap_sample)

        # Робастный парсинг вывода SHAP (защита от разных версий библиотеки)
        if isinstance(shap_vals, list):
            vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif hasattr(shap_vals, 'values'):
            vals = shap_vals.values
        else:
            vals = shap_vals
            
        if vals.ndim == 3:
            vals = vals[:, :, 1] # Берем класс 1 для бинарной

        # Средний модуль SHAP value по каждому признаку
        shap_importances += np.abs(vals).mean(axis=0) / n_folds

    # 3. АГРЕГАЦИЯ И ОТБОР (По кумулятивной сумме)
    df_imp = pd.DataFrame({
        'feature': feature_cols,
        'lgbm_imp': lgbm_importances,
        'shap_imp': shap_importances
    })

    # Нормализуем важности (в доли от 1.0)
    df_imp['lgbm_norm'] = df_imp['lgbm_imp'] / df_imp['lgbm_imp'].sum()
    df_imp['shap_norm'] = df_imp['shap_imp'] / df_imp['shap_imp'].sum()

    # Сортируем и считаем кумулятивную сумму для LGBM
    df_imp = df_imp.sort_values('lgbm_norm', ascending=False)
    df_imp['lgbm_cumsum'] = df_imp['lgbm_norm'].cumsum()
    lgbm_selected = set(df_imp[df_imp['lgbm_cumsum'] <= lgbm_threshold]['feature'])

    # Сортируем и считаем кумулятивную сумму для SHAP
    df_imp = df_imp.sort_values('shap_norm', ascending=False)
    df_imp['shap_cumsum'] = df_imp['shap_norm'].cumsum()
    shap_selected = set(df_imp[df_imp['shap_cumsum'] <= shap_threshold]['feature'])

    # ПЕРЕСЕЧЕНИЕ
    final_features = list(lgbm_selected & shap_selected)
    
    logger.info(f"Отобрано LGBM: {len(lgbm_selected)}, SHAP: {len(shap_selected)}. Пересечение: {len(final_features)}")
    
    return final_features
