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

import pyspark.sql.functions as F


def apply_stratified_sampling(
    df: "pyspark.sql.DataFrame",
    target_col: str,
    max_rows: int,
    seed: int = 42,
) -> "pyspark.sql.DataFrame":
    """
    Применяет стратифицированную выборку на основе целевой колонки.
    
    :param df: Исходный DataFrame.
    :param target_col: Целевая колонка для стратификации.
    :param max_rows: Максимальное количество строк после выборки.
    :param seed: Сид для репродуцируемости.
    :return: DataFrame с отсемплированными данными.
    """
    data_df = df.withColumn('_strat_col', F.col(target_col).cast('string'))
    strat_count = data_df.groupBy('_strat_col').count().collect()
    data_len = data_df.count()
    strat_count_dict = {row['_strat_col']: row['count'] / data_len for row in strat_count}
    fraction = max_rows / data_len
    logger.info(f"    Stratified sampling: fraction={fraction:.4f} (max_rows={max_rows}, data_len={data_len:,})")
    logger.info(f"    Stratification stats: {strat_count_dict}")
    
    fractions = {key: fraction for key in strat_count_dict.keys()}
    data_df = data_df.sampleBy('_strat_col', fractions=fractions, seed=seed).drop('_strat_col')
    data_len_after = data_df.count()
    logger.info(f"    Stratified: {data_len:,} -> {data_len_after:,} rows (fraction={data_len_after/data_len:.4f})")
    
    return data_df


def _extract_and_prep_data(
    df: "pyspark.sql.DataFrame",
    target_col: str,
    exclude_cols: List[str],
    max_rows: int,
    sample_fraction: Optional[float],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Нативный pushdown выборки данных в Pandas без UDF и full scan."""
    logger.info(f"    [_extract_and_prep_data] START")
    logger.info(f"    [_extract_and_prep_data] Input df columns: {len(df.columns)}, rows: {df.count():,}")
    
    exclude = set(exclude_cols) | {target_col}
    dtypes = dict(df.dtypes)

    # Оставляем только числа
    num_types = ("int", "bigint", "double", "float", "decimal", "smallint", "tinyint")
    features = [c for c in df.columns if c not in exclude and dtypes.get(c, "").startswith(num_types)]

    if not features:
        raise ValueError("Нет числовых признаков для анализа.")

    logger.info(f"    [_extract_and_prep_data] Features to select: {len(features)}, target: {target_col}")

    data_df = df.select(*features, target_col)
    logger.info(f"    [_extract_and_prep_data] data_df columns: {len(data_df.columns)}")

    logger.info(f"    Found {len(features)} numeric features")

    # Catalyst pushdown: стратифицированная выборка
    if sample_fraction and 0.0 < sample_fraction < 1.0:
        logger.info(f"    Stratified sampling: fraction={sample_fraction}")
        data_df = apply_stratified_sampling(data_df, target_col, int(data_df.count() * sample_fraction), seed=42)
    else:
        # Случайная выборка: используем max_rows для выборки
        data_df = apply_stratified_sampling(data_df, target_col, max_rows, seed=42)

    logger.info(f"    Converting to Pandas DataFrame...")
    pdf = data_df.toPandas()
    logger.info(f"    [_extract_and_prep_data] PDF shape: {pdf.shape}, memory: {pdf.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    logger.info(f"    Filling NaN with median for {len(features)} features")

    # Векторизованное заполнение медианой (быстрее, чем apply)
    pdf[features] = pdf[features].fillna(pdf[features].median())

    logger.info(f"    [_extract_and_prep_data] END - returning X shape: {pdf[features].shape}")
    return pdf[features].to_numpy(), pdf[target_col].to_numpy(), features


def select_robust_features(
    df: "pyspark.sql.DataFrame",
    target_col: str = "target",
    exclude_cols: Optional[List[str]] = None,
    n_trials: int = 3,
    n_folds: int = 3,
    max_rows_limit: int = 250_000,
    sample_fraction: Optional[float] = None,
    lgbm_threshold: float = 0.85,
    shap_threshold: float = 0.85,
    return_importances: bool = False,
    stratified_df: Optional["pyspark.sql.DataFrame"] = None,
) -> List[str] | Dict[str, Any]:
    """
    Выжимает максимум из данных: подбирает параметры LGBM через Optuna,
    строит K-Fold модель ОДИН РАЗ, снимает с нее встроенный feature_importance и SHAP,
    возвращая пересечение топовых признаков.

    :param stratified_df: Если задан, используется этот DataFrame вместо применения выборки.
                          Позволяет использовать один сэмпл для всех методов.
    :param return_importances: Если True, возвращает словарь с:
        - 'selected_features': список отобранных признаков
        - 'lgbm_selected': признаки, отобранные LGBM
        - 'shap_selected': признаки, отобранные SHAP
        - 'lgbm_dropped': признаки, которые LGBM не отобрал
        - 'shap_dropped': признаки, которые SHAP не отобрал
        - 'importances_df': DataFrame с важностями признаков
    """
    logger.info(f"    [select_robust_features] START")
    logger.info(f"    [select_robust_features] input df: {len(df.columns)} cols, {df.count():,} rows")
    logger.info(f"    [select_robust_features] exclude_cols: {len(exclude_cols or [])}")
    logger.info(f"    [select_robust_features] target_col: {target_col}")
    logger.info(f"    [select_robust_features] stratified_df: {stratified_df is not None}")

    exclude_cols = exclude_cols or []
    
    # Если передан stratified_df - используем его, иначе применяем выборку внутри
    if stratified_df is not None:
        logger.info(f"    [select_robust_features] Using provided stratified_df: {stratified_df.count():,} rows")
        df = stratified_df
    
    X, y, feature_cols = _extract_and_prep_data(df, target_col, exclude_cols, max_rows_limit, sample_fraction)

    logger.info(f"    [select_robust_features] After _extract_and_prep_data: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"    [select_robust_features] feature_cols count: {len(feature_cols)}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("  LGBM + SHAP FEATURE SELECTION START")
    logger.info("=" * 80)
    logger.info(f"  Target column: {target_col}")
    logger.info(f"  Parameters: n_folds={n_folds}, max_rows={max_rows_limit}, sample_fraction={sample_fraction}")
    logger.info(f"  Thresholds: LGBM={lgbm_threshold}, SHAP={shap_threshold}")
    logger.info(f"  Data: {X.shape[0]:,} rows, {X.shape[1]} features")
    logger.info("=" * 80)

    # 1. СВЕРХБЫСТРАЯ ОПТИМИЗАЦИЯ (Optuna на Hold-out)
    logger.info("")
    logger.info("  [1/3] STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("  Splitting data: 80% train, 20% validation")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    logger.info(f"    Train: {X_tr.shape[0]:,} rows, Val: {X_val.shape[0]:,} rows")
    logger.info(f"    [select_robust_features] X_tr memory: {X_tr.nbytes / 1024 / 1024:.2f} MB, X_val memory: {X_val.nbytes / 1024 / 1024:.2f} MB")

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
    logger.info(f"    Starting Optuna with {n_trials} trials...")
    best_auc_log = 0
    for trial_idx in range(n_trials):
        def objective_wrapper(trial):
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
                'n_jobs': -1,
                'random_state': 42
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            preds = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_val, preds)
            return auc
        
        trial = study.ask()
        auc = objective_wrapper(trial)
        study.tell(trial, auc)
        
        if auc > best_auc_log:
            best_auc_log = auc
            logger.info(f"    Trial {trial_idx + 1}/{n_trials}: AUC={auc:.4f} (NEW BEST)")
        elif (trial_idx + 1) % 10 == 0:
            logger.info(f"    Trial {trial_idx + 1}/{n_trials}: AUC={auc:.4f}")
    
    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42
    best_params['verbosity'] = -1

    logger.info("")
    logger.info(f"  [1/3] OPTUNA COMPLETE")
    logger.info(f"    Best AUC: {study.best_value:.4f}")
    logger.info(f"    Best params: n_estimators={best_params['n_estimators']}, learning_rate={best_params['learning_rate']:.4f}, max_depth={best_params['max_depth']}, num_leaves={best_params['num_leaves']}, subsample={best_params['subsample']:.2f}, colsample_bytree={best_params['colsample_bytree']:.2f}")

    # 2. ФИНАЛЬНАЯ МОДЕЛЬ И ЭКСТРАКЦИЯ ВАЖНОСТИ (Единый проход K-Fold)
    logger.info("")
    logger.info("  [2/3] STARTING K-FOLD TRAINING")
    logger.info(f"    K={n_folds} folds, StratifiedKFold")
    logger.info(f"    [select_robust_features] Starting K-Fold with X.shape={X.shape}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    lgbm_importances = np.zeros(len(feature_cols))
    shap_importances = np.zeros(len(feature_cols))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid = X[val_idx]
        logger.info(f"    [select_robust_features] Fold {fold_idx}: X_train.shape={X_train.shape}, X_valid.shape={X_valid.shape}")

        logger.info(f"    Fold {fold_idx}/{n_folds}: обучение с {len(X_train):,} строками...")
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        logger.info(f"      -> обучено")

        # А. Встроенная важность (split)
        lgbm_importances += model.booster_.feature_importance(importance_type='split') / n_folds

        # Б. SHAP важность (снимаем с той же самой обученной модели!)
        logger.info(f"      -> SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        logger.info(f"      [select_robust_features] TreeExplainer created")

        # Ограничиваем размер выборки для SHAP, чтобы не зависнуть на больших данных
        shap_sample_size = min(len(X_valid), 5000)
        logger.info(f"      [select_robust_features] X_valid shape: {X_valid.shape}, shap_sample_size: {shap_sample_size}")
        shap_sample = X_valid if len(X_valid) <= 5000 else X_valid[np.random.choice(len(X_valid), shap_sample_size, replace=False)]
        logger.info(f"      -> SHAP на {shap_sample_size} строках...")
        shap_vals = explainer.shap_values(shap_sample)
        logger.info(f"      -> SHAP получено, shape: {type(shap_vals)}")

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
        logger.info(f"    Fold {fold_idx}/{n_folds}: завершено")

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
    df_imp_sorted = df_imp.sort_values('lgbm_norm', ascending=False).copy()
    df_imp_sorted['lgbm_cumsum'] = df_imp_sorted['lgbm_norm'].cumsum()
    lgbm_selected = set(df_imp_sorted[df_imp_sorted['lgbm_cumsum'] <= lgbm_threshold]['feature'])
    logger.info(f"      [select_robust_features] lgbm_selected: {len(lgbm_selected)} features")

    # Сортируем и считаем кумулятивную сумму для SHAP
    df_imp_sorted = df_imp.sort_values('shap_norm', ascending=False).copy()
    df_imp_sorted['shap_cumsum'] = df_imp_sorted['shap_norm'].cumsum()
    shap_selected = set(df_imp_sorted[df_imp_sorted['shap_cumsum'] <= shap_threshold]['feature'])
    logger.info(f"      [select_robust_features] shap_selected: {len(shap_selected)} features")

    # ПЕРЕСЕЧЕНИЕ
    final_features = list(lgbm_selected & shap_selected)
    logger.info(f"      [select_robust_features] final_features (intersection): {len(final_features)} features")

    # Top 10 features по LGBM
    top_lgbm = df_imp.sort_values('lgbm_norm', ascending=False).head(10)
    logger.info("")
    logger.info("  Top 10 features по LGBM:")
    for _, row in top_lgbm.iterrows():
        logger.info(f"    {row['feature']}: {row['lgbm_norm']:.4f}")

    # Top 10 features по SHAP
    top_shap = df_imp.sort_values('shap_norm', ascending=False).head(10)
    logger.info("")
    logger.info("  Top 10 features по SHAP:")
    for _, row in top_shap.iterrows():
        logger.info(f"    {row['feature']}: {row['shap_norm']:.4f}")

    logger.info("")
    logger.info(f"  [3/3] AGGREGATION COMPLETE")
    logger.info(f"    LGBM отобрал: {len(lgbm_selected)} признаков")
    logger.info(f"    SHAP отобрал: {len(shap_selected)} признаков")
    logger.info(f"    Пересечение: {len(final_features)} признаков")

    # Если нужно вернуть дополнительную информацию
    if return_importances:
        lgbm_dropped = set(feature_cols) - lgbm_selected
        shap_dropped = set(feature_cols) - shap_selected
        logger.info(f"      [select_robust_features] lgbm_dropped: {len(lgbm_dropped)}")
        logger.info(f"      [select_robust_features] shap_dropped: {len(shap_dropped)}")

        return {
            'selected_features': final_features,
            'lgbm_selected': list(lgbm_selected),
            'shap_selected': list(shap_selected),
            'lgbm_dropped': list(lgbm_dropped),
            'shap_dropped': list(shap_dropped),
            'importances_df': df_imp,
        }

    logger.info(f"    [select_robust_features] END - returning {len(final_features)} features")
    logger.info("")
    logger.info("=" * 80)
    logger.info("  LGBM + SHAP FEATURE SELECTION END")
    logger.info("=" * 80)
    logger.info(f"  Итого отобрано: {len(final_features)} признаков")
    logger.info("=" * 80)

    logger.info(f"    [select_robust_features] FINAL returning {len(final_features)} features")
    return final_features