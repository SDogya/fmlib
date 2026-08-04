#!/usr/bin/env python3
"""
Pipeline for sequential feature selection.

"""

import sys
import os
import logging
from datetime import datetime
import yaml
from typing import Dict, List, Optional, Any

sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.insert(0, "/home/datalab/nfs/zaripov/autocampaignxfm")

from pyspark.sql import DataFrame

from tools.spark_session import create_spark_session
from methods import (
    drop_null_features,
    drop_constant_features,
    drop_low_variance_features,
    drop_correlated_features,
    select_robust_features,
    select_features_boruta_shap,

)

# === КОНФИГУРАЦИЯ ===
PATHS_FILE = "pipeline_conf.yaml"
RESULTS_DIR = "results"
PIPELINE_DIR = os.path.join(RESULTS_DIR, "pipeline")


# === ЛОГИРОВАНИЕ ===
def setup_logger(name: str, log_dir: str, timestamp: str) -> logging.Logger:
    """Настройка логгера."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def load_paths(paths_file: str) -> dict:
    """Загрузить пути из YAML файла."""
    with open(paths_file, 'r') as f:
        return yaml.safe_load(f)


def load_dropped_columns(filepath: str) -> List[str]:
    """Загрузить список колонок для дропа из текстового файла."""
    import os.path
    filepath = os.path.expanduser(filepath)  # Раскрыть ~ в путь
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_selected_columns(filepath: str) -> List[str]:
    """Загрузить список ОСТАВЛЕННЫХ колонок из текстового файла (например, stage_0_1_selected.txt)."""
    import os.path
    filepath = os.path.expanduser(filepath)  # Раскрыть ~ в путь
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_dataset_config(config: dict, dataset_name: str) -> dict:
    """Получить конфиг для конкретного датасета."""
    datasets = config.get('datasets', {})
    return datasets.get(dataset_name, {})


def get_order(config: dict, dataset_name: str) -> List[tuple]:
    """
    Получить порядок методов для датасета.
    Возвращает список кортежей: [(method_name, params), ...]
    где params это None или список параметров.
    
    Поддерживает inline параметры в order:
    - 'method_name' -> ('method_name', None)
    - 'method_name', 100 -> ('method_name', [100])
    """
    dataset_config = get_dataset_config(config, dataset_name)
    # Сначала проверяем order для датасета
    if 'order' in dataset_config:
        order = dataset_config['order']
        return _parse_order(order)
    # Иначе берем глобальный order
    global_methods = config.get('global_methods', {})
    order = global_methods.get('order', ['drop_null', 'drop_constant', 'drop_low_variance', 'drop_correlated', 'lgbm_shap'])
    return _parse_order(order)


def _parse_order(order: List) -> List[tuple]:
    """
    Парсит order список, объединяя методы с их параметрами.
    :param order: Список строк и чисел
    :return: Список кортежей (method_name, params)
    """
    result = []
    i = 0
    while i < len(order):
        item = order[i]
        
        # Если это строка - это имя метода
        if isinstance(item, str):
            method_name = item
            params = None
            
            # Следующий элемент может быть параметром (число или список)
            if i + 1 < len(order):
                next_item = order[i + 1]
                if isinstance(next_item, (int, float)):
                    params = [next_item]
                    i += 1  # Пропускаем следующий элемент
            
            result.append((method_name, params))
        i += 1
    
    return result


def get_variance_param(config: dict, dataset_name: str) -> float:
    """Получить min_variance для датасета."""
    dataset_config = get_dataset_config(config, dataset_name)
    if 'variance_param' in dataset_config:
        return dataset_config['variance_param']
    return 0.005  # дефолт


def get_chunk_size(config: dict, dataset_name: str) -> int:
    """Получить chunk_size для датасета."""
    dataset_config = get_dataset_config(config, dataset_name)
    if 'chunk_size' in dataset_config:
        return dataset_config['chunk_size']
    return 1000  # дефолт


def find_target_column(df: DataFrame) -> str:
    """Найти целевую колонку в DataFrame."""
    for col in df.columns:
        if 'target' in col.lower() or col.lower() == 'y':
            return col
    raise ValueError(f"Не найдена целевая колонка в DataFrame. Доступные колонки: {df.columns}")


def find_exclude_columns(df: DataFrame, target_col: str) -> List[str]:
    """Найти колонки для исключения (id, date, split и т.д.).
    target_attr_1 - оставляем (это целевая колонка для lgbm).
    target_attr_2, target_attr_3 и т.д. - удаляем (доп. таргеты).
    """
    exclude = set()
    exclude_patterns = ['id', 'date', 'time', 'split', 'epoch', 'seq', 'report', 'row']

    for col in df.columns:
        col_lower = col.lower()
        # target_attr_1 - оставляем, остальные target_attr_* - удаляем
        if 'target_attr_' in col_lower and col_lower != 'target_attr_1':
            exclude.add(col)
            continue
        # Добавляем целевую колонку в exclude (она не должна дропаться)
        if col == target_col:
            exclude.add(col)
            continue
        for pattern in exclude_patterns:
            if pattern in col_lower:
                exclude.add(col)
                break

    return list(exclude)


def save_columns_to_file(columns: List[str], filepath: str) -> None:
    """Сохранить список колонок в файл."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for col in columns:
            f.write(f"{col}\n")


def get_dropped_columns(original_df: DataFrame, filtered_df: DataFrame) -> List[str]:
    """Получить список выкинутых колонок."""
    original_cols = set(original_df.columns)
    filtered_cols = set(filtered_df.columns)
    return list(original_cols - filtered_cols)


# === ТЕСТИРОВАНИЕ ===
def test_drop_null(df: DataFrame, target_col: str, exclude_cols: List[str],
                   max_null_fraction: float, dataset_name: str, logger: logging.Logger) -> DataFrame:
    """Протестировать drop_null_features."""
    logger.info(f"  Test drop_null with max_null_fraction={max_null_fraction}")
    result = drop_null_features(df, max_null_fraction=max_null_fraction, exclude_cols=exclude_cols, target_col=target_col)
    return result


def test_drop_constant(df: DataFrame, target_col: str, exclude_cols: List[str],
                       tol: float, chunk_size: int, dataset_name: str, logger: logging.Logger) -> DataFrame:
    """Протестировать drop_constant_features."""
    logger.info(f"  Test drop_constant with tol={tol}, chunk_size={chunk_size}")
    result = drop_constant_features(df, tol=tol, exclude_cols=exclude_cols, chunk_size=chunk_size, target_col=target_col)
    return result


def test_drop_variance(df: DataFrame, target_col: str, exclude_cols: List[str],
                       min_variance: float, scale_method: str,
                       dataset_name: str, logger: logging.Logger,
                       chunk_size: int = 1000) -> DataFrame:
    """Протестировать drop_low_variance_features."""
    logger.info(f"  Test drop_low_variance with min_variance={min_variance}, scale={scale_method}, chunk_size={chunk_size}")
    result = drop_low_variance_features(
        df,
        min_variance=min_variance,
        scale_method=scale_method,
        exclude_cols=exclude_cols,
        chunk_size=chunk_size,
        target_col=target_col,
    )
    return result


def test_drop_correlated(df: DataFrame, target_col: str, exclude_cols: List[str],
                         corr_threshold: float, dataset_name: str, logger: logging.Logger) -> DataFrame:
    """Протестировать drop_correlated_features."""
    logger.info(f"  Test drop_correlated with corr_threshold={corr_threshold}")
    result = drop_correlated_features(df, corr_threshold=corr_threshold, exclude_cols=exclude_cols, target_col=target_col)
    return result


def test_drop_boruta_shap(
    df: DataFrame,
    target_col: str,
    exclude_cols: List[str],
    model_type: str = "lgbm",
    boruta_trials: int = 50,
    dataset_name: str = None,
    logger: logging.Logger = None,
    parameters: Optional[Dict[str, Any]] = None,
    optuna_params: Optional[Dict[str, Any]] = None,
) -> dict:
    """Протестировать BorutaSHAP feature selection с Optuna grid search."""
    logger.info(f"  Test boruta_shap with model_type={model_type}, boruta_trials={boruta_trials}")
    logger.info(f"  [test_drop_boruta_shap] df: {len(df.columns)} cols, {df.count():,} rows")
    logger.info(f"  [test_drop_boruta_shap] target_col: {target_col}")
    logger.info(f"  [test_drop_boruta_shap] exclude_cols ({len(exclude_cols)}): {exclude_cols[:10]}...")

    # Логирование Optuna параметров
    if optuna_params:
        logger.info(f"  [test_drop_boruta_shap] Optuna params:")
        logger.info(f"    n_trials: {optuna_params.get('n_trials', 'default')}")
        logger.info(f"    n_startup_trials: {optuna_params.get('n_startup_trials', 'default')}")
        logger.info(f"    sampler: {optuna_params.get('sampler', 'default')}")

    if parameters:
        logger.info(f"  [test_drop_boruta_shap] Parameter grid ({len(parameters)} params):")
        for param_name, param_config in parameters.items():
            param_type = param_config.get('type', 'unknown')
            if param_type == 'categorical':
                values_str = str(param_config.get('values', []))[:50]
                logger.info(f"    {param_name}: {param_type} values={values_str}")
            else:
                logger.info(f"    {param_name}: {param_type} [{param_config.get('min')}, {param_config.get('max')}]")

    result = select_features_boruta_shap(
        df,
        target_col=target_col,
        exclude_cols=exclude_cols,
        model_type=model_type,
        boruta_trials=boruta_trials,
        parameters=parameters,
        optuna_params=optuna_params,
    )

    selected_set = set(result)
    all_cols = set(df.columns)
    dropped = list(all_cols - selected_set - set(exclude_cols))

    return {
        'selected': result,
        'dropped': dropped,
    }


def test_drop_lgbm_shap(
    df: DataFrame,
    target_col: str,
    exclude_cols: List[str],
    threshold: float,
    dataset_name: str,
    logger: logging.Logger,
    stratified_df: DataFrame = None,
) -> dict:
    """Протестировать LGBM + SHAP feature selection."""
    logger.info(f"  Test lgbm_shap with threshold={threshold}")
    logger.info(f"  [test_drop_lgbm_shap] df: {len(df.columns)} cols, {df.count():,} rows")
    logger.info(f"  [test_drop_lgbm_shap] target_col: {target_col}")
    logger.info(f"  [test_drop_lgbm_shap] exclude_cols ({len(exclude_cols)}): {exclude_cols[:10]}...")
    logger.info(f"  [test_drop_lgbm_shap] stratified_df: {stratified_df is not None}")

    result = select_robust_features(
        df,
        target_col=target_col,
        exclude_cols=exclude_cols,
        lgbm_threshold=threshold,
        shap_threshold=threshold,
        return_importances=True,
        max_rows_limit=250_000,
        stratified_df=stratified_df,
    )

    return {
        'selected': result['selected_features'],
        'lgbm_selected': result['lgbm_selected'],
        'shap_selected': result['shap_selected'],
        'lgbm_dropped': result['lgbm_dropped'],
        'shap_dropped': result['shap_dropped'],
    }


def test_stratified_sampling(
    df: DataFrame,
    target_col: str,
    exclude_cols: List[str],
    max_rows: int,
    dataset_name: str,
    logger: logging.Logger,
) -> DataFrame:
    """Протестировать stratified sampling."""
    logger.info(f"  Test stratified_sampling with max_rows={max_rows}")
    logger.info(f"  [test_stratified_sampling] df: {len(df.columns)} cols, {df.count():,} rows")
    from methods import apply_stratified_sampling
    result = apply_stratified_sampling(df, target_col, max_rows)
    return result


# === ПИПЛайн ОТБОРА ===
def run_pipeline(
    df: DataFrame,
    original_df: DataFrame,
    dataset_name: str,
    results_dir: str,
    config: dict,
    logger: logging.Logger,
    target_col: str = None,
    exclude_cols: List[str] = None,
) -> DataFrame:
    """Запустить пайплайн отбора признаков для одного датасета."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ПИПЛИН ОТБОРА: {dataset_name}")
    logger.info(f"{'='*80}")
    logger.info(f"  [run_pipeline] START - df: {len(df.columns)} cols, {df.count():,} rows")
    logger.info(f"  [run_pipeline] target_col: {target_col}")
    logger.info(f"  [run_pipeline] exclude_cols ({len(exclude_cols)}): {exclude_cols[:10]}...")

    # Определить колонки (использовать переданные или найти автоматически)
    if target_col is None:
        target_col = find_target_column(df)
        logger.info(f"  Target column: {target_col}")

    if exclude_cols is None:
        exclude_cols = find_exclude_columns(df, target_col)
        logger.info(f"  Exclude columns: {exclude_cols}")

    # Получить min_variance и chunk_size
    variance_param = get_variance_param(config, dataset_name)
    chunk_size = get_chunk_size(config, dataset_name)

    # Получить порядок методов (формат: [(method_name, params), ...])
    order = get_order(config, dataset_name)
    logger.info(f"  Порядок методов: {order}")

    # Парсинг параметров BorutaSHAP
    def get_boruta_params(method_config: dict) -> tuple:
        """Получить параметры BorutaSHAP из config.
        :return: (enabled, model_type, boruta_trials, parameters, optuna_params)
        """
        if not method_config:
            return False, 'lgbm', 50, None, None

        enabled = method_config.get('enabled', False)
        if not enabled:
            return False, 'lgbm', 50, None, None

        model_type = method_config.get('model_type', 'lgbm')
        boruta_trials = method_config.get('boruta_trials', 50)

        # New format: parameters and optuna_params
        parameters = method_config.get('parameters', None)
        optuna_params = method_config.get('optuna_params', None)

        return enabled, model_type, boruta_trials, parameters, optuna_params

    # Логирование глобальных гиперпараметров
    logger.info(f"\nГИПЕРПАРАМЕТРЫ МЕТОДОВ:")
    logger.info(f"  drop_null:")
    enabled, params = get_method_params(config, 'drop_null', dataset_name)
    if enabled and params:
        logger.info(f"    enabled: true, params: {params}")
    else:
        logger.info(f"    enabled: false")

    logger.info(f"  drop_constant:")
    enabled, params = get_method_params(config, 'drop_constant', dataset_name)
    if enabled and params:
        logger.info(f"    enabled: true, params: {params}, chunk_size: {chunk_size}")
    else:
        logger.info(f"    enabled: false")

    variance_config = config.get('global_methods', {}).get('drop_low_variance', {})
    scale_methods = variance_config.get('scale_method', variance_config.get('params', {}).get('scale_methods', ['standard']))
    variance_values = variance_config.get('param', variance_config.get('params', {}).get('values', [variance_param]))
    if isinstance(scale_methods, list):
        scale_methods = scale_methods[0]
    if isinstance(variance_values, list):
        variance_values = variance_values[0]
    logger.info(f"  drop_low_variance:")
    enabled, params = get_method_params(config, 'drop_low_variance', dataset_name)
    if enabled and params:
        logger.info(f"    enabled: true, param: {variance_values}, scale_method: {scale_methods}")
    else:
        logger.info(f"    enabled: false")

    logger.info(f"  drop_correlated:")
    enabled, params = get_method_params(config, 'drop_correlated', dataset_name)
    if enabled and params:
        logger.info(f"    enabled: true, params: {params}")
    else:
        logger.info(f"    enabled: false")

    logger.info(f"  lgbm_shap:")
    enabled, params = get_method_params(config, 'lgbm_shap', dataset_name)
    if enabled and params:
        logger.info(f"    enabled: true, params: {params}")
    else:
        logger.info(f"    enabled: false")

    logger.info(f"  boruta_shap:")
    boruta_config = config.get('datasets', {}).get(dataset_name, {}).get('methods', {}).get('boruta_shap', {})
    if not boruta_config:
        boruta_config = config.get('methods', {}).get('boruta_shap', {})
    if not boruta_config:
        boruta_config = config.get('global_methods', {}).get('boruta_shap', {})

    if boruta_config:
        enabled = boruta_config.get('enabled', False)
        if enabled:
            model_type = boruta_config.get('model_type', 'lgbm')
            boruta_trials = boruta_config.get('boruta_trials', 50)
            parameters = boruta_config.get('parameters', None)
            optuna_params = boruta_config.get('optuna_params', None)

            logger.info(f"    enabled: true")
            logger.info(f"    model_type: {model_type}")
            logger.info(f"    boruta_trials: {boruta_trials}")

            if optuna_params:
                logger.info(f"    optuna_params:")
                logger.info(f"      n_trials: {optuna_params.get('n_trials', 'default')}")
                logger.info(f"      n_startup_trials: {optuna_params.get('n_startup_trials', 'default')}")
                logger.info(f"      sampler: {optuna_params.get('sampler', 'default')}")

            if parameters:
                logger.info(f"    parameters: {len(parameters)} params")
        else:
            logger.info(f"    enabled: false")
    else:
        logger.info(f"    not configured")

    # Промежуточные результаты
    current_df = df
    step_num = 0

    # Перебираем order как список кортежей (method_name, params)
    for item in order:
        if isinstance(item, str):
            # Старый формат: просто строка
            method_name = item
            params = None
        else:
            # Новый формат: кортеж (method_name, params)
            method_name, params = item
        
        step_num += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"Шаг {step_num}: {method_name}")
        logger.info(f"{'='*80}")

        if method_name == 'drop_null':
            # Используем params из order если есть, иначе из config
            if params is None:
                enabled, params = get_method_params(config, 'drop_null', dataset_name)
            if enabled and params:
                param = params[0]
                logger.info(f"  Гиперпараметры: max_null_fraction={param}")
                result_df = test_drop_null(current_df, target_col, exclude_cols, param, dataset_name, logger)
                dropped = get_dropped_columns(original_df, result_df)
                filepath = os.path.join(results_dir, dataset_name, f"null_{param}_cols.txt")
                save_columns_to_file(dropped, filepath)
                logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
                current_df = result_df
                logger.info(f"  Колонок на выходе: {len(current_df.columns)}")
            else:
                logger.info(f"  Пропущено (disabled)")

        elif method_name == 'stratified_sampling':
            # Используем params из order если есть, иначе из config
            if params is None:
                enabled, params = get_method_params(config, 'stratified_sampling', dataset_name)
            if enabled and params:
                max_rows = params[0]
                logger.info(f"  Гиперпараметры: max_rows={max_rows}")
                result_df = test_stratified_sampling(current_df, target_col, exclude_cols, max_rows, dataset_name, logger)
                dropped = get_dropped_columns(original_df, result_df)
                filepath = os.path.join(results_dir, dataset_name, f"stratified_{max_rows}_cols.txt")
                save_columns_to_file(dropped, filepath)
                logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
                current_df = result_df
                logger.info(f"  Колонок на выходе: {len(current_df.columns)}")
            else:
                logger.info(f"  Пропущено (disabled)")

        elif method_name == 'drop_constant':
            # Используем params из order если есть, иначе из config
            if params is None:
                enabled, params = get_method_params(config, 'drop_constant', dataset_name)
            if enabled and params:
                param = params[0]
                logger.info(f"  Гиперпараметры: tol={param}, chunk_size={chunk_size}")
                result_df = test_drop_constant(current_df, target_col, exclude_cols, param, chunk_size, dataset_name, logger)
                dropped = get_dropped_columns(original_df, result_df)
                filepath = os.path.join(results_dir, dataset_name, f"constant_{param}_cols.txt")
                save_columns_to_file(dropped, filepath)
                logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
                current_df = result_df
                logger.info(f"  Колонок на выходе: {len(current_df.columns)}")
            else:
                logger.info(f"  Пропущено (disabled)")

        elif method_name == 'drop_low_variance':
            # Получаем scale_method и param
            variance_config = config.get('global_methods', {}).get('drop_low_variance', {})
            scale_methods = variance_config.get('scale_method', variance_config.get('params', {}).get('scale_methods', ['standard']))
            variance_values = variance_config.get('param', variance_config.get('params', {}).get('values', [variance_param]))
            # Получаем chunk_size (для датасета или глобальный)
            chunk_size = get_chunk_size(config, dataset_name)
            # Приводим к спискам если не список
            if not isinstance(scale_methods, list):
                scale_methods = [scale_methods]
            if not isinstance(variance_values, list):
                variance_values = [variance_values]

            logger.info(f"  Гиперпараметры: min_variance={variance_values}, scale_method={scale_methods}, chunk_size={chunk_size}")
            for scale_method in scale_methods:
                for var_param in variance_values:
                    var_param = float(var_param)  # Convert to float (YAML may parse as string)
                    result_df = test_drop_variance(current_df, target_col, exclude_cols, var_param, scale_method, dataset_name, logger, chunk_size)
                    dropped = get_dropped_columns(original_df, result_df)
                    filepath = os.path.join(results_dir, dataset_name, f"variance_{var_param}_{scale_method}_cols.txt")
                    save_columns_to_file(dropped, filepath)
                    logger.info(f"  Выкинуто: {len(dropped)} колонок (scale={scale_method}, param={var_param}) -> {filepath}")
                    current_df = result_df
                    logger.info(f"  Колонок на выходе: {len(current_df.columns)}")

        elif method_name == 'drop_correlated':
            # Используем params из order если есть, иначе из config
            if params is None:
                enabled, params = get_method_params(config, 'drop_correlated', dataset_name)
            if enabled and params:
                param = params[0]
                logger.info(f"  Гиперпараметры: corr_threshold={param}")
                result_df = test_drop_correlated(current_df, target_col, exclude_cols, param, dataset_name, logger)
                dropped = get_dropped_columns(original_df, result_df)
                filepath = os.path.join(results_dir, dataset_name, f"correlated_{param}_cols.txt")
                save_columns_to_file(dropped, filepath)
                logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
                current_df = result_df
                logger.info(f"  Колонок на выходе: {len(current_df.columns)}")
            else:
                logger.info(f"  Пропущено (disabled)")

        elif method_name == 'lgbm_shap':
            # Используем params из order если есть, иначе из config
            if params is None:
                enabled, params = get_method_params(config, 'lgbm_shap', dataset_name)
            if enabled and params:
                param = params[0]
                logger.info(f"  Гиперпараметры: threshold={param}")
                # Используем текущий DataFrame (может быть уже отсемплированным через stratified_sampling)
                results = test_drop_lgbm_shap(current_df, target_col, exclude_cols, param, dataset_name, logger, current_df)

                save_columns_to_file(results['selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_selected_{param}.txt"))
                save_columns_to_file(results['lgbm_selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_selected_{param}.txt"))
                save_columns_to_file(results['shap_selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_selected_{param}.txt"))
                save_columns_to_file(results['lgbm_dropped'], os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_dropped_{param}.txt"))
                save_columns_to_file(results['shap_dropped'], os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_dropped_{param}.txt"))

                logger.info(f"  Selected: {len(results['selected'])}")
            else:
                logger.info(f"  Пропущено (disabled)")

        elif method_name == 'boruta_shap':
            # Используем params из order если есть, иначе из config
            if params is None:
                enabled, method_config_params = get_method_params(config, 'boruta_shap', dataset_name)
            else:
                enabled = True
                method_config_params = None

            # Получить параметры BorutaSHAP (новый формат с parameters/optuna_params)
            if enabled:
                # Пытаемся получить из метода BorutaSHAP
                boruta_config = config.get('datasets', {}).get(dataset_name, {}).get('methods', {}).get('boruta_shap', {})
                if not boruta_config:
                    boruta_config = config.get('methods', {}).get('boruta_shap', {})
                if not boruta_config:
                    boruta_config = config.get('global_methods', {}).get('boruta_shap', {})

                if boruta_config:
                    enabled, model_type, boruta_trials, parameters, optuna_params = get_boruta_params(boruta_config)
                    logger.info(f"  Гиперпараметры: model_type={model_type}, boruta_trials={boruta_trials}")
                    results = test_drop_boruta_shap(
                        current_df, target_col, exclude_cols, model_type, boruta_trials,
                        dataset_name, logger, parameters, optuna_params
                    )

                    save_columns_to_file(results['selected'], os.path.join(results_dir, dataset_name, f"boruta_shap_selected.txt"))
                    save_columns_to_file(results['dropped'], os.path.join(results_dir, dataset_name, f"boruta_shap_dropped.txt"))
                    save_columns_to_file(results['selected'], os.path.join(results_dir, dataset_name, f"stage_0_1_selected.txt"))

                    logger.info(f"  Selected: {len(results['selected'])}")
                else:
                    logger.info(f"  Пропущено (disabled)")
            else:
                logger.info(f"  Пропущено (disabled)")

    logger.info("\n" + "="*80)
    logger.info("ВСЕ ШАГИ ПИПЛАЙНА ЗАВЕРШЕНЫ!")
    logger.info("="*80)
    return current_df


def get_method_params(config: dict, method_name: str, dataset_name: str = None) -> tuple:
    """
    Получить параметры метода из конфига.
    :return: (enabled, params_list)
    """
    datasets_config = config.get('datasets', {})
    global_methods_config = config.get('global_methods', {})
    methods_config = config.get('methods', {})

    if dataset_name:
        dataset_config = datasets_config.get(dataset_name, {})
        dataset_methods = dataset_config.get('methods', {})
        method_config = dataset_methods.get(method_name, {})

        if method_config:
            enabled = method_config.get('enabled', False)
            if not enabled:
                return False, []
            # Поддержка новой структуры (param) и старой (params)
            if 'param' in method_config:
                return True, [method_config['param']]
            params = method_config.get('params', [])
            return True, params

    method_config = methods_config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        if not enabled:
            return False, []
        # Поддержка новой структуры (param) и старой (params)
        if 'param' in method_config:
            return True, [method_config['param']]
        params = method_config.get('params', [])
        return True, params

    method_config = global_methods_config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        if not enabled:
            return False, []
        # Поддержка новой структуры (param) и старой (params)
        if 'param' in method_config:
            return True, [method_config['param']]
        params = method_config.get('params', [])
        return True, params

    return False, []


# === MAIN ===
def main():
    """Основная функция."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "="*80)
    print("ПИПЛИН ОТБОРА ПРИЗНАКОВ")
    print("="*80)

    os.makedirs(PIPELINE_DIR, exist_ok=True)
    os.makedirs(os.path.join(PIPELINE_DIR, "logs"), exist_ok=True)

    logger = setup_logger("pipeline", os.path.join(PIPELINE_DIR, "logs"), timestamp)
    logger.info("Запуск пайплайна отбора признаков")

    # Загрузить пути
    logger.info(f"\nЗагрузка путей из {PATHS_FILE}")
    config = load_paths(PATHS_FILE)
    datasets = config.get('datasets', {})
    logger.info(f"Найдено {len(datasets)} датасетов: {list(datasets.keys())}")

    # Запустить Spark
    print("\n" + "="*80)
    print("ЗАПУСК SPARK СЕССИИ")
    print("="*80)

    start_spark = datetime.now()
    spark = create_spark_session(app_name='feature_selection_pipeline', executor_instances=8)
    spark_time = (datetime.now() - start_spark).total_seconds()
    logger.info(f"Spark сессия запущена! Время: {spark_time:.2f} сек")

    # Для каждого датасета
    for dataset_name, dataset_config in datasets.items():
        if not dataset_config.get('enabled', True):
            logger.info(f"\n{dataset_name}: пропущено (disabled in config)")
            continue

        print("\n" + "="*80)
        print(f"Обработка датасета: {dataset_name}")
        print("="*80)

        base_path = dataset_config.get('path', '')
        has_split_type = dataset_config.get('has_split_type', False)

        if has_split_type:
            train_path = os.path.join(base_path, "split_type=train")
        else:
            train_path = base_path

        logger.info(f"\nЗагрузка данных: {train_path}")

        try:
            original_df = spark.read.parquet(train_path)
            total_rows = original_df.count()
            total_cols = len(original_df.columns)

            logger.info(f"  Загружено {total_rows:,} строк и {total_cols} колонок")

            # Найти exclude колонки
            target_col = find_target_column(original_df)
            exclude_cols = find_exclude_columns(original_df, target_col)
            logger.info(f"  Target column: {target_col}")
            logger.info(f"  Exclude columns ({len(exclude_cols)}): {exclude_cols}")

            # Загрузить файл с колонками для дропа если есть
            drop_cols_file = dataset_config.get('drop_cols_file', '')
            if drop_cols_file:
                # Раскрыть ~ в путь ДО проверки существования
                expanded_path = os.path.expanduser(drop_cols_file)
                if os.path.exists(expanded_path):
                    logger.info(f"  Загрузка колонок для дропа из {drop_cols_file}")
                    dropped_cols = load_dropped_columns(drop_cols_file)
                    logger.info(f"  Найдено {len(dropped_cols)} колонок для дропа")
                    logger.info(f"  dropped_cols (first 20): {dropped_cols[:20]}")

                    # Исключить exclude_cols из списка для дропа (не удалять target и другие ключевые колонки)
                    exclude_set = set(exclude_cols)
                    dropped_cols = [c for c in dropped_cols if c not in exclude_set]
                    logger.info(f"  После фильтрации exclude_cols: {len(dropped_cols)} колонок для дропа")
                    logger.info(f"  dropped_cols after filter (first 20): {dropped_cols[:20]}")

                    if dropped_cols:
                        cols_to_keep = [c for c in original_df.columns if c not in set(dropped_cols)]
                        current_df = original_df.select(*cols_to_keep)
                        logger.info(f"  После дропа: {len(current_df.columns)} колонок")
                        logger.info(f"  current_df columns (first 20): {current_df.columns[:20]}")
                    else:
                        current_df = original_df
                else:
                    logger.info(f"  Файл для дропа не найден: {drop_cols_file}")
            else:
                current_df = original_df

            # Загрузить файл с ОСТАВЛЕННЫМИ колонками если есть (stage_0_1_selected.txt)
            selected_cols_file = dataset_config.get('selected_cols_file', '')
            if selected_cols_file:
                expanded_path = os.path.expanduser(selected_cols_file)
                if os.path.exists(expanded_path):
                    logger.info(f"  Загрузка ОСТАВЛЕННЫХ колонок из {selected_cols_file}")
                    selected_cols = load_selected_columns(selected_cols_file)
                    logger.info(f"  Найдено {len(selected_cols)} ОСТАВЛЕННЫХ колонок")
                    logger.info(f"  selected_cols (first 20): {selected_cols[:20]}")

                    # Проверить какие колонки из файла присутствуют в DataFrame
                    df_cols_set = set(current_df.columns)
                    available_cols = [c for c in selected_cols if c in df_cols_set]
                    not_found = [c for c in selected_cols if c not in df_cols_set]

                    if not_found:
                        logger.info(f"  Не найдено в DataFrame: {len(not_found)} колонок (первые 10): {not_found[:10]}")

                    if available_cols:
                        # Оставить только колонки из файла + exclude_cols (которые не дропаются)
                        cols_to_keep = set(available_cols) | set(exclude_cols)
                        current_df = current_df.select(*list(cols_to_keep))
                        logger.info(f"  После фильтрации по selected_cols: {len(current_df.columns)} колонок")
                        logger.info(f"  current_df columns (first 20): {current_df.columns[:20]}")
                    else:
                        logger.info(f"  Нет доступных колонок из файла, используем оригинальный df")
                else:
                    logger.info(f"  Файл с оставленными колонками не найден: {selected_cols_file}")

            # Запустить пайплайн
            final_df = run_pipeline(current_df, original_df, dataset_name, PIPELINE_DIR, config, logger, target_col, exclude_cols)

            logger.info(f"\nИтоговое количество колонок: {len(final_df.columns)}")

        except Exception as e:
            logger.error(f"  ОШИБКА: {str(e)}")
            logger.exception("Детали ошибки:")
            continue

    logger.info("\n" + "="*80)
    logger.info("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    logger.info("="*80)
    print("\n" + "="*80)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    print("="*80)

    spark.stop()


if __name__ == '__main__':
    main()
