#!/usr/bin/env python3
"""
Test script for feature selection methods.

Протестирует 3 метода отбора признаков с разными гиперпараметрами на 4 датасетах.
Для каждого метода и параметра сохранит список выкинутых колонок.

Гиперпараметры:
- drop_null: max_null_fraction = 0.9, 0.95, 0.99
- drop_constant: tol = 0.95, 0.99, 1.0
- drop_low_variance: min_variance = 0.001, 0.005, 0.01
"""

import sys
import os
import logging
from datetime import datetime
import yaml
from typing import Dict, List, Optional

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
)

# === КОНФИГУРАЦИЯ ===
PATHS_FILE = "paths.yaml"
RESULTS_DIR = "results"

# Стратегии для low_variance
VARIANCE_SCALE_METHODS = ["standard"]

# Параметры сэмплирования
SAMPLE_FRACTION = 0.1  # 10% данных для теста
RANDOM_STATE = 42


# === ЛОГИРОВАНИЕ ===
def setup_logger(name: str, log_dir: str) -> logging.Logger:
    """Настройка логгера."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Handler для файла
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Handler для stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Формат
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


def get_method_params(config: dict, method_name: str, dataset_name: str = None) -> tuple:
    """
    Получить параметры метода из конфига.
    Сначала проверяет dataset-specific, затем глобальные.
    :return: (enabled, params_list)
    """
    datasets_config = config.get('datasets', {})
    global_methods_config = config.get('global_methods', {})
    methods_config = config.get('methods', {})

    # Проверить dataset-specific параметры
    if dataset_name:
        dataset_config = datasets_config.get(dataset_name, {})
        dataset_methods = dataset_config.get('methods', {})
        method_config = dataset_methods.get(method_name, {})

        if method_config:
            enabled = method_config.get('enabled', False)
            _get_logger().info(f"  get_method_params({method_name}): found in dataset.methods, enabled={enabled}")
            if not enabled:
                return False, []
            params = method_config.get('params', [])
            return True, params

    # Проверить глобальные параметры метода
    method_config = methods_config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        _get_logger().info(f"  get_method_params({method_name}): found in methods, enabled={enabled}")
        if not enabled:
            return False, []
        params = method_config.get('params', [])
        return True, params

    # Проверить глобальные (для совместимости)
    method_config = global_methods_config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        _get_logger().info(f"  get_method_params({method_name}): found in global_methods, enabled={enabled}")
        if not enabled:
            return False, []
        params = method_config.get('params', [])
        return True, params

    _get_logger().info(f"  get_method_params({method_name}): not found in config")
    return False, []


def find_target_column(df: DataFrame) -> str:
    """Найти целевую колонку в DataFrame."""
    for col in df.columns:
        if 'target' in col.lower() or col.lower() == 'y':
            return col
    raise ValueError(f"Не найдена целевая колонка в DataFrame. Доступные колонки: {df.columns}")


def find_exclude_columns(df: DataFrame, target_col: str) -> List[str]:
    """Найти колонки для исключения (id, date, split и т.д.)."""
    exclude = {target_col}
    exclude_patterns = ['id', 'date', 'time', 'split', 'epoch', 'seq', 'report', 'row']
    
    for col in df.columns:
        col_lower = col.lower()
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


# === ТЕСТИРОВАНИЕ ===
def test_drop_null(df: DataFrame, target_col: str, exclude_cols: List[str], 
                   max_null_fraction: float, dataset_name: str, logger: logging.Logger) -> List[str]:
    """Протестировать drop_null_features."""
    logger.info(f"  Test drop_null with max_null_fraction={max_null_fraction}")
    
    result = drop_null_features(df, max_null_fraction=max_null_fraction, exclude_cols=exclude_cols)
    dropped = set(df.columns) - set(result.columns)
    
    return list(dropped)


def test_drop_constant(df: DataFrame, target_col: str, exclude_cols: List[str],
                       tol: float, dataset_name: str, logger: logging.Logger) -> List[str]:
    """Протестировать drop_constant_features."""
    logger.info(f"  Test drop_constant with tol={tol}")
    
    result = drop_constant_features(df, tol=tol, exclude_cols=exclude_cols)
    dropped = set(df.columns) - set(result.columns)
    
    return list(dropped)


def test_drop_variance(df: DataFrame, target_col: str, exclude_cols: List[str],
                       min_variance: float, scale_method: str,
                       dataset_name: str, logger: logging.Logger) -> List[str]:
    """Протестировать drop_low_variance_features."""
    logger.info(f"  Test drop_low_variance with min_variance={min_variance}, scale={scale_method}")

    result = drop_low_variance_features(
        df,
        min_variance=min_variance,
        scale_method=scale_method,
        exclude_cols=exclude_cols
    )
    dropped = set(df.columns) - set(result.columns)

    return list(dropped)


def test_drop_correlated(df: DataFrame, target_col: str, exclude_cols: List[str],
                         corr_threshold: float, dataset_name: str, logger: logging.Logger) -> List[str]:
    """Протестировать drop_correlated_features."""
    logger.info(f"  Test drop_correlated with corr_threshold={corr_threshold}")

    result = drop_correlated_features(df, corr_threshold=corr_threshold, exclude_cols=exclude_cols)
    dropped = set(df.columns) - set(result.columns)

    return list(dropped)


def test_drop_lgbm_shap(
    df: DataFrame,
    target_col: str,
    exclude_cols: List[str],
    threshold: float,
    dataset_name: str,
    logger: logging.Logger,
) -> dict:
    """Протестировать LGBM + SHAP feature selection."""
    logger.info(f"  Test lgbm_shap with threshold={threshold}")

    result = select_robust_features(
        df,
        target_col=target_col,
        exclude_cols=exclude_cols,
        lgbm_threshold=threshold,
        shap_threshold=threshold,
        return_importances=True,
    )

    # Сохранить результаты
    results = {
        'selected': result['selected_features'],
        'lgbm_selected': result['lgbm_selected'],
        'shap_selected': result['shap_selected'],
        'lgbm_dropped': result['lgbm_dropped'],
        'shap_dropped': result['shap_dropped'],
    }

    return results


def run_dataset_tests_with_config(
    df: DataFrame,
    dataset_name: str,
    results_dir: str,
    config: dict,
    logger: logging.Logger
) -> None:
    """Запустить тесты для одного датасета с конфигом."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Тестирование датасета: {dataset_name}")
    logger.info(f"{'='*80}")

    # Определить колонки
    target_col = find_target_column(df)
    logger.info(f"  Target column: {target_col}")

    exclude_cols = find_exclude_columns(df, target_col)
    logger.info(f"  Exclude columns: {exclude_cols}")

    logger.info(f"  Total columns: {len(df.columns)}")

    # Тест drop_null
    enabled, params = get_method_params(config, 'drop_null', dataset_name)
    if enabled:
        logger.info("\n--- drop_null_features ---")
        for param in params:
            dropped = test_drop_null(df, target_col, exclude_cols, param, dataset_name, logger)
            filepath = os.path.join(results_dir, dataset_name, f"null_{param}_cols.txt")
            save_columns_to_file(dropped, filepath)
            logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    else:
        logger.info("\n--- drop_null_features: пропущено (disabled) ---")

    # Тест drop_constant (на оригинальном df!)
    enabled, params = get_method_params(config, 'drop_constant', dataset_name)
    if enabled:
        logger.info("\n--- drop_constant_features ---")
        for param in params:
            dropped = test_drop_constant(df, target_col, exclude_cols, param, dataset_name, logger)
            filepath = os.path.join(results_dir, dataset_name, f"constant_{param}_cols.txt")
            save_columns_to_file(dropped, filepath)
            logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    else:
        logger.info("\n--- drop_constant_features: пропущено (disabled) ---")

    # Тест drop_low_variance (на оригинальном df!)
    enabled, method_config = get_method_params(config, 'drop_low_variance', dataset_name)
    if enabled:
        logger.info("\n--- drop_low_variance_features ---")
        scale_methods = method_config.get('scale_methods', ['standard'])
        variance_params = method_config.get('values', [0.001, 0.005, 0.01])
        for param in variance_params:
            for scale in scale_methods:
                dropped = test_drop_variance(df, target_col, exclude_cols, param, scale, dataset_name, logger)
                filepath = os.path.join(results_dir, dataset_name, f"variance_{param}_{scale}_cols.txt")
                save_columns_to_file(dropped, filepath)
                logger.info(f"  Dropped {len(dropped)} columns (scale={scale}) -> {filepath}")
    else:
        logger.info("\n--- drop_low_variance_features: пропущено (disabled) ---")

    # Тест drop_correlated (на оригинальном df!)
    enabled, params = get_method_params(config, 'drop_correlated', dataset_name)
    if enabled:
        logger.info("\n--- drop_correlated_features ---")
        for param in params:
            dropped = test_drop_correlated(df, target_col, exclude_cols, param, dataset_name, logger)
            filepath = os.path.join(results_dir, dataset_name, f"correlated_{param}_cols.txt")
            save_columns_to_file(dropped, filepath)
            logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    else:
        logger.info("\n--- drop_correlated_features: пропущено (disabled) ---")

    # Тест LGBM + SHAP (на оригинальном df!)
    enabled, params = get_method_params(config, 'lgbm_shap', dataset_name)
    if enabled:
        logger.info("\n--- LGBM + SHAP ---")
        for param in params:
            results = test_drop_lgbm_shap(df, target_col, exclude_cols, param, dataset_name, logger)
            
            # Сохранить все списки
            save_columns_to_file(results['selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_selected_{param}.txt"))
            save_columns_to_file(results['lgbm_selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_selected_{param}.txt"))
            save_columns_to_file(results['shap_selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_selected_{param}.txt"))
            save_columns_to_file(results['lgbm_dropped'], os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_dropped_{param}.txt"))
            save_columns_to_file(results['shap_dropped'], os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_dropped_{param}.txt"))
            
            logger.info(f"  Selected: {len(results['selected'])}, LGBM selected: {len(results['lgbm_selected'])}, SHAP selected: {len(results['shap_selected'])}")
            logger.info(f"  LGBM dropped: {len(results['lgbm_dropped'])}, SHAP dropped: {len(results['shap_dropped'])}")
    else:
        logger.info("\n--- LGBM + SHAP: пропущено (disabled) ---")

    logger.info("  DONE!")


# === MAIN ===
def main():
    """Основная функция."""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ МЕТОДОВ ОТБОРА ПРИЗНАКОВ")
    print("="*80)
    
    # Создать папку результатов
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
    
    # Настройка логгера
    logger = setup_logger("methods_test", os.path.join(RESULTS_DIR, "logs"))
    logger.info("Запуск тестирования методов")
    
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
    spark = create_spark_session(app_name='feature_selection_test', executor_instances=8)
    spark_time = (datetime.now() - start_spark).total_seconds()
    logger.info(f"Spark сессия запущена! Время: {spark_time:.2f} сек")

    # Для каждого датасета
    for dataset_name, dataset_config in datasets.items():
        # Проверить enabled для датасета
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

        # Загрузить конфигурацию методов
        method_config = load_paths(PATHS_FILE)

        try:
            # Загрузить train партицию
            df = spark.read.parquet(train_path)
            total_rows = df.count()
            total_cols = len(df.columns)

            logger.info(f"  Загружено {total_rows:,} строк и {total_cols} колонок")

            # Запустить тесты с конфигом
            run_dataset_tests_with_config(df, dataset_name, RESULTS_DIR, method_config, logger)

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

    # Закрыть Spark
    spark.stop()


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
# Глобальный logger для функций без доступа к logger
_logger = None


def _get_logger() -> logging.Logger:
    """Получить глобальный logger или создать новый."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger('methods_test')
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            ch.setFormatter(formatter)
            _logger.addHandler(ch)
    return _logger


if __name__ == '__main__':
    main()
