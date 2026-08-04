#!/usr/bin/env python3
"""
Sequential feature selection test script.

Протестирует методы отбора признаков в последовательном порядке:
null -> constant -> variance -> correlated -> lgbm_shap

Каждый следующий метод применяется к результату предыдущего.
Если correlated упадет - скрипт продолжит работу.
Результаты сохраняются в results/sequential/
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
)

# === КОНФИГУРАЦИЯ ===
PATHS_FILE = "paths_sequential.yaml"
RESULTS_DIR = "results"
SEQUENTIAL_DIR = os.path.join(RESULTS_DIR, "sequential")


# === ЛОГИРОВАНИЕ ===
def setup_logger(name: str, log_dir: str, timestamp: str) -> logging.Logger:
    """Настройка логгера."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sequential_{timestamp}.log")

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


def get_method_param(config: dict, method_name: str) -> tuple:
    """
    Получить параметр метода из конфига.
    Проверяет: methods -> global_methods -> корень конфига (для paths_sequential.yaml)
    :return: (enabled, param_value)
    """
    methods_config = config.get('methods', {})
    global_methods_config = config.get('global_methods', {})

    # Сначала глобальные методы
    method_config = global_methods_config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        if not enabled:
            return False, None
        param = method_config.get('param', None)
        if param is not None:
            return True, param

    # Проверить methods
    method_config = methods_config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        if not enabled:
            return False, None
        param = method_config.get('param', None)
        if param is not None:
            return True, param

    # Проверить корень конфига (для paths_sequential.yaml где параметры на верхнем уровне)
    method_config = config.get(method_name, {})
    if method_config:
        enabled = method_config.get('enabled', False)
        if not enabled:
            return False, None
        param = method_config.get('param', None)
        if param is not None:
            return True, param

    return False, None


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
    result = drop_null_features(df, max_null_fraction=max_null_fraction, exclude_cols=exclude_cols)
    return result


def test_drop_constant(df: DataFrame, target_col: str, exclude_cols: List[str],
                       tol: float, dataset_name: str, logger: logging.Logger) -> DataFrame:
    """Протестировать drop_constant_features."""
    logger.info(f"  Test drop_constant with tol={tol}")
    result = drop_constant_features(df, tol=tol, exclude_cols=exclude_cols)
    return result


def test_drop_variance(df: DataFrame, target_col: str, exclude_cols: List[str],
                       min_variance: float, scale_method: str,
                       dataset_name: str, logger: logging.Logger) -> DataFrame:
    """Протестировать drop_low_variance_features."""
    logger.info(f"  Test drop_low_variance with min_variance={min_variance}, scale={scale_method}")
    result = drop_low_variance_features(
        df,
        min_variance=min_variance,
        scale_method=scale_method,
        exclude_cols=exclude_cols
    )
    return result


def test_drop_correlated(df: DataFrame, target_col: str, exclude_cols: List[str],
                         corr_threshold: float, dataset_name: str, logger: logging.Logger) -> DataFrame:
    """Протестировать drop_correlated_features."""
    logger.info(f"  Test drop_correlated with corr_threshold={corr_threshold}")
    result = drop_correlated_features(df, corr_threshold=corr_threshold, exclude_cols=exclude_cols)
    return result


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

    return {
        'selected': result['selected_features'],
        'lgbm_selected': result['lgbm_selected'],
        'shap_selected': result['shap_selected'],
        'lgbm_dropped': result['lgbm_dropped'],
        'shap_dropped': result['shap_dropped'],
    }


# === ПОСЛЕДОВАТЕЛЬНОЕ ТЕСТИРОВАНИЕ ===
def run_sequential_tests(
    df: DataFrame,
    original_df: DataFrame,
    dataset_name: str,
    results_dir: str,
    config: dict,
    logger: logging.Logger
) -> None:
    """Запустить последовательные тесты для одного датасета."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ПОСЛЕДОВАТЕЛЬНОЕ ТЕСТИРОВАНИЕ: {dataset_name}")
    logger.info(f"{'='*80}")

    # Определить колонки
    target_col = find_target_column(df)
    logger.info(f"  Target column: {target_col}")

    exclude_cols = find_exclude_columns(df, target_col)
    logger.info(f"  Exclude columns: {exclude_cols}")

    # Порядок методов
    order = config.get('order', ['drop_null', 'drop_constant', 'drop_low_variance', 'drop_correlated', 'lgbm_shap'])
    logger.info(f"  Порядок методов: {order}")

    # Промежуточные результаты
    current_df = df
    step_num = 0

    for method_name in order:
        step_num += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"Шаг {step_num}: {method_name}")
        logger.info(f"{'='*80}")

        # Получить параметр метода
        enabled, param = get_method_param(config, method_name)

        if not enabled:
            logger.info(f"  Пропущено (disabled in config)")
            continue

        logger.info(f"  Количество колонок на входе: {len(current_df.columns)}")

        if method_name == 'drop_null':
            result_df = test_drop_null(current_df, target_col, exclude_cols, param, dataset_name, logger)
            dropped = get_dropped_columns(original_df, result_df)
            filepath = os.path.join(results_dir, dataset_name, f"null_{param}_cols.txt")
            save_columns_to_file(dropped, filepath)
            logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
            current_df = result_df

        elif method_name == 'drop_constant':
            result_df = test_drop_constant(current_df, target_col, exclude_cols, param, dataset_name, logger)
            dropped = get_dropped_columns(original_df, result_df)
            filepath = os.path.join(results_dir, dataset_name, f"constant_{param}_cols.txt")
            save_columns_to_file(dropped, filepath)
            logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
            current_df = result_df

        elif method_name == 'drop_low_variance':
            # Берем scale_method из конфига
            method_config_dict = config.get('drop_low_variance', {})
            scale_method = method_config_dict.get('scale_method', 'standard')
            var_param = method_config_dict.get('param', 0.005)

            result_df = test_drop_variance(current_df, target_col, exclude_cols, var_param, scale_method, dataset_name, logger)
            dropped = get_dropped_columns(original_df, result_df)
            filepath = os.path.join(results_dir, dataset_name, f"variance_{var_param}_{scale_method}_cols.txt")
            save_columns_to_file(dropped, filepath)
            logger.info(f"  Выкинуто: {len(dropped)} колонок (scale={scale_method}, param={var_param}) -> {filepath}")
            current_df = result_df

        elif method_name == 'drop_correlated':
            try:
                result_df = test_drop_correlated(current_df, target_col, exclude_cols, param, dataset_name, logger)
                dropped = get_dropped_columns(original_df, result_df)
                filepath = os.path.join(results_dir, dataset_name, f"correlated_{param}_cols.txt")
                save_columns_to_file(dropped, filepath)
                logger.info(f"  Выкинуто: {len(dropped)} колонок -> {filepath}")
                current_df = result_df
            except Exception as e:
                logger.error(f"  ОШИБКА в correlated (threshold={param}): {str(e)}")
                logger.error(f"  Продолжаем работу без correlated...")
                filepath = os.path.join(results_dir, dataset_name, f"correlated_{param}_error.txt")
                with open(filepath, 'w') as f:
                    f.write(f"ERROR: {str(e)}\n")
                    f.write("Корреляции не были вычислены из-за ошибки\n")

        elif method_name == 'lgbm_shap':
            try:
                results = test_drop_lgbm_shap(current_df, target_col, exclude_cols, param, dataset_name, logger)

                save_columns_to_file(results['selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_selected_{param}.txt"))
                save_columns_to_file(results['lgbm_selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_selected_{param}.txt"))
                save_columns_to_file(results['shap_selected'], os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_selected_{param}.txt"))
                save_columns_to_file(results['lgbm_dropped'], os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_dropped_{param}.txt"))
                save_columns_to_file(results['shap_dropped'], os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_dropped_{param}.txt"))

                logger.info(f"  Selected: {len(results['selected'])}")
                logger.info(f"  LGBM selected: {len(results['lgbm_selected'])}, dropped: {len(results['lgbm_dropped'])}")
                logger.info(f"  SHAP selected: {len(results['shap_selected'])}, dropped: {len(results['shap_dropped'])}")
            except Exception as e:
                logger.error(f"  ОШИБКА в lgbm_shap (threshold={param}): {str(e)}")

        logger.info(f"  Количество колонок на выходе: {len(current_df.columns)}")

    logger.info("\n" + "="*80)
    logger.info("ВСЕ ПОСЛЕДОВАТЕЛЬНЫЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    logger.info("="*80)


# === MAIN ===
def main():
    """Основная функция."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "="*80)
    print("ПОСЛЕДОВАТЕЛЬНОЕ ТЕСТИРОВАНИЕ МЕТОДОВ ОТБОРА ПРИЗНАКОВ")
    print("="*80)

    # Создать папку результатов
    os.makedirs(SEQUENTIAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(SEQUENTIAL_DIR, "logs"), exist_ok=True)

    # Настройка логгера
    logger = setup_logger("sequential_test", os.path.join(SEQUENTIAL_DIR, "logs"), timestamp)
    logger.info("Запуск последовательного тестирования методов")

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
    spark = create_spark_session(app_name='sequential_feature_selection', executor_instances=8)
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

        try:
            # Загрузить train партицию
            original_df = spark.read.parquet(train_path)
            total_rows = original_df.count()
            total_cols = len(original_df.columns)

            logger.info(f"  Загружено {total_rows:,} строк и {total_cols} колонок")

            # Запустить последовательные тесты
            run_sequential_tests(original_df, original_df, dataset_name, SEQUENTIAL_DIR, config, logger)

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


if __name__ == '__main__':
    main()
