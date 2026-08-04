#!/usr/bin/env python3
"""
Простой пайплайн отбора признаков.
Конфигурация в YAML, шаги в порядке следования.

Пример YAML:
    datasets:
      SA_SMA:
        path: /path/to/data
        has_split_type: false
        
    pipeline:
      steps:
        - method: drop_constant
          params:
            tol: 0.98
        - method: drop_low_variance
          params:
            min_variance: 1e-5
            scale_method: standard
        - method: boruta_shap
          params:
            boruta_trials: 50
"""

import sys
import os
import logging
from datetime import datetime
import yaml
from typing import Dict, List, Any

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
    apply_stratified_sampling,
)

# === КОНФИГУРАЦИЯ ===
CONFIG_FILE = "new_config.yaml"
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


# === МЕТОДЫ ПИПЛАЙНА ===
def apply_drop_null(df: DataFrame, params: Dict, target_col: str, exclude_cols: List[str],
                    original_df: DataFrame, dataset_name: str, results_dir: str) -> DataFrame:
    """Применить drop_null_features."""
    max_null_fraction = params.get('max_null_fraction', 0.5)
    logger = logging.getLogger('pipeline')
    logger.info(f"  drop_null: max_null_fraction={max_null_fraction}")
    
    result = drop_null_features(df, max_null_fraction=max_null_fraction, 
                                 exclude_cols=exclude_cols, target_col=target_col)
    
    dropped = get_dropped_columns(original_df, result)
    filepath = os.path.join(results_dir, dataset_name, f"drop_null_{max_null_fraction}_cols.txt")
    save_columns_to_file(dropped, filepath)
    logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    
    return result


def apply_drop_constant(df: DataFrame, params: Dict, target_col: str, exclude_cols: List[str],
                        original_df: DataFrame, dataset_name: str, results_dir: str) -> DataFrame:
    """Применить drop_constant_features."""
    tol = params.get('tol', 0.98)
    chunk_size = params.get('chunk_size', 1000)
    logger = logging.getLogger('pipeline')
    logger.info(f"  drop_constant: tol={tol}, chunk_size={chunk_size}")
    
    result = drop_constant_features(df, tol=tol, exclude_cols=exclude_cols,
                                    chunk_size=chunk_size, target_col=target_col)
    
    dropped = get_dropped_columns(original_df, result)
    filepath = os.path.join(results_dir, dataset_name, f"drop_constant_{tol}_cols.txt")
    save_columns_to_file(dropped, filepath)
    logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    
    return result


def apply_drop_low_variance(df: DataFrame, params: Dict, target_col: str, exclude_cols: List[str],
                            original_df: DataFrame, dataset_name: str, results_dir: str) -> DataFrame:
    """Применить drop_low_variance_features."""
    min_variance = params.get('min_variance', 0.005)
    scale_method = params.get('scale_method', 'standard')
    chunk_size = params.get('chunk_size', 1000)
    logger = logging.getLogger('pipeline')
    logger.info(f"  drop_low_variance: min_variance={min_variance}, scale_method={scale_method}")
    
    result = drop_low_variance_features(df, min_variance=min_variance,
                                        scale_method=scale_method,
                                        exclude_cols=exclude_cols,
                                        chunk_size=chunk_size,
                                        target_col=target_col)
    
    dropped = get_dropped_columns(original_df, result)
    filepath = os.path.join(results_dir, dataset_name, f"drop_low_variance_{min_variance}_{scale_method}_cols.txt")
    save_columns_to_file(dropped, filepath)
    logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    
    return result


def apply_drop_correlated(df: DataFrame, params: Dict, target_col: str, exclude_cols: List[str],
                          original_df: DataFrame, dataset_name: str, results_dir: str) -> DataFrame:
    """Применить drop_correlated_features."""
    corr_threshold = params.get('corr_threshold', 0.95)
    logger = logging.getLogger('pipeline')
    logger.info(f"  drop_correlated: corr_threshold={corr_threshold}")
    
    result = drop_correlated_features(df, corr_threshold=corr_threshold,
                                      exclude_cols=exclude_cols, target_col=target_col)
    
    dropped = get_dropped_columns(original_df, result)
    filepath = os.path.join(results_dir, dataset_name, f"drop_correlated_{corr_threshold}_cols.txt")
    save_columns_to_file(dropped, filepath)
    logger.info(f"  Dropped {len(dropped)} columns -> {filepath}")
    
    return result


def apply_stratified_sampling_method(df: DataFrame, params: Dict, target_col: str, 
                                     exclude_cols: List[str], dataset_name: str, 
                                     results_dir: str) -> DataFrame:
    """Применить stratified sampling."""
    max_rows = params.get('max_rows', 100000)
    logger = logging.getLogger('pipeline')
    logger.info(f"  stratified_sampling: max_rows={max_rows}")
    
    original_rows = df.count()
    result = apply_stratified_sampling(df, target_col, max_rows)
    result_rows = result.count()
    
    logger.info(f"  Rows: {original_rows:,} -> {result_rows:,}")
    
    return result


def apply_lgbm_shap(df: DataFrame, params: Dict, target_col: str, exclude_cols: List[str],
                    original_df: DataFrame, dataset_name: str, results_dir: str) -> DataFrame:
    """Применить LGBM + SHAP feature selection."""
    threshold = params.get('threshold', 0.85)
    max_rows_limit = params.get('max_rows_limit', 250000)
    logger = logging.getLogger('pipeline')
    logger.info(f"  lgbm_shap: threshold={threshold}, max_rows_limit={max_rows_limit}")
    
    result = select_robust_features(
        df, target_col=target_col, exclude_cols=exclude_cols,
        lgbm_threshold=threshold, shap_threshold=threshold,
        return_importances=True, max_rows_limit=max_rows_limit
    )
    
    # Сохранить результаты
    save_columns_to_file(result['selected'], 
                        os.path.join(results_dir, dataset_name, f"lgbm_shap_selected_{threshold}.txt"))
    save_columns_to_file(result['lgbm_selected'], 
                        os.path.join(results_dir, dataset_name, f"lgbm_shap_lgbm_selected_{threshold}.txt"))
    save_columns_to_file(result['shap_selected'], 
                        os.path.join(results_dir, dataset_name, f"lgbm_shap_shap_selected_{threshold}.txt"))
    
    logger.info(f"  Selected: {len(result['selected'])}, LGBM: {len(result['lgbm_selected'])}, SHAP: {len(result['shap_selected'])}")
    
    return df.select(*result['selected_features'])


def apply_boruta_shap(df: DataFrame, params: Dict, target_col: str, exclude_cols: List[str],
                      original_df: DataFrame, dataset_name: str, results_dir: str) -> DataFrame:
    """Применить BorutaSHAP feature selection."""
    model_type = params.get('model_type', 'lgbm')
    boruta_trials = params.get('boruta_trials', 50)
    parameters = params.get('parameters', None)
    optuna_params = params.get('optuna_params', None)
    
    logger = logging.getLogger('pipeline')
    logger.info(f"  boruta_shap: model_type={model_type}, boruta_trials={boruta_trials}")
    
    result = select_features_boruta_shap(
        df, target_col=target_col, exclude_cols=exclude_cols,
        model_type=model_type, boruta_trials=boruta_trials,
        parameters=parameters, optuna_params=optuna_params
    )
    
    # Сохранить результаты
    save_columns_to_file(result, os.path.join(results_dir, dataset_name, "boruta_shap_selected.txt"))
    logger.info(f"  Selected: {len(result)} columns")
    
    return df.select(*result)


# === ГЛАВНАЯ ФУНКЦИЯ ===
def run_pipeline(df: DataFrame, config: Dict, dataset_name: str, 
                 results_dir: str, logger: logging.Logger) -> DataFrame:
    """
    Запустить пайплайн для одного датасета.
    
    Просто проходим по шагам и применяем методы по очереди.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPLINE: {dataset_name}")
    logger.info(f"{'='*80}")
    
    # Авто-определение target и exclude колонок
    target_col = find_target_column(df)
    exclude_cols = find_exclude_columns(df, target_col)
    logger.info(f"  Target: {target_col}, Exclude: {exclude_cols}")
    
    original_df = df  # Сохраняем для сравнения
    current_df = df
    
    # Получить шаги из конфига
    steps = config.get('pipeline', {}).get('steps', [])
    logger.info(f"  Steps: {[s['method'] for s in steps]}")
    
    for i, step in enumerate(steps):
        method = step['method']
        params = step.get('params', {})
        enabled = step.get('enabled', True)
        
        if not enabled:
            logger.info(f"  Step {i+1}: {method} - SKIPPED (disabled)")
            continue
        
        logger.info(f"\n{'-'*60}")
        logger.info(f"Step {i+1}: {method}")
        logger.info(f"{'-'*60}")
        
        # Словарь методов - просто добавляешь новый метод сюда
        method_map = {
            'drop_null': apply_drop_null,
            'drop_constant': apply_drop_constant,
            'drop_low_variance': apply_drop_low_variance,
            'drop_correlated': apply_drop_correlated,
            'stratified_sampling': apply_stratified_sampling_method,
            'lgbm_shap': apply_lgbm_shap,
            'boruta_shap': apply_boruta_shap,
        }
        
        if method not in method_map:
            logger.error(f"  UNKNOWN METHOD: {method}")
            continue
        
        try:
            # Применяем метод
            current_df = method_map[method](
                current_df, params, target_col, exclude_cols,
                original_df, dataset_name, results_dir
            )
            logger.info(f"  Output: {len(current_df.columns)} columns")
            
        except Exception as e:
            logger.error(f"  ERROR: {str(e)}")
            logger.exception("Stack trace:")
            raise
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL: {len(current_df.columns)} columns")
    logger.info(f"{'='*80}")
    
    return current_df


# === MAIN ===
def main():
    """Основная функция."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*80)
    print("ПИПЛИН ОТБОРА ПРИЗНАКОВ (SIMPLE VERSION)")
    print("="*80)
    
    # Создать папки
    os.makedirs(PIPELINE_DIR, exist_ok=True)
    os.makedirs(os.path.join(PIPELINE_DIR, "logs"), exist_ok=True)
    
    # Логгер
    logger = setup_logger("pipeline", os.path.join(PIPELINE_DIR, "logs"), timestamp)
    logger.info(f"Запуск пайплайна, config={CONFIG_FILE}")
    
    # Загрузить конфиг
    logger.info(f"\nЗагрузка конфига из {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = config.get('datasets', {})
    logger.info(f"Найдено {len(datasets)} датасетов: {list(datasets.keys())}")
    
    # Запустить Spark
    print("\n" + "="*80)
    print("ЗАПУСК SPARK СЕССИИ")
    print("="*80)
    
    start_spark = datetime.now()
    spark = create_spark_session(app_name='simple_feature_selection_pipeline', executor_instances=8)
    spark_time = (datetime.now() - start_spark).total_seconds()
    logger.info(f"Spark сессия запущена! Время: {spark_time:.2f} сек")
    
    # Обработать каждый датасет
    for dataset_name, dataset_config in datasets.items():
        if not dataset_config.get('enabled', True):
            logger.info(f"\n{dataset_name}: пропущено (disabled)")
            continue
        
        print("\n" + "="*80)
        print(f"ОБРАБОТКА: {dataset_name}")
        print("="*80)
        
        # Путь
        base_path = dataset_config.get('path', '')
        has_split_type = dataset_config.get('has_split_type', False)
        
        if has_split_type:
            train_path = os.path.join(base_path, "split_type=train")
        else:
            train_path = base_path
        
        logger.info(f"\nЗагрузка данных: {train_path}")
        
        try:
            # Загрузить данные
            df = spark.read.parquet(train_path)
            total_rows = df.count()
            total_cols = len(df.columns)
            logger.info(f"  Загружено {total_rows:,} строк и {total_cols} колонок")
            
            # Запустить пайплайн
            final_df = run_pipeline(df, config, dataset_name, PIPELINE_DIR, logger)
            logger.info(f"\n  Итог: {len(final_df.columns)} колонок")
            
        except Exception as e:
            logger.error(f"  ОШИБКА: {str(e)}")
            logger.exception("Детали ошибки:")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("ВСЕ ДАТАСЕТЫ ОБРАБОТАНЫ!")
    logger.info("="*80)
    print("\n" + "="*80)
    print("ВСЕ ДАТАСЕТЫ ОБРАБОТАНЫ!")
    print("="*80)
    
    spark.stop()


if __name__ == '__main__':
    main()
