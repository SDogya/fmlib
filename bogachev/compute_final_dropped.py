#!/usr/bin/env python3
"""
Скрипт для вычисления дропнутых фичей.

На вход: path к YAML с конфигом
"""

import sys
import os
import yaml
import os.path
import subprocess as sss



sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.insert(0, "/home/datalab/nfs/zaripov/autocampaignxfm")

from pyspark.sql import DataFrame

from tools.spark_session import create_spark_session


def load_selected_columns(filepath: str) -> list:
    """Загрузить список оставленных колонок из текстового файла."""
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_columns_to_file(columns: list, filepath: str) -> None:
    """Сохранить список колонок в файл."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w') as f:
        for col in columns:
            f.write(f"{col}\n")


def main():
    if len(sys.argv) != 2:
        print("Использование:")
        print("  python compute_final_dropped.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = config.get('datasets', {})
    if not datasets:
        print("ERROR: в YAML нет datasets")
        sys.exit(1)
    
    # Создать Spark сессию
    spark = create_spark_session(app_name='compute_final_dropped', executor_instances=8)
    
    for dataset_name, ds_config in datasets.items():
        path = ds_config.get('path')
        selected_file = ds_config.get('selected_file')
        has_split_type = ds_config.get('has_split_type', False)
        
        if not path or not selected_file:
            print(f"SKIP {dataset_name}: нет path или selected_file")
            continue
        
        print(f"\n{dataset_name}:")
        
        # Загрузить данные (с учетом split_type)
        if has_split_type:
            train_path = os.path.join(path, "split_type=train")
        else:
            train_path = path
        print(f"  Загрузка: {train_path}")
        df = spark.read.format("parquet").load(train_path)
        print(f"  Колонок: {len(df.columns)}")
        
        # Загрузить оставленные
        selected_cols = load_selected_columns(selected_file)
        print(f"  Оставлено: {len(selected_cols)} (file: {selected_file})")
        stop_list = set(["target_attr_1", "target_attr_2", "target_attr_3", "bucket_num", "month_part", "split_type", "report_month"])
        # Вычислить дропнутые
        dropped_cols = sorted(set(df.columns) - set(selected_cols) - stop_list)
        print(f"  Дропнуто: {len(dropped_cols)}")
        
        the_way = f"stage_012_DROPPED_{dataset_name}.txt"
        the_way_out = f"/user/team/team_ai_avatar/ds/bogachev/featureselection/pipeline/stage_012_DROPPED_{dataset_name}.txt"
        save_columns_to_file(dropped_cols,the_way )
        print(f"  Сохранено: dropped_final_{dataset_name}.txt")
        sss.run(["hdfs","dfs","-put",the_way,the_way_out])
    print("\nГотово!")


if __name__ == "__main__":
    main()
