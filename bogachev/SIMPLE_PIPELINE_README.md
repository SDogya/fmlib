# Простой гайд: как использовать new_simple_pipeline.py

## Сравнение подходов

### Старый pipeline.py (сложный)
```
860 строк кода
Много вложенных функций
Жесткая логика переключения методов
Понимание - 30 минут
Добавление метода - 10 минут
```

### Новый new_simple_pipeline.py (простой)
```
350 строк кода
Простая функция run_pipeline()
Методы в словаре method_map
Понимание - 5 минут
Добавление метода - 2 минуты
```

## Использование

### Шаг 1: Создайте config.yaml

```yaml
datasets:
  MY_DATASET:
    enabled: true
    path: /path/to/data
    has_split_type: false

pipeline:
  steps:
    - method: drop_constant
      params:
        tol: 0.98
    - method: boruta_shap
      params:
        boruta_trials: 50
```

### Шаг 2: Запустите

```bash
cd /home/datalab/nfs/bogachev/fmlib-feature-selection/dev/bogachev
python new_simple_pipeline.py
```

## Добавление нового метода - 2 минуты

### 1. Создайте функцию применения (уже есть шаблоны):

```python
def apply_my_new_method(df, params, target_col, exclude_cols, ...):
    """Применить мой новый метод."""
    logger.info(f"  my_new_method: param1={params.get('param1')}")
    result = my_new_function(df, **params)
    return result
```

### 2. Добавьте в method_map:

```python
method_map = {
    'drop_null': apply_drop_null,
    'drop_constant': apply_drop_constant,
    # ... другие
    'my_new_method': apply_my_new_method,  # <-- добавил!
}
```

### 3. Используйте в YAML:

```yaml
pipeline:
  steps:
    - method: my_new_method
      params:
        param1: value1
        param2: value2
```

## Список готовых методов

| Метод | Параметры | Описание |
|-------|-----------|----------|
| `drop_null` | `max_null_fraction: 0.5` | Удалить колонки с многими NULL |
| `drop_constant` | `tol: 0.98` | Удалить константные признаки |
| `drop_low_variance` | `min_variance: 1e-5` | Удалить низкодисперсные признаки |
| `drop_correlated` | `corr_threshold: 0.95` | Удалить скоррелированные признаки |
| `stratified_sampling` | `max_rows: 750000` | Стратифицированная выборка |
| `lgbm_shap` | `threshold: 0.85` | LGBM + SHAP отбор |
| `boruta_shap` | `boruta_trials: 50` | BorutaSHAP отбор |

## Почему просто?

1. **Нет классов** - только функции
2. **Нет registry** - просто словарь методов
3. **YAML прямолинеен** - шаги идут подряд
4. **Легко расширять** - добавил функцию + одну строчку в словарь

## Миграция с pipeline_conf.yaml

### Старый формат
```yaml
global_methods:
  order:
    - drop_constant
    - drop_low_variance
    - boruta_shap
  drop_constant:
    enabled: true
    param: 0.98
```

### Новый формат
```yaml
pipeline:
  steps:
    - method: drop_constant
      params:
        tol: 0.98
    - method: drop_low_variance
      params:
        min_variance: 1e-5
    - method: boruta_shap
      params:
        boruta_trials: 50
```

## Результаты

Все результаты сохраняются в:
```
results/pipeline/DATASET_NAME/METHOD_cols.txt
```

---

**main() всего 50 строк!** Никакой магии, только простой код.
