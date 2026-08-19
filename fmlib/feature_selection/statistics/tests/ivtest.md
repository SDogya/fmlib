
# TASK: 100% TEST COVERAGE FOR `fmlib.feature_selection.iv` (IvSelector)

## ВНИМАНИЕ: ЖЕСТКИЕ ПРАВИЛА ДЛЯ АГЕНТА (READ THIS FIRST)

> **НИКАКОЙ САМОДЕЯТЕЛЬНОСТИ И ХАЛТУРЫ.** Ты пишешь production-grade тесты. 
> Если тест написан «для галочки», не проверяет реальные значения или падает при изменении одного условия — задание провалено.

### СТРОГИЕ ЗАПРЕТЫ:
1. **ЗАПРЕЩЕНО писать заглушки, `pass`, `TODO` или неполные тесты.** Каждый тестовый кейс обязан содержать явные ассерты на реальные входные/выходные данные.
2. **ЗАПРЕЩЕНО мокать чистые функции.** `information_value`, `_quantile_bins_pandas`, `_categorical_bins_pandas`, `_merge_categorical_counts`, `_binary_mapping`, `_levels_to_keep` тестируются НА РЕАЛЬНЫХ ДАННЫХ, а не через `unittest.mock`.
3. **ЗАПРЕЩЕНО игнорировать ветвления (branches).** Условие `if/else`, блоки `try/except/finally`, `should_unpersist` и парсинг ошибок обязаны быть покрыты тестами на 100%.
4. **ЗАПРЕЩЕНО менять продакшн-код** `fmlib/feature_selection/iv.py` без крайней необходимости. Если тест падает — сначала проверь корректность теста.

---

## 1. ЦЕЛЬ ЗАДАЧИ
Создать файл с тестами (например, `tests/feature_selection/test_iv.py`), который покрывает модуль `fmlib.feature_selection.iv` на **100% statement и branch coverage**.

---

## 2. ЧТО И КАК ТЫ ОБЯЗАН РЕАЛИЗОВАТЬ

### БЛОК 1: Чистая математика и вспомогательные функции (БЕЗ МОКОВ)
Обязательно протестировать следующие функции напрямую:

1. **`information_value(goods, bads, eps)`**:
   - `total_good <= 0` или `total_bad <= 0` -> возврат строго `0.0`.
   - Пропуск пустых бинов (`good <= 0 and bad <= 0`).
   - Проверка точности математики против ручного расчета формулы:
     $$\sum (\text{dist\_good} - \text{dist\_bad}) \cdot \ln\left(\frac{\text{dist\_good} + \text{eps}}{\text{dist\_bad} + \text{eps}}\right)$$

2. **`information_value_from_bins(y, bins, eps)`**:
   - Корректная группировка по `pd.unique(bins)`.
   - Подсчет `goods` и `bads` для каждого уникального бина.

3. **`_quantile_bins_pandas(series, num_bins)`**:
   - Серия только из `NaN` / `None` -> все элементы должны стать `__iv_null__`.
   - Константная колонка (1 уникальное значение) -> `c0` + `__iv_null__` для пропусков.
   - Обычное квантильное разбиение на `num_bins`.
   - Срабатывание `except ValueError` внутри `pd.qcut` -> fallback на `c0`.

4. **`_categorical_bins_pandas(series, min_bin_share, max_levels)`**:
   - Выделение `__iv_null__` для `NaN`/`None`.
   - Схлопывание редких категорий (ниже `min_bin_share`) в `__iv_other__`.
   - Ограничение по `max_levels`.

5. **`_levels_to_keep(counts, n_rows, min_bin_share, max_levels)`**:
   - Сортировка по частоте (убывание) и алфавиту (возрастание).
   - Фильтрация по `min_bin_share`.
   - Срез по `max_levels`.

6. **`_merge_categorical_counts(levels, min_bin_share, max_levels)`**:
   - Корректное объединение счетчиков `goods`/`bads` для категорий, ушедших в `__iv_other__`.
   - Сохранение `__iv_null__`.

7. **`_binary_mapping(unique, method)`**:
   - Числовые и булевы значения `{0, 1}`, `{0.0, 1.0}`, `{False, True}`.
   - Строковые метки (например, `["no", "yes"]`) -> сортировка и маппинг в `{0.0, 1.0}`.
   - Несравнимые типы (проверка ветки `except TypeError: sorted(..., key=str)`).
   - Граничные случаи: 1 уникальное значение (`{label: 1.0}`), пустой список (`{}`).
   - Ошибка: > 2 классов -> выброс `ExecutionError`.

8. **`_is_null(value)` и `_root_cause(exc)`**:
   - `_is_null`: скалярные `None`, `np.nan`, `pd.NA`, обычные типы, кастомные объекты (проверка ветки `except (TypeError, ValueError)`).
   - `_root_cause`: `java_exception`, `__cause__`, стандартный `Exception`.

---

### БЛОК 2: Контракты, валидация и принятие решений (`IvSelector.select`)

1. **Валидация схемы и входа:**
   - Пустой список `candidates` -> немедленный возврат `[]`.
   - `schema.target is None` -> выброс `ConfigError`.
   - `schema.task_type != "binary_classification"` -> выброс `ConfigError`.
   - Неподдерживаемый тип датасета (не pandas и не Spark) -> выброс `ExecutionError`.

2. **Логика отбора (`FeatureDecision`):**
   - $IV < \text{threshold}$ -> `keep=False`, `reason="low_iv"`.
   - $\text{threshold} \le IV \le \text{max\_threshold}$ -> признак сохраняется (нет drop decision).
   - $IV > \text{max\_threshold}$ -> `keep=False`, `reason="high_iv"`.

3. **Debug-логирование:**
   - При `debug_enabled == True` проверить факт вызова `debug_emit` со всеми переданными аргументами (`n_evaluated`, `n_below_threshold`, `iv_min`, `iv_max`, `iv_mean` и т.д.).

---

### БЛОК 3: Pandas Execution Path (`_compute_iv_pandas`)

1. Отсутствие колонок в DataFrame -> выброс `ExecutionError`.
2. Датасет пуст после удаления `NaN` из таргета -> возврат `0.0` для всех фичей.
3. Полный end-to-end прогон на DataFrame со смешанными типами:
   - Непрерывные фичи (`continuous`).
   - Категориальные фичи (`categorical`).
   - Фичи с пропусками (`np.nan`).

---

### БЛОК 4: Spark Execution Path (`_compute_iv_spark`, `_spark_numeric_iv`, `_spark_categorical_iv`)
*(Реализовать через контролируемые `MagicMock` или локальную сессию PySpark, если доступна)*

1. **Импорт и окружение:**
   - Отсутствие `pyspark` в `sys.modules` -> выброс `BackendError`.
   - `_is_spark_dataframe`: проверка детекции объектов Spark.

2. **Валидация Spark-схемы:**
   - Пропущенные колонки в `train.schema.fields` -> выброс `ExecutionError`.
   - Нечисловой тип для continuous-фичи (например, `StringType`) -> выброс `ExecutionError`.

3. **Управление памятью (Caching):**
   - Если `prepared.is_cached == False` -> вызовы `.persist()` и `.unpersist()` в блоке `finally`.
   - Если `prepared.is_cached == True` -> не вызывать persist повторно.

4. **Батчинг и агрегации:**
   - Проверить разбиение колонок по `batch_size`.
   - Обработка исключений в `approxQuantile` и `frame.agg` -> выброс `ExecutionError` с парсингом `_root_cause`.
   - Сбор и расчет `scores` для числовых и категориальных колонок.
   - Проверка вспомогательных функций Spark: `_spark_binary_target`, `_count_exprs`, `_quoted_col`.

---

## 3. ПОШАГОВЫЙ ПЛАН ВЫПОЛНЕНИЯ

1. **Создай/открой файл тестов:** `tests/feature_selection/test_iv.py`.
2. **Импортируй все необходимые сущности** из `fmlib.feature_selection.iv` и сопутствующих модулей.
3. **Напиши фикстуры и вспомогательные классы** (`DummySchema`, `make_context`).
4. **Реализуй тест-сьюты** по блокам:
   - `TestMathAndHelperFunctions`
   - `TestIvSelectorValidationAndDecisions`
   - `TestIvSelectorPandas`
   - `TestIvSelectorSpark`
5. **Запусти проверку покрытия:**
   ```bash
   pytest --cov=fmlib.feature_selection.iv --cov-report=term-missing --cov-fail-under=100 tests/feature_selection/test_iv.py

```

6. **Если покрытие < 100%:** найди пропущенные строки в выводе `pytest --cov-report=term-missing` и добавь недостающие тест-кейсы на конкретные ветвления.

