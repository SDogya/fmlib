# Benchmark feature selection

Скрипт `feature_selection_benchmark.py` сравнивает наборы признаков с помощью
`CatBoostClassifier`. Для каждого датасета он автоматически добавляет baseline
без удаления признаков, подбирает гиперпараметры на фолдах train, обучает модели
с лучшими параметрами и оценивает ансамбль на фиксированных valid и test.

## Запуск

Необходимы Python 3.11+ и пакеты `catboost`, `numpy`, `optuna`, `pandas`,
`PyYAML` и `scikit-learn`.

```bash
python3 scripts/feature_selection_benchmark.py \
  --config scripts/feature_selection_benchmark.yaml
```

Относительные пути в конфиге разрешаются относительно директории, в которой
находится YAML-файл.

## Секция `benchmark`

| Аргумент | Тип | Описание |
|---|---|---|
| `seed` | `int` | Общий seed для разбиения на фолды, Optuna и CatBoost. По умолчанию `42`. |
| `n_splits` | `int` | Количество stratified-фолдов внутри train. Минимум `2`; в каждом классе train должно быть не меньше этого количества объектов. |
| `run_baseline` | `bool` | Запускать вариант без удаления признаков. По умолчанию `true`. Если `false`, benchmark запускает только подходы из `feature_sets`. |
| `output_dir` | `str` | Директория для JSON-результатов и общего `summary.csv`. |

Одни и те же фолды используются для baseline и всех feature-selection
подходов на конкретном датасете.

## Секция `optuna`

| Аргумент | Тип | Описание |
|---|---|---|
| `n_trials` | `int` | Число trials для каждого сочетания датасета и набора признаков. Минимум `1`. |
| `timeout_seconds` | `int \| null` | Ограничение времени одного запуска `study.optimize`. `null` отключает ограничение. |
| `n_jobs` | `int` | Число параллельных trials. Для наиболее воспроизводимых результатов рекомендуется `1`. |
| `show_progress_bar` | `bool` | Показывать progress bar Optuna. |
| `pruner_startup_trials` | `int` | Сколько trials завершить до включения `MedianPruner`. |
| `storage` | `str \| null` | URL хранилища Optuna, например `sqlite:///benchmark.db`. `null` использует in-memory study. |
| `load_if_exists` | `bool` | Продолжать существующую study с тем же именем при использовании persistent storage. |

Optuna максимизирует средний ROC-AUC только на внутренних фолдах train.
Valid и test не участвуют в подборе гиперпараметров.

## Секция `catboost`

| Аргумент | Тип | Описание |
|---|---|---|
| `thread_count` | `int` | Количество CPU-потоков одной модели CatBoost. `-1` означает использовать все доступные ядра. |
| `early_stopping_rounds` | `int \| null` | Остановить обучение, если ROC-AUC на validation-фолде не улучшается указанное число итераций. |
| `fixed_params` | `mapping` | Параметры CatBoost, одинаковые для всех trials и финальных моделей, например `iterations`. |
| `search_space` | `mapping` | Параметры CatBoost, которые подбирает Optuna. Ключ — имя параметра CatBoost, значение — описание диапазона. |

Скрипт принудительно задаёт `loss_function=Logloss`, `eval_metric=AUC`,
`random_seed`, `verbose=false`, `allow_writing_files=false` и `thread_count`.
Эти значения имеют приоритет над `fixed_params`.

### Форматы параметров `search_space`

Целое число:

```yaml
depth:
  type: int
  low: 4
  high: 10
  step: 1      # необязательно
  log: false   # необязательно
```

Вещественное число:

```yaml
learning_rate:
  type: float
  low: 0.01
  high: 0.3
  step: null   # необязательно
  log: true    # необязательно
```

Категориальный выбор:

```yaml
bootstrap_type:
  type: categorical
  choices:
    - Bayesian
    - Bernoulli
```

## Секция `datasets`

`datasets` — список независимых датасетов. Каждый элемент поддерживает
следующие аргументы:

| Аргумент | Тип | Описание |
|---|---|---|
| `name` | `str` | Уникальное имя датасета; используется в именах study и файлов результатов. |
| `train_path` | `str` | Путь к train split. |
| `valid_path` | `str` | Путь к фиксированному valid split. Он оценивается после Optuna. |
| `test_path` | `str` | Путь к финальному test split. Он оценивается только после Optuna. |
| `target` | `str` | Имя бинарной target-колонки. Каждый split должен содержать оба класса и не иметь пропусков в target. |
| `technical_columns` | `list[str]` | Технические колонки, которые не передаются в модель. |
| `id_columns` | `list[str]` | Колонки, образующие идентификатор объекта для опциональной проверки пересечений split. |
| `enforce_disjoint_ids` | `bool` | Если `true`, завершить запуск с ошибкой при пересечении ID между train, valid и test. Требует непустой `id_columns`. |
| `categorical_columns` | `list[str]` | Категориальные признаки. Они передаются в CatBoost через `cat_features`; пропуски заменяются строкой `__MISSING__`. |
| `continuous_columns` | `list[str]` | Непрерывные признаки. |
| `feature_sets` | `mapping[str, list[str] \| str]` | Имя подхода feature selection и inline-список удалённых признаков либо путь к файлу с этим списком. |

Поддерживаются Parquet-файлы и директории, CSV (`.csv`, `.csv.gz`) и Feather
(`.feather`, `.ft`). Все объявленные колонки должны присутствовать в каждом
split. Категориальные, непрерывные и технические колонки не должны
пересекаться. Нельзя удалить все признаки.

Имя `baseline` зарезервировано: скрипт автоматически создаёт этот вариант с
пустым списком удалённых признаков.

Для больших наборов рекомендуется файл с одним именем колонки на строку.
Пустые строки и строки, начинающиеся с `#`, игнорируются. Относительный путь
разрешается относительно YAML-конфига. Пустой файл допустим; дубликаты и
неизвестные признаки считаются ошибкой.

Можно одновременно передавать несколько файлов и короткие inline-списки:

```yaml
feature_sets:
  mutual_information: feature_sets/mutual_information.txt
  permutation_importance: feature_sets/permutation_importance.txt
  short_experiment:
    - feature_3
    - feature_10
```

Пример содержимого `mutual_information.txt`:

```text
# dropped features
feature_10
feature_24
feature_38
```

Сохранить результат алгоритма можно так:

```python
from pathlib import Path

Path("mutual_information.txt").write_text("\n".join(dropped_features), encoding="utf-8")
```

## Результаты

Для каждого сочетания датасета и подхода создаётся файл
`<dataset>__<method>.json`. Он содержит:

- оставленные и удалённые признаки;
- лучшие параметры и ROC-AUC Optuna;
- validation ROC-AUC и test ROC-AUC каждой fold-модели;
- основной `ensemble_test_auc`, рассчитанный по среднему предсказанию моделей;
- `delta_test_auc_vs_baseline` либо `null`, если baseline отключён;
- seed, hash схемы, hash фолдов, длительность, trials и версии библиотек.

`summary.csv` объединяет основные показатели всех датасетов и подходов.
Среднее test ROC-AUC отдельных fold-моделей помечено как диагностическое;
основной test-показатель — `ensemble_test_auc`.
