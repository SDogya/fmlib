# Окружение для feature_selection

Полный набор зависимостей, на котором проходят тесты модуля и работает
бенчмарк. Проверено на Python 3.10.12, Linux, 2026-09.

## Сборка

```bash
cd /path/to/repos/feature_selection
python3.10 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -
.venv/bin/python -m pip install \
    "numpy==1.26.4" "scipy==1.11.4" "pandas>=2.0,<2.3" \
    "scikit-learn>=1.3,<1.6" joblib pyarrow \
    "lightgbm>=4.3" catboost shap optuna BorutaShap \
    "pyspark==3.5.3" hydra-core omegaconf \
    pytest ruff polars requests matplotlib seaborn tqdm
```

## Два пина, которые важны

**`scipy==1.11.4`** — `BorutaShap` импортирует `scipy.stats.binom_test`,
удалённый в scipy 1.12 (переименован в `binomtest`). С более новой scipy
`import BorutaShap` падает на `ImportError`.

**`numpy==1.26.4`** — тот же пин, что в `pyproject.toml`; `BorutaShap`
несовместим с numpy 2.x.

С `shap >= 0.45` пин не нужен: несовместимость (бинарный `shap_values`
возвращает один 3-D массив там, где `BorutaShap` ждёт список по классам)
закрыта шимом `shap_binary_list_compat` в `precise/boruta_shap.py`.

## Локальный Spark

Тесты поднимают настоящую `SparkSession` (`local[1]`). Нужна **Java 17** —
`pyspark 3.5` не работает на Java 21+:

```bash
export JAVA_HOME="$HOME/.sdkman/candidates/java/17.0.13-tem"
export PATH="$JAVA_HOME/bin:$PATH"
```

Без Java весь прогон тестов падает на session-фикстуре, включая чисто
pandas-ные тесты (`utils/conftest.py::require_spark_session` вызывает
`pytest.fail`). Для самого пайплайна Spark не обязателен: `fit_select`
принимает pandas-фреймы и без сессии.

## Прогон тестов

```bash
export JAVA_HOME="$HOME/.sdkman/candidates/java/17.0.13-tem"
export PATH="$JAVA_HOME/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python -m pytest fmlib/feature_selection -q
```

Ожидаемо: `462 passed`.

## Бюджет ресурсов

Бенчмарк воспроизводит рабочие ограничения: **8 ядер CPU, 50 GB RAM** на
отбор признаков. Ядра ограничиваются переменными `OMP_NUM_THREADS` /
`MKL_NUM_THREADS` и параметром `n_jobs: 8` в конфигах; память —
`resource.setrlimit(RLIMIT_AS, 50 GB)` в `run_benchmark.py`, чтобы превышение
падало сразу, а не уходило в своп. Снять: `--no-memory-cap`.
