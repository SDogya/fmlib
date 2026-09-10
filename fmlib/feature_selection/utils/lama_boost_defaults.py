"""Скорость обучения, лимит деревьев и ранняя остановка по правилам LightAutoML.

Скопировано из LightAutoML ``boost_lgbm.py`` / ``boost_cb.py``.
Optuna не подбирает эти параметры: методы отбора заполняют отсутствующие значения, когда ``n_rows``
загруженной обучающей выборки уже известно. Скаляр YAML сохраняется. Фактическое число деревьев
определяется ранней остановкой; ``n_estimators`` / ``iterations`` задаёт лишь верхнюю границу.

LightGBM использует одну таблицу зависимости от числа строк для всех задач (как в LightAutoML). CatBoost
использует отдельные таблицы для бинарной классификации, многоклассовой классификации и регрессии.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

BoostLibrary = Literal["lightgbm", "catboost"]

_LR_KEYS = frozenset({"learning_rate", "eta", "shrinkage_rate"})
_ES_KEYS = frozenset({"early_stopping_rounds", "od_wait"})
_STRIP_FROM_SPACE = _LR_KEYS | _ES_KEYS

_LGBM_TREE_CAP_KEYS = (
    "n_estimators",
    "num_iterations",
    "num_trees",
    "num_boost_round",
    "num_tree",
    "nrounds",
)
_CB_TREE_CAP_KEYS = (
    "iterations",
    "n_estimators",
    "num_trees",
    "num_boost_round",
)


def _has_any_key(mapping: Mapping[str, Any], keys: frozenset[str] | tuple[str, ...]) -> bool:
    """Возвращает, задано ли уже в ``mapping`` фиксированное значение для любого имени из ``keys``."""
    return any(key in mapping for key in keys)


def boost_fixed_params(
    n_rows: int,
    *,
    library: BoostLibrary,
    task_type: str = "binary_classification",
) -> dict[str, Any]:
    """Возвращает табличные ``learning_rate``, лимит деревьев и число раундов ожидания до ранней остановки.

    Args:
        n_rows: Число строк загруженной выборки train, фактически используемых для обучения
            (обучающая часть временного разбиения CatBoost RFE; локальная выборка LightGBM/Boruta).
        library: ``lightgbm`` или ``catboost``.
        task_type: ``binary_classification``, ``classification`` или
            ``regression``. LightGBM игнорирует значение (одна таблица). CatBoost выбирает
            соответствующую таблицу LightAutoML по числу строк.

    Returns:
        Словарь параметров для объединения с блоком фиксированных значений метода отбора.

    Raises:
        ValueError: Если ``library`` не является поддерживаемым бэкендом бустинга.
    """
    rows = max(0, int(n_rows))
    if library == "lightgbm":
        return _lightgbm_table(rows)
    if library == "catboost":
        return _catboost_table(rows, task_type=task_type)
    msg = f"unsupported boost library {library!r}"
    raise ValueError(msg)


def apply_boost_heuristics(
    fixed: Mapping[str, Any] | None,
    search_space: Mapping[str, Mapping[str, Any]] | None,
    *,
    n_rows: int,
    library: BoostLibrary,
    task_type: str = "binary_classification",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Заполняет отсутствующие lr/patience/cap по таблице; сохраняет скаляры YAML.

    Optuna никогда не подбирает ``learning_rate`` / ``eta`` / ``early_stopping_rounds``
    / ``od_wait``: эти словари удаляются из ``search_space``. Скаляр,
    уже заданный в ``fixed`` (включая псевдонимы), не меняется. Отсутствующие ключи
    берутся из таблицы LightAutoML по числу строк. Лимит деревьев и параметр CatBoost
    ``use_best_model`` заполняются по тому же правилу: только если отсутствуют. Словарь для
    лимита деревьев допустим и остаётся в ``search_space``.

    Args:
        fixed: Скалярные параметры из ``resolve_tuning_space``.
        search_space: Спецификации Optuna из ``resolve_tuning_space``.
        n_rows: Размер загруженной выборки train для поиска в таблице.
        library: ``lightgbm`` или ``catboost``.
        task_type: Задача моделирования; только таблицы CatBoost зависят от задачи.

    Returns:
        Кортеж ``(fixed, search_space)`` после применения настроек LightAutoML.
    """
    table = boost_fixed_params(n_rows, library=library, task_type=task_type)
    new_space = {
        str(name): dict(spec)
        for name, spec in dict(search_space or {}).items()
        if str(name) not in _STRIP_FROM_SPACE
    }
    new_fixed = dict(fixed or {})
    if not _has_any_key(new_fixed, _LR_KEYS):
        new_fixed["learning_rate"] = table["learning_rate"]
    if not _has_any_key(new_fixed, _ES_KEYS):
        new_fixed["early_stopping_rounds"] = table["early_stopping_rounds"]
    if library == "catboost":
        if "use_best_model" not in new_fixed:
            new_fixed["use_best_model"] = table["use_best_model"]
        cap_keys = _CB_TREE_CAP_KEYS
        cap_name = "iterations"
    else:
        cap_keys = _LGBM_TREE_CAP_KEYS
        cap_name = "n_estimators"
    if not _has_any_key(new_fixed, cap_keys):
        new_fixed[cap_name] = table[cap_name]
    return new_fixed, new_space


def split_lgbm_early_stopping(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None]:
    """Удаляет параметры ожидания ранней остановки, не предназначенные для конструктора LightGBM.

    ``early_stopping_rounds <= 0`` означает обучение до лимита деревьев без
    обработчика ранней остановки.
    """
    params = dict(parameters)
    raw = params.pop("early_stopping_rounds", None)
    params.pop("od_wait", None)
    if raw is None:
        return params, None
    rounds = int(raw)
    if rounds <= 0:
        return params, None
    return params, rounds


def fit_lgbm_with_early_stopping(
    model: Any,
    features: Any,
    target: Any,
    *,
    eval_set: Any,
    early_stopping_rounds: int | None,
) -> Any:
    """Обучает модель LightGBM с ранней остановкой при наличии eval.

    Использует ``callbacks=[early_stopping(...)]`` в LightGBM 4+, а в более ранних версиях
    передаёт аргумент ``early_stopping_rounds=`` в fit.

    Args:
        model: Созданный ``LGBMClassifier`` (параметры ожидания ранней остановки уже удалены).
        features: Матрица обучающих признаков.
        target: Обучающие метки.
        eval_set: Аргумент ``eval_set`` для LightGBM или ``None``.
        early_stopping_rounds: Число раундов ожидания, ``None`` или ``<= 0`` для обучения до
            лимита без обработчика остановки.

    Returns:
        Обученная ``model``.
    """
    if (
        early_stopping_rounds is None
        or int(early_stopping_rounds) <= 0
        or eval_set is None
    ):
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        model.fit(features, target, **fit_kwargs)
        return model

    import lightgbm as lgb

    rounds = int(early_stopping_rounds)
    try:
        try:
            callback = lgb.early_stopping(rounds, verbose=False)
        except TypeError:
            callback = lgb.early_stopping(rounds)
        model.fit(features, target, eval_set=eval_set, callbacks=[callback])
    except TypeError:
        model.fit(
            features,
            target,
            eval_set=eval_set,
            early_stopping_rounds=rounds,
        )
    return model


def _lightgbm_table(n_rows: int) -> dict[str, Any]:
    """Таблица LightAutoML ``init_params_on_input`` по числу строк (все задачи)."""
    if n_rows <= 10_000:
        lr, trees, patience = 0.01, 3000, 200
    elif n_rows <= 20_000:
        lr, trees, patience = 0.02, 3000, 200
    elif n_rows <= 100_000:
        lr, trees, patience = 0.03, 1200, 200
    elif n_rows <= 300_000:
        lr, trees, patience = 0.04, 2000, 100
    else:
        lr, trees, patience = 0.05, 2000, 100
    return {
        "learning_rate": lr,
        "n_estimators": trees,
        "early_stopping_rounds": patience,
    }


def _catboost_table(
    n_rows: int,
    *,
    task_type: str = "binary_classification",
) -> dict[str, Any]:
    """Таблица LightAutoML CatBoost ``num_trees`` / ``learning_rate``."""
    if task_type == "classification":
        lr = 0.03
        trees = 3000 if n_rows <= 100_000 else 4000
        patience = 100
    elif task_type == "regression":
        lr, trees, patience = 0.05, 2000, 300
    elif n_rows <= 6_000:
        lr, trees, patience = 0.02, 500, 100
    elif n_rows <= 20_000:
        lr, trees, patience = 0.035, 5000, 100
    elif n_rows <= 50_000:
        lr, trees, patience = 0.03, 5000, 100
    elif n_rows <= 60_000:
        lr, trees, patience = 0.05, 2000, 100
    elif n_rows <= 100_000:
        lr, trees, patience = 0.045, 1500, 100
    elif n_rows <= 150_000:
        lr, trees, patience = 0.045, 3000, 100
    elif n_rows <= 300_000:
        lr, trees, patience = 0.045, 2000, 100
    else:
        lr, trees, patience = 0.05, 3000, 100
    return {
        "learning_rate": lr,
        "iterations": trees,
        "early_stopping_rounds": patience,
        "use_best_model": True,
    }
