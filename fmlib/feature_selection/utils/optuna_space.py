"""Общие функции работы с пространством поиска Optuna для методов отбора на основе моделей.

Блоки параметров полиморфны: скалярное значение используется как есть, а словарь
описывает элемент пространства поиска. Это соответствует соглашению, уже используемому в
``model.params.parameters`` метода отбора BorutaSHAP.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from fmlib.feature_selection.exceptions import ConfigError, ExecutionError

SAMPLERS = frozenset({"TPE", "RANDOM", "GRID"})
PARAMETER_TYPES = frozenset({"int", "float", "categorical"})


def split_parameters(
    parameters: Mapping[str, Any],
    *,
    method_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Разделяет блок параметров на фиксированные значения и пространство поиска.

    Args:
        parameters: Словарь имён параметров и соответствующих скаляров или спецификаций.
        method_name: Имя метода отбора для сообщений об ошибках.

    Returns:
        Кортеж ``(fixed, search_space)``. ``fixed`` содержит скаляры, передаваемые в
        модель без изменений, а ``search_space`` — словари параметров, подбираемых Optuna.

    Raises:
        ExecutionError: Если блок не является словарём.
    """
    if not isinstance(parameters, Mapping):
        msg = f"{method_name}: params.parameters must be a mapping."
        raise ExecutionError(msg)

    fixed: dict[str, Any] = {}
    search_space: dict[str, dict[str, Any]] = {}
    for name, value in parameters.items():
        key = str(name)
        if isinstance(value, Mapping):
            search_space[key] = dict(value)
        else:
            fixed[key] = value
    return fixed, search_space


def build_search_space(
    defaults: Mapping[str, Mapping[str, Any]],
    *,
    overrides: Mapping[str, Mapping[str, Any]],
    fixed: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Копирует ``defaults``, применяет ``overrides`` и удаляет параметры с фиксированными скалярными значениями.

    Используется, если в конфигурации нет значений-словарей: файл настроек по умолчанию задаёт
    всё пространство, кроме ключей, зафиксированных пользователем как константы. Вызывающий код,
    получивший хотя бы один словарь YAML, не должен использовать эту функцию — у него уже есть
    полная пользовательская сетка.

    Args:
        defaults: Встроенное пространство поиска метода отбора.
        overrides: Элементы пространства поиска из ``params.parameters``.
        fixed: Скалярные значения из ``params.parameters``.

    Returns:
        Пространство поиска для передачи в Optuna.
    """
    space = {str(name): dict(spec) for name, spec in defaults.items()}
    space.update({str(name): dict(spec) for name, spec in overrides.items()})
    for name in fixed:
        space.pop(str(name), None)
    return space


def resolve_tuning_space(
    parameters: Mapping[str, Any],
    *,
    defaults: Mapping[str, Mapping[str, Any]],
    enabled: bool,
    method_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Выбирает пространство поиска Optuna из YAML и файла настроек по умолчанию.

    Словари YAML полностью заменяют пространство поиска по умолчанию: даже один словарь означает, что ни один
    ключ по умолчанию не добавляется. При отсутствии словарей и включённом подборе
    используется пространство поиска по умолчанию без фиксированных скаляров. При отключённом подборе пространство
    пусто и возвращаются только скаляры.

    Args:
        parameters: Исходный словарь ``params.parameters``.
        defaults: Пространство поиска по умолчанию для этого метода отбора.
        enabled: ``params.optuna_params.enabled``.
        method_name: Имя метода отбора для сообщений об ошибках.

    Returns:
        Кортеж ``(fixed, search_space)``.
    """
    fixed, overrides = split_parameters(parameters, method_name=method_name)
    if not enabled:
        return fixed, {}
    if overrides:
        return fixed, {str(name): dict(spec) for name, spec in overrides.items()}
    return fixed, build_search_space(defaults, overrides={}, fixed=fixed)


def _bounds(specification: Mapping[str, Any], name: str, method_name: str) -> tuple[Any, Any]:
    """Читает нижнюю и верхнюю границы спецификации, поддерживая оба варианта записи."""
    low = specification.get("min", specification.get("low"))
    high = specification.get("max", specification.get("high"))
    if low is None or high is None:
        msg = (
            f"{method_name}: parameter {name!r} must define 'min' and 'max' "
            "(or 'low' and 'high')."
        )
        raise ConfigError(msg)
    return low, high


def validate_parameter_spec(
    name: str,
    specification: Mapping[str, Any],
    *,
    method_name: str,
) -> None:
    """Отклоняет некорректный элемент пространства поиска Optuna.

    Поддерживаемые формы::

        {"type": "int", "min": 4, "max": 8}
        {"type": "float", "min": 0.01, "max": 0.3, "log": true}
        {"type": "categorical", "values": ["a", "b"]}
        {"values": ["a", "b"]}

    Args:
        name: Имя параметра.
        specification: Элемент пространства поиска.
        method_name: Имя метода отбора для сообщений об ошибках.

    Raises:
        ConfigError: Если из спецификации невозможно выбрать значение.
    """
    if not isinstance(specification, Mapping):
        msg = f"{method_name}: parameter {name!r} search space must be a mapping."
        raise ConfigError(msg)

    has_values = "values" in specification
    parameter_type = str(
        specification.get("type", "categorical" if has_values else "float"),
    )
    if parameter_type not in PARAMETER_TYPES:
        msg = (
            f"{method_name}: parameter {name!r} has unsupported type="
            f"{parameter_type!r}. Expected one of: {sorted(PARAMETER_TYPES)}."
        )
        raise ConfigError(msg)

    if has_values and parameter_type != "categorical":
        msg = (
            f"{method_name}: parameter {name!r} combines 'values' with "
            f"type={parameter_type!r}. An explicit list is always a categorical "
            "choice: use type='categorical' or omit 'type'."
        )
        raise ConfigError(msg)

    if parameter_type == "categorical":
        values = specification.get("values")
        if not isinstance(values, (list, tuple)) or not values:
            msg = (
                f"{method_name}: categorical parameter {name!r} must define a "
                "non-empty 'values' list."
            )
            raise ConfigError(msg)
        return

    low, high = _bounds(specification, name, method_name)
    try:
        if parameter_type == "int":
            low_value = int(low)
            high_value = int(high)
        else:
            low_value = float(low)
            high_value = float(high)
    except (TypeError, ValueError) as exc:
        msg = f"{method_name}: parameter {name!r} has invalid bounds: {exc}."
        raise ConfigError(msg) from exc
    if low_value > high_value:
        msg = (
            f"{method_name}: parameter {name!r} has min > max "
            f"({low_value} > {high_value})."
        )
        raise ConfigError(msg)


def suggest_parameter(
    trial: Any,
    name: str,
    specification: Mapping[str, Any],
    *,
    method_name: str,
) -> Any:
    """Предлагает одно значение Optuna по спецификации параметра.

    Поддерживаемые формы::

        {"type": "int", "min": 4, "max": 8}
        {"type": "float", "min": 0.01, "max": 0.3, "log": true}
        {"type": "categorical", "values": ["a", "b"]}
        {"values": ["a", "b"]}

    Args:
        trial: Активное испытание Optuna.
        name: Имя параметра.
        specification: Элемент пространства поиска.
        method_name: Имя метода отбора для сообщений об ошибках.

    Returns:
        Значение, предложенное для этого испытания.

    Raises:
        ExecutionError: Если спецификация некорректна.
    """
    try:
        validate_parameter_spec(name, specification, method_name=method_name)
    except ConfigError as exc:
        raise ExecutionError(str(exc)) from exc

    has_values = "values" in specification
    parameter_type = str(
        specification.get("type", "categorical" if has_values else "float"),
    )
    if parameter_type == "categorical":
        return trial.suggest_categorical(name, list(specification["values"]))

    low, high = _bounds(specification, name, method_name)
    try:
        if parameter_type == "int":
            # log is passed only when requested: integer search spaces rarely use it,
            # and omitting it keeps the call compatible with simpler trial objects.
            if specification.get("log", False):
                return trial.suggest_int(name, int(low), int(high), log=True)
            return trial.suggest_int(name, int(low), int(high))
        return trial.suggest_float(
            name,
            float(low),
            float(high),
            log=bool(specification.get("log", False)),
        )
    except (TypeError, ValueError) as exc:
        msg = f"{method_name}: parameter {name!r} has invalid bounds: {exc}."
        raise ExecutionError(msg) from exc


def resolve_optuna_settings(
    optuna_params: Any,
    *,
    method_name: str,
    n_trials: int = 20,
    n_startup_trials: int = 10,
    sampler: str = "TPE",
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """Читает общий блок ``params.optuna_params``.

    Все методы отбора с подбором через Optuna принимают одни и те же ключи, поэтому новому
    методу достаточно вызвать эту функцию и передать результат в :func:`build_sampler`
    и ``study.optimize``.

    Args:
        optuna_params: Исходный словарь ``params.optuna_params`` из конфигурации.
        method_name: Имя метода отбора для сообщений об ошибках.
        n_trials: Число испытаний по умолчанию для этого метода отбора.
        n_startup_trials: Число начальных случайных испытаний по умолчанию для ``TPE``.
        sampler: Имя сэмплера по умолчанию.
        timeout: Лимит фактического времени по умолчанию в секундах; ``None`` — без ограничения.

    Returns:
        Словарь с ``enabled``, ``n_trials``, ``n_startup_trials``,
        ``sampler``, ``timeout``.

    Raises:
        ExecutionError: Если блок или одно из его значений некорректны.
    """
    if not isinstance(optuna_params, Mapping):
        msg = f"{method_name}: params.optuna_params must be a mapping."
        raise ExecutionError(msg)

    raw_enabled = optuna_params.get("enabled", True)
    if not isinstance(raw_enabled, bool):
        msg = f"{method_name}: optuna_params.enabled must be boolean."
        raise ExecutionError(msg)

    raw_timeout = optuna_params.get("timeout", timeout)
    try:
        settings: dict[str, Any] = {
            "enabled": raw_enabled,
            "n_trials": int(optuna_params.get("n_trials", n_trials)),
            "n_startup_trials": int(
                optuna_params.get("n_startup_trials", n_startup_trials),
            ),
            "sampler": str(optuna_params.get("sampler", sampler)).upper(),
            "timeout": None if raw_timeout is None else int(raw_timeout),
        }
    except (TypeError, ValueError) as exc:
        msg = f"{method_name}: invalid params.optuna_params value. Root cause: {exc}."
        raise ExecutionError(msg) from exc

    for name in ("n_trials", "n_startup_trials"):
        if settings[name] < 1:
            msg = f"{method_name}: optuna_params.{name} must be at least 1."
            raise ExecutionError(msg)
    if settings["timeout"] is not None and settings["timeout"] < 1:
        msg = f"{method_name}: optuna_params.timeout must be at least 1 second or null."
        raise ExecutionError(msg)
    if settings["sampler"] not in SAMPLERS:
        msg = (
            f"{method_name}: optuna_params.sampler must be one of {sorted(SAMPLERS)}; "
            f"got {settings['sampler']!r}."
        )
        raise ExecutionError(msg)
    return settings


def build_sampler(
    optuna_module: Any,
    *,
    sampler_name: str,
    search_space: Mapping[str, Mapping[str, Any]],
    seed: int,
    n_startup_trials: int,
    method_name: str,
) -> Any:
    """Создаёт сэмплер Optuna с заданным seed для воспроизводимости.

    Args:
        optuna_module: Импортированный модуль ``optuna``.
        sampler_name: ``TPE``, ``RANDOM`` или ``GRID``.
        search_space: Пространство поиска после парсинга; обязательно для ``GRID``.
        seed: Детерминированный seed сэмплера.
        n_startup_trials: Число начальных случайных испытаний для ``TPE``.
        method_name: Имя метода отбора для сообщений об ошибках.

    Returns:
        Настроенный сэмплер Optuna.

    Raises:
        ExecutionError: Если сэмплер не поддерживается или ``GRID`` не является конечным.
    """
    normalized = str(sampler_name).upper()
    if normalized not in SAMPLERS:
        msg = f"{method_name}: sampler must be one of {sorted(SAMPLERS)}; got {sampler_name!r}."
        raise ExecutionError(msg)

    if normalized == "RANDOM":
        return optuna_module.samplers.RandomSampler(seed=seed)
    if normalized == "GRID":
        grid: dict[str, list[Any]] = {}
        for name, specification in search_space.items():
            values = specification.get("values")
            if not isinstance(values, (list, tuple)) or not values:
                msg = f"{method_name}: GRID sampler requires parameter {name!r} to define a non-empty 'values' list."
                raise ExecutionError(msg)
            grid[name] = list(values)
        return optuna_module.samplers.GridSampler(search_space=grid, seed=seed)
    return optuna_module.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
