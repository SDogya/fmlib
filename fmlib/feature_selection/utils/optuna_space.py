"""Shared Optuna search-space helpers for model-based selectors.

Parameter blocks are polymorphic: a scalar value is used as-is, a mapping
describes a search space entry. This mirrors the convention already used by
``model.params.parameters`` in the BorutaSHAP selector.
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
    """Split a parameter block into fixed values and a search space.

    Args:
        parameters: Mapping of parameter name to a scalar or a specification.
        method_name: Selector name used in error messages.

    Returns:
        Tuple of ``(fixed, search_space)``. ``fixed`` holds scalars passed to the
        model unchanged, ``search_space`` holds mappings tuned by Optuna.

    Raises:
        ExecutionError: When the block is not a mapping.
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
    """Copy ``defaults``, overlay ``overrides``, and drop pinned scalars.

    Used when the config has no mapping entries: the fallback file is the
    whole space, minus keys the user pinned to a constant. Callers that
    received any YAML mapping must not use this helper — they already have
    the complete user grid.

    Args:
        defaults: Built-in search space of the selector.
        overrides: Search-space entries coming from ``params.parameters``.
        fixed: Scalar entries coming from ``params.parameters``.

    Returns:
        Search space to hand to Optuna.
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
    """Choose the Optuna search space from YAML and the fallback file.

    YAML mappings fully replace the fallback: a single mapping means none of
    the default keys are mixed in. With no mappings and tuning enabled, the
    fallback is used minus pinned scalars. With tuning disabled the space is
    empty and only scalars are returned.

    Args:
        parameters: Raw ``params.parameters`` mapping.
        defaults: Fallback search space for this selector.
        enabled: ``params.optuna_params.enabled``.
        method_name: Selector name used in error messages.

    Returns:
        Tuple of ``(fixed, search_space)``.
    """
    fixed, overrides = split_parameters(parameters, method_name=method_name)
    if not enabled:
        return fixed, {}
    if overrides:
        return fixed, {str(name): dict(spec) for name, spec in overrides.items()}
    return fixed, build_search_space(defaults, overrides={}, fixed=fixed)


def _bounds(specification: Mapping[str, Any], name: str, method_name: str) -> tuple[Any, Any]:
    """Read the low/high bounds of a specification, accepting both dialects."""
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
    """Reject a malformed Optuna search-space entry.

    Supported forms::

        {"type": "int", "min": 4, "max": 8}
        {"type": "float", "min": 0.01, "max": 0.3, "log": true}
        {"type": "categorical", "values": ["a", "b"]}
        {"values": ["a", "b"]}

    Args:
        name: Parameter name.
        specification: Search space entry.
        method_name: Selector name used in error messages.

    Raises:
        ConfigError: When the specification cannot be sampled.
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
    """Generate one Optuna suggestion from a parameter specification.

    Supported forms::

        {"type": "int", "min": 4, "max": 8}
        {"type": "float", "min": 0.01, "max": 0.3, "log": true}
        {"type": "categorical", "values": ["a", "b"]}
        {"values": ["a", "b"]}

    Args:
        trial: Active Optuna trial.
        name: Parameter name.
        specification: Search space entry.
        method_name: Selector name used in error messages.

    Returns:
        Suggested value for this trial.

    Raises:
        ExecutionError: When the specification is malformed.
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
    """Read the shared ``params.optuna_params`` block.

    Every selector that tunes with Optuna accepts the same keys, so a new
    method only has to call this and hand the result to :func:`build_sampler`
    and ``study.optimize``.

    Args:
        optuna_params: Raw ``params.optuna_params`` mapping from the config.
        method_name: Selector name used in error messages.
        n_trials: Fallback trial count for this selector.
        n_startup_trials: Fallback random startup trials for ``TPE``.
        sampler: Fallback sampler name.
        timeout: Fallback wall-clock limit in seconds, ``None`` for unlimited.

    Returns:
        Mapping with ``enabled``, ``n_trials``, ``n_startup_trials``,
        ``sampler``, ``timeout``.

    Raises:
        ExecutionError: When the block or one of its values is invalid.
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
    """Build an Optuna sampler seeded for reproducibility.

    Args:
        optuna_module: Imported ``optuna`` module.
        sampler_name: ``TPE``, ``RANDOM``, or ``GRID``.
        search_space: Parsed search space, required for ``GRID``.
        seed: Deterministic sampler seed.
        n_startup_trials: Random startup trials for ``TPE``.
        method_name: Selector name used in error messages.

    Returns:
        Configured Optuna sampler.

    Raises:
        ExecutionError: When the sampler is unsupported or ``GRID`` is not finite.
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
