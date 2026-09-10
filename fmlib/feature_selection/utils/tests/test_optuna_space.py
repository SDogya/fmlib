"""Тесты функций Optuna, общих для всех методов отбора с подбором параметров.

Парсинг конфигурации выполняется локально. ``suggest_parameter`` и ``build_sampler`` используют
реальные испытания и сэмплеры Optuna; отсутствие Optuna завершает запуск ошибкой, а не пропуском.
"""

from __future__ import annotations

from typing import Any

import pytest

from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.utils.optuna_space import (
    PARAMETER_TYPES,
    SAMPLERS,
    build_sampler,
    build_search_space,
    resolve_optuna_settings,
    resolve_tuning_space,
    split_parameters,
    suggest_parameter,
)

_METHOD = "test_method"


def _require_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        pytest.fail(f"Install the optuna extra. Root cause: {exc}")
    return optuna


# --- split_parameters ------------------------------------------------------


def test_scalars_are_fixed_and_mappings_become_a_search_space() -> None:
    fixed, search_space = split_parameters(
        {
            "iterations": 3000,
            "loss_function": "Logloss",
            "depth": {"type": "int", "min": 4, "max": 8},
        },
        method_name=_METHOD,
    )

    assert fixed == {"iterations": 3000, "loss_function": "Logloss"}
    assert set(search_space) == {"depth"}


def test_an_all_scalar_block_leaves_the_search_space_empty() -> None:
    fixed, search_space = split_parameters({"depth": 6}, method_name=_METHOD)

    assert fixed == {"depth": 6}
    assert search_space == {}


def test_a_non_mapping_block_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="must be a mapping"):
        split_parameters(["depth"], method_name=_METHOD)


# --- build_search_space ----------------------------------------------------


_DEFAULTS = {
    "depth": {"type": "int", "min": 3, "max": 10},
    "learning_rate": {"type": "float", "min": 0.01, "max": 0.2},
}


def test_overrides_replace_defaults_point_by_point() -> None:
    space = build_search_space(
        _DEFAULTS,
        overrides={"depth": {"type": "int", "min": 4, "max": 6}},
        fixed={},
    )

    assert space["depth"] == {"type": "int", "min": 4, "max": 6}
    assert space["learning_rate"] == _DEFAULTS["learning_rate"]


def test_pinned_parameters_leave_the_search_space() -> None:
    # Without this, Optuna keeps suggesting values and the suggestion silently
    # overrides the configured constant.
    space = build_search_space(
        _DEFAULTS,
        overrides={},
        fixed={"learning_rate": 0.05},
    )

    assert "learning_rate" not in space
    assert "depth" in space


def test_defaults_are_copied_not_shared() -> None:
    space = build_search_space(_DEFAULTS, overrides={}, fixed={})
    space["depth"]["min"] = 99

    assert _DEFAULTS["depth"]["min"] == 3


# --- suggest_parameter -----------------------------------------------------


def test_int_and_float_ranges_accept_both_key_dialects() -> None:
    optuna = _require_optuna()
    captured: dict[str, Any] = {}

    def objective(trial: Any) -> float:
        captured["learning_rate"] = suggest_parameter(
            trial,
            "learning_rate",
            {"type": "float", "min": 0.01, "max": 0.3, "log": True},
            method_name=_METHOD,
        )
        captured["depth"] = suggest_parameter(
            trial,
            "depth",
            {"type": "int", "low": 4, "high": 8},
            method_name=_METHOD,
        )
        return 0.0

    optuna.create_study(direction="maximize").optimize(objective, n_trials=1)
    assert 0.01 <= captured["learning_rate"] <= 0.3
    assert 4 <= captured["depth"] <= 8


def test_a_values_list_means_a_categorical_choice() -> None:
    optuna = _require_optuna()
    captured: dict[str, Any] = {}

    def objective(trial: Any) -> float:
        captured["mode"] = suggest_parameter(
            trial,
            "mode",
            {"values": ["a", "b"]},
            method_name=_METHOD,
        )
        captured["boosting"] = suggest_parameter(
            trial,
            "boosting",
            {"type": "categorical", "values": ["gbdt", "goss"]},
            method_name=_METHOD,
        )
        return 0.0

    optuna.create_study(direction="maximize").optimize(objective, n_trials=1)
    assert captured["mode"] in {"a", "b"}
    assert captured["boosting"] in {"gbdt", "goss"}


def test_values_combined_with_a_numeric_type_is_rejected() -> None:
    # Optuna samples an explicit set only through suggest_categorical, so the
    # two keys cannot both be honoured; refusing beats ignoring one of them.
    with pytest.raises(ExecutionError, match="combines 'values' with type"):
        suggest_parameter(
            object(),
            "max_depth",
            {"type": "int", "values": [3, 5, 7]},
            method_name=_METHOD,
        )


def test_categorical_without_values_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="non-empty 'values' list"):
        suggest_parameter(object(), "mode", {"type": "categorical"}, method_name=_METHOD)


def test_missing_bounds_are_reported() -> None:
    with pytest.raises(ExecutionError, match="must define 'min' and 'max'"):
        suggest_parameter(object(), "depth", {"type": "int"}, method_name=_METHOD)


def test_unsupported_type_is_reported() -> None:
    with pytest.raises(ExecutionError, match="unsupported type"):
        suggest_parameter(
            object(),
            "depth",
            {"type": "boolean", "min": 0, "max": 1},
            method_name=_METHOD,
        )
    assert set(PARAMETER_TYPES) == {"int", "float", "categorical"}


# --- resolve_optuna_settings -----------------------------------------------


def test_settings_fall_back_to_the_per_method_defaults() -> None:
    settings = resolve_optuna_settings({}, method_name=_METHOD, n_trials=15)

    assert settings == {
        "enabled": True,
        "n_trials": 15,
        "n_startup_trials": 10,
        "sampler": "TPE",
        "timeout": None,
    }


def test_settings_are_read_and_normalized() -> None:
    settings = resolve_optuna_settings(
        {"n_trials": 25, "n_startup_trials": 4, "sampler": "random", "timeout": 120},
        method_name=_METHOD,
    )

    assert settings["enabled"] is True
    assert settings["n_trials"] == 25
    assert settings["n_startup_trials"] == 4
    assert settings["sampler"] == "RANDOM"
    assert settings["timeout"] == 120


def test_enabled_flag_is_read_from_yaml() -> None:
    settings = resolve_optuna_settings(
        {"enabled": False},
        method_name=_METHOD,
    )

    assert settings["enabled"] is False


def test_non_boolean_enabled_flag_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="enabled must be boolean"):
        resolve_optuna_settings({"enabled": "yes"}, method_name=_METHOD)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"n_trials": 0}, "n_trials"),
        ({"n_startup_trials": 0}, "n_startup_trials"),
        ({"timeout": 0}, "timeout"),
        ({"sampler": "BAYES"}, "sampler"),
    ],
)
def test_invalid_settings_are_rejected(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ExecutionError, match=message):
        resolve_optuna_settings(payload, method_name=_METHOD)


def test_a_non_mapping_settings_block_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="must be a mapping"):
        resolve_optuna_settings([1, 2], method_name=_METHOD)


# --- resolve_tuning_space --------------------------------------------------


def test_disabled_tuning_returns_only_scalars() -> None:
    fixed, space = resolve_tuning_space(
        {"depth": {"type": "int", "min": 4, "max": 8}, "iterations": 100},
        defaults=_DEFAULTS,
        enabled=False,
        method_name=_METHOD,
    )

    assert fixed == {"iterations": 100}
    assert space == {}


def test_no_mappings_use_the_fallback_minus_pinned_scalars() -> None:
    fixed, space = resolve_tuning_space(
        {"learning_rate": 0.05},
        defaults=_DEFAULTS,
        enabled=True,
        method_name=_METHOD,
    )

    assert fixed == {"learning_rate": 0.05}
    assert "learning_rate" not in space
    assert space["depth"] == _DEFAULTS["depth"]


def test_any_yaml_mapping_replaces_the_fallback_entirely() -> None:
    fixed, space = resolve_tuning_space(
        {
            "depth": {"type": "int", "min": 4, "max": 6},
            "iterations": 3000,
        },
        defaults=_DEFAULTS,
        enabled=True,
        method_name=_METHOD,
    )

    assert fixed == {"iterations": 3000}
    assert space == {"depth": {"type": "int", "min": 4, "max": 6}}
    assert "learning_rate" not in space


# --- build_sampler ---------------------------------------------------------


def test_tpe_receives_the_startup_trials_and_the_seed() -> None:
    optuna = _require_optuna()
    sampler = build_sampler(
        optuna,
        sampler_name="TPE",
        search_space={},
        seed=17,
        n_startup_trials=5,
        method_name=_METHOD,
    )

    assert type(sampler).__name__ == "TPESampler"
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
    assert 0.0 <= study.best_params["x"] <= 1.0


def test_grid_is_built_from_the_configured_values() -> None:
    optuna = _require_optuna()
    sampler = build_sampler(
        optuna,
        sampler_name="GRID",
        search_space={"max_depth": {"values": [3, 5]}},
        seed=17,
        n_startup_trials=1,
        method_name=_METHOD,
    )

    assert type(sampler).__name__ == "GridSampler"
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: float(trial.suggest_categorical("max_depth", [3, 5])), n_trials=2)
    assert study.best_params["max_depth"] in {3, 5}


def test_grid_requires_every_parameter_to_be_finite() -> None:
    optuna = _require_optuna()
    with pytest.raises(ExecutionError, match="non-empty 'values' list"):
        build_sampler(
            optuna,
            sampler_name="GRID",
            search_space={"learning_rate": {"type": "float", "min": 0.01, "max": 0.1}},
            seed=17,
            n_startup_trials=1,
            method_name=_METHOD,
        )


def test_unknown_sampler_names_are_rejected() -> None:
    optuna = _require_optuna()
    with pytest.raises(ExecutionError, match="sampler must be one of"):
        build_sampler(
            optuna,
            sampler_name="BAYES",
            search_space={},
            seed=17,
            n_startup_trials=1,
            method_name=_METHOD,
        )
    assert set(SAMPLERS) == {"TPE", "RANDOM", "GRID"}
