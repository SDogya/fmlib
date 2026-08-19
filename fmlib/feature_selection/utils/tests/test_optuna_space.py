"""Tests for the Optuna helpers shared by every tuning selector.

Optuna itself is an optional dependency, so the trial and the module are
doubled here: the helpers only translate configuration into calls.
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
    split_parameters,
    suggest_parameter,
)

_METHOD = "test_method"


class FakeTrial:
    """Optuna trial double recording the suggestion calls it receives."""

    def __init__(self: FakeTrial) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def suggest_int(self: FakeTrial, name: str, low: int, high: int, **kwargs: Any) -> int:
        self.calls.append((name, (low, high), kwargs))
        return low

    def suggest_float(self: FakeTrial, name: str, low: float, high: float, **kwargs: Any) -> float:
        self.calls.append((name, (low, high), kwargs))
        return low

    def suggest_categorical(self: FakeTrial, name: str, values: list) -> Any:
        self.calls.append((name, tuple(values), {}))
        return values[0]


class _FakeSampler:
    def __init__(self: _FakeSampler, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSamplers:
    TPESampler = _FakeSampler
    RandomSampler = _FakeSampler
    GridSampler = _FakeSampler


class _FakeOptuna:
    samplers = _FakeSamplers


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
    # This is how tuning is switched off: no ranges, no Optuna run.
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
    trial = FakeTrial()

    suggest_parameter(
        trial,
        "learning_rate",
        {"type": "float", "min": 0.01, "max": 0.3, "log": True},
        method_name=_METHOD,
    )
    suggest_parameter(trial, "depth", {"type": "int", "low": 4, "high": 8}, method_name=_METHOD)

    assert trial.calls[0] == ("learning_rate", (0.01, 0.3), {"log": True})
    # log is omitted for int unless explicitly requested
    assert trial.calls[1] == ("depth", (4, 8), {})


def test_a_values_list_means_a_categorical_choice() -> None:
    trial = FakeTrial()

    suggest_parameter(trial, "mode", {"values": ["a", "b"]}, method_name=_METHOD)
    suggest_parameter(
        trial,
        "boosting",
        {"type": "categorical", "values": ["gbdt", "goss"]},
        method_name=_METHOD,
    )

    assert trial.calls[0] == ("mode", ("a", "b"), {})
    assert trial.calls[1] == ("boosting", ("gbdt", "goss"), {})


def test_values_combined_with_a_numeric_type_is_rejected() -> None:
    # Optuna samples an explicit set only through suggest_categorical, so the
    # two keys cannot both be honoured; refusing beats ignoring one of them.
    with pytest.raises(ExecutionError, match="combines 'values' with type"):
        suggest_parameter(
            FakeTrial(),
            "max_depth",
            {"type": "int", "values": [3, 5, 7]},
            method_name=_METHOD,
        )


def test_categorical_without_values_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="non-empty 'values' list"):
        suggest_parameter(FakeTrial(), "mode", {"type": "categorical"}, method_name=_METHOD)


def test_missing_bounds_are_reported() -> None:
    with pytest.raises(ExecutionError, match="must define 'min' and 'max'"):
        suggest_parameter(FakeTrial(), "depth", {"type": "int"}, method_name=_METHOD)


def test_unsupported_type_is_reported() -> None:
    with pytest.raises(ExecutionError, match="unsupported type"):
        suggest_parameter(
            FakeTrial(),
            "depth",
            {"type": "boolean", "min": 0, "max": 1},
            method_name=_METHOD,
        )
    assert set(PARAMETER_TYPES) == {"int", "float", "categorical"}


# --- resolve_optuna_settings -----------------------------------------------


def test_settings_fall_back_to_the_per_method_defaults() -> None:
    settings = resolve_optuna_settings({}, method_name=_METHOD, n_trials=15)

    assert settings == {
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

    assert settings["n_trials"] == 25
    assert settings["n_startup_trials"] == 4
    assert settings["sampler"] == "RANDOM"
    assert settings["timeout"] == 120


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


# --- build_sampler ---------------------------------------------------------


def test_tpe_receives_the_startup_trials_and_the_seed() -> None:
    sampler = build_sampler(
        _FakeOptuna,
        sampler_name="TPE",
        search_space={},
        seed=17,
        n_startup_trials=5,
        method_name=_METHOD,
    )

    assert sampler.kwargs == {"seed": 17, "n_startup_trials": 5}


def test_grid_is_built_from_the_configured_values() -> None:
    sampler = build_sampler(
        _FakeOptuna,
        sampler_name="GRID",
        search_space={"max_depth": {"values": [3, 5]}},
        seed=17,
        n_startup_trials=1,
        method_name=_METHOD,
    )

    assert sampler.kwargs["search_space"] == {"max_depth": [3, 5]}


def test_grid_requires_every_parameter_to_be_finite() -> None:
    with pytest.raises(ExecutionError, match="non-empty 'values' list"):
        build_sampler(
            _FakeOptuna,
            sampler_name="GRID",
            search_space={"learning_rate": {"type": "float", "min": 0.01, "max": 0.1}},
            seed=17,
            n_startup_trials=1,
            method_name=_METHOD,
        )


def test_unknown_sampler_names_are_rejected() -> None:
    with pytest.raises(ExecutionError, match="sampler must be one of"):
        build_sampler(
            _FakeOptuna,
            sampler_name="BAYES",
            search_space={},
            seed=17,
            n_startup_trials=1,
            method_name=_METHOD,
        )
    assert set(SAMPLERS) == {"TPE", "RANDOM", "GRID"}
