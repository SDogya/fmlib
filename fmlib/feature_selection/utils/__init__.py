"""Pre-statistics helpers: drop, sample, and driver-local materialization."""

from typing import TYPE_CHECKING, Any

from fmlib.feature_selection.utils.optuna_space import (
    PARAMETER_TYPES,
    SAMPLERS,
    build_sampler,
    build_search_space,
    resolve_optuna_settings,
    split_parameters,
    suggest_parameter,
)

if TYPE_CHECKING:
    from fmlib.feature_selection.utils.stage import UtilsStage

__all__ = [
    "PARAMETER_TYPES",
    "SAMPLERS",
    "UtilsStage",
    "build_sampler",
    "build_search_space",
    "resolve_optuna_settings",
    "split_parameters",
    "suggest_parameter",
]


def __getattr__(name: str) -> Any:
    if name == "UtilsStage":
        from fmlib.feature_selection.utils.stage import UtilsStage

        return UtilsStage
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
