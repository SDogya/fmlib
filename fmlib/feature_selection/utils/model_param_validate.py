"""Load-time checks for model ``params.parameters`` keys and Optuna specs.

CatBoost, LightGBM and sklearn RandomForest accept overlapping aliases
(``iterations`` / ``n_estimators``, ``eta`` / ``learning_rate``, …). Setting two
names from the same group, or a typo, used to fail only when the model was
constructed — often hours into a run. These helpers reject that while the
YAML is parsed.
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.utils.default_model_param_spaces import (
    BORUTA_LGBM_SEARCH_SPACE,
    BORUTA_RF_SEARCH_SPACE,
    CATBOOST_RFE_SEARCH_SPACE,
    LIGHTGBM_SEARCH_SPACE,
)
from fmlib.feature_selection.utils.optuna_space import validate_parameter_spec

CATBOOST_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"iterations", "n_estimators", "num_boost_round", "num_trees"}),
    frozenset({"depth", "max_depth"}),
    frozenset({"learning_rate", "eta"}),
    frozenset({"random_seed", "random_state"}),
    frozenset({"l2_leaf_reg", "reg_lambda"}),
    frozenset({"border_count", "max_bin"}),
    frozenset({"rsm", "colsample_bylevel"}),
    frozenset({"min_data_in_leaf", "min_child_samples"}),
    frozenset({"thread_count", "n_jobs"}),
)

LIGHTGBM_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"n_estimators", "num_iterations", "num_boost_round", "num_tree", "nrounds"}),
    frozenset({"learning_rate", "eta", "shrinkage_rate"}),
    frozenset({"subsample", "bagging_fraction"}),
    frozenset({"colsample_bytree", "feature_fraction"}),
    frozenset({"min_child_samples", "min_data_in_leaf", "min_data"}),
    frozenset({"min_child_weight", "min_sum_hessian_in_leaf"}),
    frozenset({"reg_alpha", "lambda_l1"}),
    frozenset({"reg_lambda", "lambda_l2"}),
    frozenset({"min_split_gain", "min_gain_to_split"}),
    frozenset({"subsample_freq", "bagging_freq"}),
    frozenset({"boosting_type", "boosting"}),
    frozenset({"verbosity", "verbose"}),
)

# Keys ``LightGbmSelector._finalize_parameters`` overwrites. Putting them in
# YAML looks applied and is silently discarded.
LIGHTGBM_FORCED_KEYS: frozenset[str] = frozenset(
    {
        "objective",
        "metric",
        "verbosity",
        "n_jobs",
        "random_state",
        "bagging_seed",
        "deterministic",
        "force_row_wise",
        "feature_fraction_seed",
        "data_random_seed",
        "extra_seed",
    },
)

_CATBOOST_FALLBACK: frozenset[str] = frozenset(CATBOOST_RFE_SEARCH_SPACE) | {
    "iterations",
    "n_estimators",
    "num_boost_round",
    "num_trees",
    "depth",
    "max_depth",
    "learning_rate",
    "eta",
    "random_seed",
    "random_state",
    "l2_leaf_reg",
    "reg_lambda",
    "min_data_in_leaf",
    "min_child_samples",
    "border_count",
    "max_bin",
    "rsm",
    "colsample_bylevel",
    "loss_function",
    "eval_metric",
    "custom_metric",
    "custom_loss",
    "early_stopping_rounds",
    "od_type",
    "od_wait",
    "od_pval",
    "use_best_model",
    "verbose",
    "logging_level",
    "metric_period",
    "allow_writing_files",
    "train_dir",
    "boosting_type",
    "bootstrap_type",
    "grow_policy",
    "leaf_estimation_method",
    "leaf_estimation_iterations",
    "max_ctr_complexity",
    "one_hot_max_size",
    "has_time",
    "task_type",
    "devices",
    "thread_count",
    "n_jobs",
    "bagging_temperature",
    "subsample",
    "random_strength",
    "class_weights",
    "auto_class_weights",
    "scale_pos_weight",
    "nan_mode",
    "ignored_features",
    "name",
    "score_function",
    "model_size_reg",
    "max_leaves",
    "sampling_frequency",
    "sampling_unit",
    "fold_len_multiplier",
}

_LIGHTGBM_FALLBACK: frozenset[str] = (
    frozenset(LIGHTGBM_SEARCH_SPACE)
    | frozenset(BORUTA_LGBM_SEARCH_SPACE)
    | LIGHTGBM_FORCED_KEYS
    | {
        "boosting_type",
        "boosting",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "eta",
        "n_estimators",
        "num_iterations",
        "num_boost_round",
        "min_child_samples",
        "min_data_in_leaf",
        "min_data",
        "min_child_weight",
        "min_sum_hessian_in_leaf",
        "early_stopping_rounds",
        "min_split_gain",
        "min_gain_to_split",
        "subsample",
        "bagging_fraction",
        "subsample_freq",
        "bagging_freq",
        "colsample_bytree",
        "feature_fraction",
        "reg_alpha",
        "lambda_l1",
        "reg_lambda",
        "lambda_l2",
        "scale_pos_weight",
        "max_bin",
        "class_weight",
        "importance_type",
        "silent",
        "verbose",
        "n_jobs",
        "random_state",
        "seed",
    }
)

_RF_FALLBACK: frozenset[str] = frozenset(BORUTA_RF_SEARCH_SPACE) | {
    "n_estimators",
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "bootstrap",
    "oob_score",
    "n_jobs",
    "random_state",
    "verbose",
    "warm_start",
    "class_weight",
    "ccp_alpha",
    "max_samples",
}


def validate_model_parameters(
    parameters: Mapping[str, Any],
    *,
    library: str,
    method_name: str,
    require_non_empty: bool = False,
    ignore_keys: frozenset[str] = frozenset(),
) -> None:
    """Reject alias clashes, forced keys, unknown names and bad Optuna specs.

    Args:
        parameters: Raw ``params.parameters`` mapping (scalars and/or specs).
        library: ``catboost``, ``lightgbm`` or ``random_forest``.
        method_name: Selector name used in error messages.
        require_non_empty: When true, an empty mapping is a config error.
        ignore_keys: Keys skipped entirely (legacy leftovers such as
            ``bootstrap_type`` on Boruta LightGBM).

    Raises:
        ConfigError: When the block cannot be handed to the model as written.
    """
    if not isinstance(parameters, Mapping):
        msg = f"{method_name}: params.parameters must be a mapping."
        raise ConfigError(msg)
    if require_non_empty and not parameters:
        msg = (
            f"{method_name}: params.parameters is required and must not be empty. "
            "Provide model parameters as scalars and/or as search spaces such as "
            "{'type': 'int', 'min': 4, 'max': 8}."
        )
        raise ConfigError(msg)

    keys: list[str] = []
    for raw_name, value in parameters.items():
        name = str(raw_name)
        if name in ignore_keys:
            continue
        keys.append(name)
        if isinstance(value, Mapping):
            validate_parameter_spec(name, value, method_name=method_name)

    if library == "lightgbm":
        _reject_forced_keys(keys, method_name=method_name)
        groups = LIGHTGBM_ALIAS_GROUPS
        groups = groups + _lightgbm_library_alias_groups()
        known = _lightgbm_known_names()
    elif library == "catboost":
        groups = CATBOOST_ALIAS_GROUPS
        known = _catboost_known_names()
    elif library == "random_forest":
        groups = ()
        known = _random_forest_known_names()
    else:
        msg = f"{method_name}: unsupported parameter library {library!r}."
        raise ConfigError(msg)

    _reject_alias_clashes(keys, groups=groups, method_name=method_name)
    _reject_unknown_keys(keys, known=known, method_name=method_name)


def validate_optuna_parameter_block(
    parameters: Mapping[str, Any],
    *,
    method_name: str,
) -> None:
    """Validate Optuna specs in a parameters mapping without library checks."""
    if not isinstance(parameters, Mapping):
        msg = f"{method_name}: params.parameters must be a mapping."
        raise ConfigError(msg)
    for raw_name, value in parameters.items():
        if isinstance(value, Mapping):
            validate_parameter_spec(str(raw_name), value, method_name=method_name)


def _reject_forced_keys(keys: list[str], *, method_name: str) -> None:
    """Reject LightGBM keys the selector always overwrites."""
    present = {key.lower(): key for key in keys}
    collisions = [
        present[forced]
        for forced in sorted(LIGHTGBM_FORCED_KEYS)
        if forced in present
    ]
    if not collisions:
        return
    listed = ", ".join(repr(name) for name in collisions)
    msg = (
        f"{method_name}: params.parameters must not set {listed}. "
        "The pipeline forces objective, metric, verbosity, n_jobs and the "
        "LightGBM RNG seeds from the step seed."
    )
    raise ConfigError(msg)


def _reject_alias_clashes(
    keys: list[str],
    *,
    groups: tuple[frozenset[str], ...],
    method_name: str,
) -> None:
    """Reject two names that the library treats as the same parameter."""
    lowered = {key.lower(): key for key in keys}
    seen_groups: set[frozenset[str]] = set()
    for group in groups:
        canonical = frozenset(name.lower() for name in group)
        if canonical in seen_groups:
            continue
        seen_groups.add(canonical)
        hits = [lowered[name] for name in canonical if name in lowered]
        if len(hits) < 2:
            continue
        listed = ", ".join(repr(name) for name in hits)
        msg = (
            f"{method_name}: params.parameters sets aliases together: {listed}. "
            "Keep exactly one name from that group."
        )
        raise ConfigError(msg)


def _reject_unknown_keys(
    keys: list[str],
    *,
    known: set[str],
    method_name: str,
) -> None:
    """Reject names the target library / allowlist does not recognise."""
    known_lower = {name.lower(): name for name in known}
    for key in keys:
        if key.lower() in known_lower:
            continue
        suggestion = difflib.get_close_matches(
            key.lower(),
            list(known_lower),
            n=3,
            cutoff=0.6,
        )
        hint = ""
        if suggestion:
            pretty = ", ".join(repr(known_lower[name]) for name in suggestion)
            hint = f" Did you mean {pretty}?"
        msg = f"{method_name}: unknown parameter {key!r}.{hint}"
        raise ConfigError(msg)


def _lightgbm_known_names() -> set[str]:
    """Union of the fallback allowlist and, when installed, library names."""
    names = set(_LIGHTGBM_FALLBACK)
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return names
    try:
        names.update(LGBMClassifier().get_params())
    except Exception:  # noqa: BLE001, S110 - constructor/get_params differ by LightGBM version
        pass
    try:
        from lightgbm.basic import _ConfigAliases
    except ImportError:
        return names
    names.update(_flatten_config_aliases(_ConfigAliases))
    return names


def _lightgbm_library_alias_groups() -> tuple[frozenset[str], ...]:
    """Alias groups advertised by LightGBM, if the extra is installed."""
    try:
        from lightgbm.basic import _ConfigAliases
    except ImportError:
        return ()
    groups: list[frozenset[str]] = []
    for aliases in _config_alias_sets(_ConfigAliases):
        if len(aliases) >= 2:
            groups.append(frozenset(aliases))
    return tuple(groups)


def _catboost_known_names() -> set[str]:
    """Union of the fallback allowlist and CatBoost names when installed."""
    names = set(_CATBOOST_FALLBACK)
    for group in CATBOOST_ALIAS_GROUPS:
        names.update(group)
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return names
    try:
        names.update(CatBoostClassifier().get_params())
    except Exception:  # noqa: BLE001, S110 - CatBoost constructor surface varies by version
        pass
    try:
        from catboost import CatBoost

        params = getattr(CatBoost, "_get_all_params", None)
        if callable(params):
            names.update(params(CatBoost()))
    except Exception:  # noqa: BLE001, S110 - private CatBoost API is optional
        pass
    return names


def _random_forest_known_names() -> set[str]:
    """sklearn RandomForest names when installed, otherwise the fallback."""
    names = set(_RF_FALLBACK)
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return names
    try:
        names.update(RandomForestClassifier().get_params())
    except Exception:  # noqa: BLE001 - sklearn get_params is version-dependent
        return names
    return names


def _flatten_config_aliases(config_aliases: Any) -> set[str]:
    """Collect every alias string LightGBM knows about."""
    names: set[str] = set()
    for aliases in _config_alias_sets(config_aliases):
        names.update(aliases)
    return names


def _config_alias_sets(config_aliases: Any) -> list[set[str]]:
    """Read LightGBM ``_ConfigAliases`` in whichever shape this version uses."""
    sets: list[set[str]] = []
    mapping = getattr(config_aliases, "aliases", None)
    if isinstance(mapping, Mapping):
        for key, value in mapping.items():
            bucket = {str(key)}
            if isinstance(value, (list, tuple, set, frozenset)):
                bucket.update(str(item) for item in value)
            elif value is not None:
                bucket.add(str(value))
            sets.append(bucket)
        return sets
    getter = getattr(config_aliases, "get", None)
    if callable(getter):
        candidates = set(_LIGHTGBM_FALLBACK)
        for name in list(candidates):
            try:
                aliases = getter(name)
            except Exception:  # noqa: BLE001 - _ConfigAliases.get is not a public contract
                continue
            if isinstance(aliases, (list, tuple, set, frozenset)):
                sets.append({str(item) for item in aliases} | {str(name)})
    return sets
