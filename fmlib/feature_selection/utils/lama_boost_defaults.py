"""LightAutoML-style learning rate, tree cap and early stopping.

Copied from LightAutoML ``boost_lgbm.py`` / ``boost_cb.py`` binary tables.
Optuna does not sample these keys: selectors inject them after ``n_rows`` of
the materialized train is known. Real tree count comes from early stopping;
``n_estimators`` / ``iterations`` is only a ceiling.
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


def boost_fixed_params(
    n_rows: int,
    *,
    library: BoostLibrary,
) -> dict[str, Any]:
    """Return table ``learning_rate``, tree cap and early-stopping patience.

    Args:
        n_rows: Row count of the materialized train actually used for fit
            (CatBoost RFE out-of-time fit part; LightGBM/Boruta local sample).
        library: ``lightgbm`` or ``catboost``.

    Returns:
        Parameter dict to merge into the selector's fixed block.

    Raises:
        ValueError: When ``library`` is not a supported boosting backend.
    """
    rows = max(0, int(n_rows))
    if library == "lightgbm":
        return _lightgbm_table(rows)
    if library == "catboost":
        return _catboost_table(rows)
    msg = f"unsupported boost library {library!r}"
    raise ValueError(msg)


def apply_boost_heuristics(
    fixed: Mapping[str, Any] | None,
    search_space: Mapping[str, Mapping[str, Any]] | None,
    *,
    n_rows: int,
    library: BoostLibrary,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Force table lr/patience into ``fixed`` and strip them from Optuna.

    Tree cap is injected only when YAML did not already pin
    ``iterations`` / ``n_estimators`` / ``num_trees`` (or an alias) as a scalar.
    A mapping of the tree cap stays legal and is left in ``search_space``.

    Args:
        fixed: Scalar parameters from ``resolve_tuning_space``.
        search_space: Optuna specs from ``resolve_tuning_space``.
        n_rows: Materialized train size for the table lookup.
        library: ``lightgbm`` or ``catboost``.

    Returns:
        Tuple of ``(fixed, search_space)`` after the LightAutoML overlay.
    """
    table = boost_fixed_params(n_rows, library=library)
    new_space = {
        str(name): dict(spec)
        for name, spec in dict(search_space or {}).items()
        if str(name) not in _STRIP_FROM_SPACE
    }
    new_fixed = dict(fixed or {})
    for key in _STRIP_FROM_SPACE:
        new_fixed.pop(key, None)
    new_fixed["learning_rate"] = table["learning_rate"]
    new_fixed["early_stopping_rounds"] = table["early_stopping_rounds"]
    if library == "catboost":
        new_fixed["use_best_model"] = True
        cap_keys = _CB_TREE_CAP_KEYS
        cap_name = "iterations"
    else:
        cap_keys = _LGBM_TREE_CAP_KEYS
        cap_name = "n_estimators"
    if not any(key in new_fixed for key in cap_keys):
        new_fixed[cap_name] = table[cap_name]
    return new_fixed, new_space


def split_lgbm_early_stopping(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None]:
    """Drop patience keys that do not belong on the LightGBM constructor."""
    params = dict(parameters)
    raw = params.pop("early_stopping_rounds", None)
    params.pop("od_wait", None)
    if raw is None:
        return params, None
    return params, int(raw)


def fit_lgbm_with_early_stopping(
    model: Any,
    features: Any,
    target: Any,
    *,
    eval_set: Any,
    early_stopping_rounds: int | None,
) -> Any:
    """Fit a LightGBM estimator with patience when an eval set is present.

    Uses ``callbacks=[early_stopping(...)]`` on LightGBM 4+, and falls back to
    the ``early_stopping_rounds=`` fit argument on older releases.

    Args:
        model: Constructed ``LGBMClassifier`` (patience already removed).
        features: Train feature matrix.
        target: Train labels.
        eval_set: LightGBM ``eval_set`` argument, or ``None``.
        early_stopping_rounds: Table patience, or ``None`` to train to the cap.

    Returns:
        The fitted ``model``.
    """
    if early_stopping_rounds is None or eval_set is None:
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
    """Binary (non-regression) LightAutoML ``init_params_on_input``."""
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


def _catboost_table(n_rows: int) -> dict[str, Any]:
    """Binary LightAutoML CatBoost ``num_trees`` / ``learning_rate`` table."""
    if n_rows <= 6_000:
        lr, trees = 0.02, 500
    elif n_rows <= 20_000:
        lr, trees = 0.035, 5000
    elif n_rows <= 50_000:
        lr, trees = 0.03, 5000
    elif n_rows <= 60_000:
        lr, trees = 0.05, 2000
    elif n_rows <= 100_000:
        lr, trees = 0.045, 1500
    elif n_rows <= 150_000:
        lr, trees = 0.045, 3000
    elif n_rows <= 300_000:
        lr, trees = 0.045, 2000
    else:
        lr, trees = 0.05, 3000
    return {
        "learning_rate": lr,
        "iterations": trees,
        "early_stopping_rounds": 100,
        "use_best_model": True,
    }
