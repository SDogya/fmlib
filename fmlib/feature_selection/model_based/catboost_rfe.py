"""CatBoost recursive feature elimination with optional Optuna tuning.

The selector materializes a bounded sample of the train split; the module-level
helpers below operate on that pandas frame only. They never touch Spark, never
read files, and never look at the validation or test splits.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext, resolve_step_seed
from fmlib.feature_selection.config import ModelConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.default_model_param_spaces import (
    CATBOOST_RFE_SEARCH_SPACE,
)
from fmlib.feature_selection.utils.lama_boost_defaults import apply_boost_heuristics
from fmlib.feature_selection.utils.local_data import prepare_mixed_frame, root_cause
from fmlib.feature_selection.utils.optuna_space import (
    build_sampler,
    resolve_optuna_settings,
    resolve_tuning_space,
    suggest_parameter,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ROWS = 500_000
_DEFAULT_EVAL_MONTHS = 1
# Elimination steps, i.e. how many points the reported loss curve carries.
# Each step retrains CatBoost, so this is also a direct time multiplier.
_DEFAULT_RFE_STEPS = 10


def constant_drop_targets(
    n_features: int,
    max_features: int,
    drop_per_step: int,
) -> list[int]:
    """Return ``num_features_to_select`` after each constant-drop round.

    The last round may drop fewer than ``drop_per_step`` so the count never
    goes below ``max_features``.
    """
    if drop_per_step < 1:
        msg = "feature_drop_per_step must be >= 1."
        raise ValueError(msg)
    remaining = n_features
    targets: list[int] = []
    while remaining > max_features:
        drop_now = min(drop_per_step, remaining - max_features)
        remaining -= drop_now
        targets.append(remaining)
    return targets


ALGORITHMS = frozenset(
    {
        "RecursiveByLossFunctionChange",
        "RecursiveByShapValues",
        "RecursiveByPredictionValuesChange",
    },
)

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "verbose": False,
    "allow_writing_files": False,
}


class CatBoostRfeSelector:
    """Select features with CatBoost ``select_features`` on an out-of-time split.

    The train split is materialized as a bounded stratified sample, then divided
    by ``FeatureSchema.time``: the latest ``eval_months`` periods become the eval
    set used for early stopping and elimination importances, everything earlier
    is used for fitting. External ``valid`` and ``test`` splits are never read,
    so they stay usable as an unbiased check of the selected feature set.

    Optuna runs when ``params.optuna_params.enabled`` is true (the default).
    If ``params.parameters`` contains search-space mappings
    (``{"type": "int", "min": 4, "max": 8}``), those mappings are the entire
    grid. If it contains only scalars, the fallback in
    ``CATBOOST_RFE_SEARCH_SPACE`` is used. ``enabled: false`` skips Optuna and
    passes scalars to CatBoost unchanged. ``learning_rate`` and
    ``early_stopping_rounds`` are always taken from the LightAutoML row-count
    table after the out-of-time fit part is known; Optuna does not sample them. Both categorical and continuous
    candidates are evaluated — categorical ones are handed to CatBoost as
    ``cat_features``.

    Args:
        config: Model-based stage settings.
    """

    method_name = "catboost_rfe"
    stage_name = "model"

    def __init__(self: CatBoostRfeSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: CatBoostRfeSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Run recursive elimination and return keep/drop decisions.

        Args:
            context: Shared stage context with datasets, schema, config and seed.
            candidates: Features still under consideration.

        Returns:
            Decisions for every evaluated candidate. Empty when the stage is a
            no-op, for example when candidates already fit the target count.

        Raises:
            BackendError: When an optional ML dependency is unavailable.
            ExecutionError: When the input, split, or model execution is invalid.
        """
        features = list(candidates)
        if not features:
            return []

        if context.schema.task_type != "binary_classification":
            msg = (
                "catboost_rfe: only task_type='binary_classification' is supported; "
                f"got {context.schema.task_type!r}."
            )
            raise ExecutionError(msg)

        target_col = context.schema.target
        if not target_col:
            msg = "catboost_rfe: FeatureSchema.target is required."
            raise ExecutionError(msg)

        time_col = context.schema.time
        if not time_col:
            msg = (
                "catboost_rfe: FeatureSchema.time is required for the out-of-time split. "
                "Set time to the period column (for example 'month_part')."
            )
            raise ExecutionError(msg)

        train = context.datasets.get("train")
        if train is None:
            msg = "catboost_rfe: context.datasets must contain a 'train' split."
            raise ExecutionError(msg)

        options = self._resolve_options(context)
        target_count = options["num_features_to_select"]
        if len(features) <= target_count:
            logger.info(
                "%s: %d candidates already fit num_features_to_select=%d, stage is a no-op",
                self.method_name,
                len(features),
                target_count,
            )
            context.scores[self.method_name] = {
                "skipped": True,
                "reason": "candidates_below_target",
                "candidate_count": len(features),
                "num_features_to_select": target_count,
            }
            return []

        categorical = [feature for feature in features if feature in set(context.schema.categorical)]

        try:
            local = prepare_mixed_frame(
                train,
                target_col=target_col,
                feature_cols=features,
                categorical_cols=categorical,
                extra_cols=(time_col,),
                max_rows=options["max_rows"],
                sample_fraction=options["sample_fraction"],
                seed=options["seed"],
                method_name=self.method_name,
            )
            details = run_catboost_rfe(
                local,
                feature_cols=features,
                categorical_cols=categorical,
                target_col=target_col,
                time_col=time_col,
                eval_months=options["eval_months"],
                parameters=options["parameters"],
                optuna_params=options["optuna_params"],
                feature_selection_params=options["feature_selection_params"],
                num_features_to_select=target_count,
                seed=options["seed"],
                method_name=self.method_name,
            )
        except (BackendError, ExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures
            msg = f"catboost_rfe: feature selection failed. Root cause: {root_cause(exc)}."
            raise ExecutionError(msg) from exc

        selected = set(details["selected_features"])
        elimination_rank = {name: index + 1 for index, name in enumerate(details["eliminated_features"])}

        decisions = [
            FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="catboost_rfe_selected" if feature in selected else "catboost_rfe_eliminated",
                value=None if feature in selected else float(elimination_rank.get(feature, 0)),
                threshold=float(target_count),
                keep=feature in selected,
            )
            for feature in features
        ]

        context.scores[self.method_name] = {
            "selected_features": list(details["selected_features"]),
            "elimination_order": list(details["eliminated_features"]),
            "best_params": details["best_params"],
            "best_metric": details["best_metric"],
            "optuna_trials": details["optuna_trials"],
            "algorithm": details["algorithm"],
            "num_features_to_select": details["num_features_to_select"],
            # CatBoost measures the eval loss after every elimination step.
            # Paired with elimination_order it shows where the loss starts
            # rising, so num_features_to_select can be picked from the curve
            # instead of guessed up front. n_candidates converts the curve's
            # "features removed" axis into "features left".
            "loss_graph": details["loss_graph"],
            "steps": details["steps"],
            "elimination_mode": details["elimination_mode"],
            "n_candidates": len(features),
            "eval_strategy": "out_of_time",
            "eval_periods": details["eval_periods"],
            "fit_rows": details["fit_rows"],
            "eval_rows": details["eval_rows"],
            "categorical_evaluated": categorical,
        }
        if details.get("feature_drop_per_step") is not None:
            context.scores[self.method_name]["feature_drop_per_step"] = details[
                "feature_drop_per_step"
            ]
        logger.info(
            "%s: evaluated %d features, kept %d",
            self.method_name,
            len(features),
            len(selected),
        )
        return decisions

    def _resolve_options(self: CatBoostRfeSelector, context: StageContext) -> dict[str, Any]:
        """Resolve and validate method options.

        Args:
            context: Shared stage context.

        Returns:
            Normalized option dictionary.

        Raises:
            ExecutionError: When an option is missing or has an invalid value.
        """
        params = self.config.params
        parameters = params.get("parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            msg = (
                "catboost_rfe: model.params.parameters is required. Provide CatBoost parameters as scalars "
                "(used as-is) and/or as search spaces such as {'type': 'int', 'min': 4, 'max': 8} (tuned by Optuna)."
            )
            raise ExecutionError(msg)

        optuna_params = params.get("optuna_params", {})
        if not isinstance(optuna_params, Mapping):
            msg = "catboost_rfe: model.params.optuna_params must be a mapping."
            raise ExecutionError(msg)

        feature_selection_params = params.get("feature_selection_params", {})
        if not isinstance(feature_selection_params, Mapping):
            msg = "catboost_rfe: model.params.feature_selection_params must be a mapping."
            raise ExecutionError(msg)

        target_count = self.config.selection.max_features
        if target_count is None:
            msg = (
                "catboost_rfe: model.selection.max_features is required; it sets "
                "num_features_to_select for recursive elimination."
            )
            raise ExecutionError(msg)

        try:
            optuna_settings = resolve_optuna_settings(
                optuna_params,
                method_name=self.method_name,
            )
            fixed, search_space = resolve_tuning_space(
                parameters,
                defaults=CATBOOST_RFE_SEARCH_SPACE,
                enabled=optuna_settings["enabled"],
                method_name=self.method_name,
            )
            options: dict[str, Any] = {
                "eval_months": int(params.get("eval_months", _DEFAULT_EVAL_MONTHS)),
                "max_rows": min(
                    int(params.get("max_rows", _DEFAULT_MAX_ROWS)),
                    int(context.config.execution.max_local_rows),
                ),
                "sample_fraction": params.get("sample_fraction"),
                "num_features_to_select": int(target_count),
                "parameters": dict(parameters),
                "fixed_params": fixed,
                "search_space": search_space,
                "optuna_enabled": optuna_settings["enabled"],
                "optuna_params": dict(optuna_params),
                "feature_selection_params": dict(feature_selection_params),
                "seed": resolve_step_seed(params, context),
            }
            if options["sample_fraction"] is not None:
                options["sample_fraction"] = float(options["sample_fraction"])
        except (TypeError, ValueError) as exc:
            msg = f"catboost_rfe: invalid numeric model parameter. Root cause: {exc}."
            raise ExecutionError(msg) from exc

        for name in ("eval_months", "max_rows", "num_features_to_select"):
            if options[name] < 1:
                msg = f"catboost_rfe: {name} must be at least 1."
                raise ExecutionError(msg)
        sample_fraction = options["sample_fraction"]
        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            msg = "catboost_rfe: sample_fraction must be in (0, 1]."
            raise ExecutionError(msg)
        return options


def split_out_of_time(
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    eval_months: int,
    method_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split a frame into fit and eval parts by the latest periods of the time column.

    The most recent ``eval_months`` distinct values of ``time_col`` form the eval
    part; everything earlier forms the fit part. This keeps early stopping and
    RFE importances out-of-time without consuming the external validation split.

    Args:
        frame: Local frame containing ``time_col`` and ``target_col``.
        time_col: Column holding the period identifier.
        target_col: Target column, checked for class presence in both parts.
        eval_months: Number of latest periods reserved for evaluation.
        method_name: Selector name used in error messages.

    Returns:
        Tuple of ``(fit_frame, eval_frame, eval_periods)``.

    Raises:
        ExecutionError: When the split is impossible or degenerate.
    """
    if time_col not in frame.columns:
        msg = f"{method_name}: time column {time_col!r} is missing from the materialized sample."
        raise ExecutionError(msg)
    if frame[time_col].isna().any():
        msg = f"{method_name}: time column {time_col!r} contains missing values; out-of-time split is not possible."
        raise ExecutionError(msg)

    try:
        periods = sorted(frame[time_col].unique())
    except TypeError as exc:
        msg = f"{method_name}: time column {time_col!r} holds values that cannot be ordered: {exc}."
        raise ExecutionError(msg) from exc

    if len(periods) <= eval_months:
        msg = (
            f"{method_name}: out-of-time split needs more than eval_months={eval_months} distinct periods "
            f"in {time_col!r}, found {len(periods)}. Widen the training window or lower eval_months."
        )
        raise ExecutionError(msg)

    eval_periods = periods[-eval_months:]
    eval_mask = frame[time_col].isin(eval_periods)
    fit_frame = frame.loc[~eval_mask]
    eval_frame = frame.loc[eval_mask]

    for name, part in (("fit", fit_frame), ("eval", eval_frame)):
        if part.empty:
            msg = f"{method_name}: the {name} part of the out-of-time split is empty."
            raise ExecutionError(msg)
        classes = part[target_col].unique()
        if len(classes) < 2:
            msg = (
                f"{method_name}: the {name} part of the out-of-time split contains a single target class. "
                "Adjust eval_months or the sampling bounds."
            )
            raise ExecutionError(msg)

    return fit_frame, eval_frame, [str(period) for period in eval_periods]


def _load_backends(*, require_optuna: bool = True) -> tuple[Any, Any, Any]:
    """Import CatBoost, Optuna and scikit-learn metrics lazily."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        msg = "catboost_rfe: CatBoost is required. Install the catboost optional dependency."
        raise BackendError(msg) from exc
    optuna_module: Any = None
    roc_auc_score: Any = None
    if require_optuna:
        try:
            import optuna as optuna_module
        except ImportError as exc:
            msg = "catboost_rfe: Optuna is required. Install the optuna optional dependency."
            raise BackendError(msg) from exc
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:
            msg = "catboost_rfe: scikit-learn is required. Install the scikit-learn optional dependency."
            raise BackendError(msg) from exc
    return (CatBoostClassifier, Pool), optuna_module, roc_auc_score


def _finalize_parameters(parameters: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Apply library defaults and force the reproducibility seed."""
    finalized = dict(_DEFAULT_PARAMETERS)
    finalized.update(parameters)
    finalized["random_seed"] = seed
    return finalized


def tune_parameters(
    *,
    fit_pool: Any,
    eval_pool: Any,
    eval_labels: Any,
    fixed: Mapping[str, Any],
    search_space: Mapping[str, Mapping[str, Any]],
    optuna_params: Mapping[str, Any],
    seed: int,
    method_name: str,
    backends: tuple[Any, Any, Any],
) -> tuple[dict[str, Any], float, int]:
    """Tune CatBoost parameters with Optuna on the out-of-time eval part.

    Args:
        fit_pool: CatBoost ``Pool`` for training.
        eval_pool: CatBoost ``Pool`` used for early stopping and scoring.
        eval_labels: Ground-truth labels aligned with ``eval_pool``.
        fixed: Scalar parameters passed through unchanged.
        search_space: Parameter specifications tuned by Optuna.
        optuna_params: ``n_trials``, ``n_startup_trials``, ``sampler``, ``timeout``.
        seed: Deterministic seed for the sampler and CatBoost.
        method_name: Selector name used in error messages.
        backends: Result of :func:`_load_backends`.

    Returns:
        Tuple of ``(best_params, best_metric, completed_trials)``.

    Raises:
        ExecutionError: When tuning fails or produces no usable trial.
    """
    (catboost_classifier, _pool), optuna, roc_auc_score = backends

    settings = resolve_optuna_settings(optuna_params, method_name=method_name)
    sampler = build_sampler(
        optuna,
        sampler_name=settings["sampler"],
        search_space=search_space,
        seed=seed,
        n_startup_trials=settings["n_startup_trials"],
        method_name=method_name,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        suggested = {
            name: suggest_parameter(trial, name, specification, method_name=method_name)
            for name, specification in search_space.items()
        }
        parameters = _finalize_parameters({**fixed, **suggested}, seed=seed)
        model = catboost_classifier(**parameters)
        model.fit(fit_pool, eval_set=eval_pool)
        predictions = model.predict_proba(eval_pool)[:, 1]
        return float(roc_auc_score(eval_labels, predictions))

    try:
        study.optimize(
            objective,
            n_trials=settings["n_trials"],
            timeout=settings["timeout"],
            n_jobs=1,
        )
    except ExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize third-party tuning failures
        msg = f"catboost_rfe: Optuna tuning failed: {exc}"
        raise ExecutionError(msg) from exc

    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        msg = "catboost_rfe: Optuna finished without a completed trial. Increase n_trials or the timeout."
        raise ExecutionError(msg)

    best_params = _finalize_parameters({**fixed, **study.best_params}, seed=seed)
    return best_params, float(study.best_value), len(completed)


def run_catboost_rfe(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    time_col: str,
    eval_months: int,
    parameters: Mapping[str, Any],
    optuna_params: Mapping[str, Any],
    feature_selection_params: Mapping[str, Any],
    num_features_to_select: int,
    seed: int,
    method_name: str = "catboost_rfe",
) -> dict[str, Any]:
    """Tune parameters when requested, then run CatBoost recursive elimination.

    Args:
        frame: Bounded local sample with features, target and the time column.
        feature_cols: Candidate features evaluated by the selector.
        categorical_cols: Subset of ``feature_cols`` passed as CatBoost ``cat_features``.
        target_col: Target column name.
        time_col: Column used for the out-of-time split.
        eval_months: Number of latest periods reserved for evaluation.
        parameters: Polymorphic CatBoost parameter block (scalars and/or specs).
        optuna_params: Optuna settings, used when a search space is resolved.
        feature_selection_params: ``algorithm``, ``steps`` and other ``select_features`` options.
        num_features_to_select: Target feature count for elimination.
        seed: Root reproducibility seed.
        method_name: Selector name used in error messages.

    Returns:
        Dictionary with selected/eliminated features, tuning details and split sizes.

    Raises:
        BackendError: When CatBoost, Optuna or scikit-learn is unavailable.
        ExecutionError: When the split, tuning or elimination fails.
    """
    backends = _load_backends(require_optuna=False)
    (catboost_classifier, pool_class), _optuna, _roc_auc_score = backends

    fit_frame, eval_frame, eval_periods = split_out_of_time(
        frame,
        time_col=time_col,
        target_col=target_col,
        eval_months=eval_months,
        method_name=method_name,
    )

    features = list(feature_cols)
    categorical = [column for column in features if column in set(categorical_cols)]
    fit_pool = pool_class(
        fit_frame.loc[:, features],
        fit_frame[target_col],
        cat_features=categorical,
    )
    eval_pool = pool_class(
        eval_frame.loc[:, features],
        eval_frame[target_col],
        cat_features=categorical,
    )

    settings = resolve_optuna_settings(optuna_params, method_name=method_name)
    fixed, search_space = resolve_tuning_space(
        parameters,
        defaults=CATBOOST_RFE_SEARCH_SPACE,
        enabled=settings["enabled"],
        method_name=method_name,
    )
    fixed, search_space = apply_boost_heuristics(
        fixed,
        search_space,
        n_rows=len(fit_frame),
        library="catboost",
    )
    if search_space:
        backends = _load_backends(require_optuna=True)
        best_params, best_metric, completed_trials = tune_parameters(
            fit_pool=fit_pool,
            eval_pool=eval_pool,
            eval_labels=eval_frame[target_col].to_numpy(),
            fixed=fixed,
            search_space=search_space,
            optuna_params=optuna_params,
            seed=seed,
            method_name=method_name,
            backends=backends,
        )
        logger.info(
            "%s: Optuna finished %d trials, best eval ROC-AUC %.5f",
            method_name,
            completed_trials,
            best_metric,
        )
    else:
        best_params = _finalize_parameters(fixed, seed=seed)
        best_metric = None
        completed_trials = 0
        logger.info("%s: Optuna disabled or search space empty, skipping tuning", method_name)

    selection_params = dict(feature_selection_params)
    algorithm = str(selection_params.pop("algorithm", "RecursiveByLossFunctionChange"))
    if algorithm not in ALGORITHMS:
        msg = f"{method_name}: unsupported algorithm={algorithm!r}. Expected one of: {sorted(ALGORITHMS)}."
        raise ExecutionError(msg)
    selection_params.pop("num_features_to_select", None)
    selection_params.pop("train_final_model", None)
    mode, schedule_value = _pop_elimination_schedule(selection_params, method_name=method_name)

    if mode == "feature_drop_per_step":
        selected, eliminated, loss_graph, n_rounds = _eliminate_constant_drop(
            catboost_classifier=catboost_classifier,
            pool_class=pool_class,
            fit_frame=fit_frame,
            eval_frame=eval_frame,
            features=features,
            categorical_cols=categorical,
            target_col=target_col,
            best_params=best_params,
            extra_selection_params=selection_params,
            algorithm=algorithm,
            drop_per_step=schedule_value,
            num_features_to_select=num_features_to_select,
            method_name=method_name,
        )
        payload_steps = n_rounds
    else:
        model = catboost_classifier(**best_params)
        try:
            summary = model.select_features(
                fit_pool,
                eval_set=eval_pool,
                features_for_select=features,
                num_features_to_select=num_features_to_select,
                algorithm=algorithm,
                steps=schedule_value,
                train_final_model=False,
                **selection_params,
            )
        except Exception as exc:  # noqa: BLE001 - normalize CatBoost failures
            msg = f"{method_name}: CatBoost recursive elimination failed: {exc}"
            raise ExecutionError(msg) from exc
        selected = _names_from_summary(summary, features, "selected_features_names", "selected_features")
        eliminated = _names_from_summary(
            summary,
            features,
            "eliminated_features_names",
            "eliminated_features",
        )
        loss_graph = summary.get("loss_graph")
        payload_steps = schedule_value

    result: dict[str, Any] = {
        "selected_features": selected,
        "eliminated_features": eliminated,
        "best_params": best_params,
        "best_metric": best_metric,
        "optuna_trials": completed_trials,
        "algorithm": algorithm,
        "num_features_to_select": num_features_to_select,
        "steps": payload_steps,
        "elimination_mode": mode,
        "eval_periods": eval_periods,
        "fit_rows": len(fit_frame),
        "eval_rows": len(eval_frame),
        "loss_graph": loss_graph,
    }
    if mode == "feature_drop_per_step":
        result["feature_drop_per_step"] = schedule_value
    return result


def _pop_elimination_schedule(
    selection_params: dict[str, Any],
    *,
    method_name: str,
) -> tuple[str, int]:
    """Return ``(mode, steps_or_drop)`` and strip those keys from ``selection_params``."""
    has_steps = "steps" in selection_params
    has_drop = "feature_drop_per_step" in selection_params
    if has_steps and has_drop:
        msg = (
            f"{method_name}: feature_selection_params cannot set both "
            "'steps' and 'feature_drop_per_step'."
        )
        raise ExecutionError(msg)
    if has_drop:
        raw = selection_params.pop("feature_drop_per_step")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            msg = f"{method_name}: feature_selection_params.feature_drop_per_step must be >= 1."
            raise ExecutionError(msg)
        return "feature_drop_per_step", raw
    raw_steps = selection_params.pop("steps", _DEFAULT_RFE_STEPS)
    steps = int(raw_steps)
    if steps < 1:
        msg = f"{method_name}: feature_selection_params.steps must be >= 1."
        raise ExecutionError(msg)
    return "steps", steps


def _names_from_summary(
    summary: Mapping[str, Any],
    remaining: Sequence[str],
    names_key: str,
    index_key: str,
) -> list[str]:
    """Prefer name lists from CatBoost; fall back to indices into ``remaining``."""
    names = summary.get(names_key)
    if names:
        return [str(name) for name in names]
    indices = summary.get(index_key) or []
    resolved: list[str] = []
    for index in indices:
        position = int(index)
        if 0 <= position < len(remaining):
            resolved.append(remaining[position])
    return resolved


def _eliminate_constant_drop(
    *,
    catboost_classifier: Any,
    pool_class: Any,
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    best_params: Mapping[str, Any],
    extra_selection_params: Mapping[str, Any],
    algorithm: str,
    drop_per_step: int,
    num_features_to_select: int,
    method_name: str,
) -> tuple[list[str], list[str], Any, int]:
    """Drop a fixed count per ``select_features(..., steps=1)`` round."""
    remaining = list(features)
    eliminated: list[str] = []
    graphs: list[Any] = []
    cumulative_removed: list[int] = []
    try:
        targets = constant_drop_targets(len(remaining), num_features_to_select, drop_per_step)
    except ValueError as exc:
        msg = f"{method_name}: {exc}"
        raise ExecutionError(msg) from exc

    categorical_set = set(categorical_cols)
    for target in targets:
        cats = [name for name in remaining if name in categorical_set]
        fit_pool = pool_class(
            fit_frame.loc[:, remaining],
            fit_frame[target_col],
            cat_features=cats,
        )
        eval_pool = pool_class(
            eval_frame.loc[:, remaining],
            eval_frame[target_col],
            cat_features=cats,
        )
        model = catboost_classifier(**best_params)
        try:
            summary = model.select_features(
                fit_pool,
                eval_set=eval_pool,
                features_for_select=remaining,
                num_features_to_select=target,
                algorithm=algorithm,
                steps=1,
                train_final_model=False,
                **extra_selection_params,
            )
        except Exception as exc:
            msg = f"{method_name}: CatBoost recursive elimination failed: {exc}"
            raise ExecutionError(msg) from exc
        dropped = _names_from_summary(
            summary,
            remaining,
            "eliminated_features_names",
            "eliminated_features",
        )
        kept = _names_from_summary(
            summary,
            remaining,
            "selected_features_names",
            "selected_features",
        )
        if not kept:
            kept = [name for name in remaining if name not in set(dropped)]
        eliminated.extend(dropped)
        remaining = kept
        graphs.append(summary.get("loss_graph"))
        cumulative_removed.append(len(eliminated))

    return remaining, eliminated, _stitch_constant_drop_loss(graphs, cumulative_removed), len(targets)


def _stitch_constant_drop_loss(
    graphs: Sequence[Any],
    cumulative_removed: Sequence[int],
) -> dict[str, Any] | None:
    """Rebuild a loss_graph whose x axis is features removed from the original set."""
    if not graphs:
        return None
    removed_counts: list[int] = [0]
    loss_values: list[Any] = []
    first = graphs[0] if isinstance(graphs[0], Mapping) else None
    if first and first.get("loss_values"):
        loss_values.append(first["loss_values"][0])
    else:
        loss_values.append(None)
    for graph, total_removed in zip(graphs, cumulative_removed, strict=False):
        if not isinstance(graph, Mapping) or not graph.get("loss_values"):
            continue
        values = list(graph["loss_values"])
        removed_counts.append(int(total_removed))
        loss_values.append(values[-1])
    return {
        "removed_features_count": removed_counts,
        "loss_values": loss_values,
        "main_indices": list(range(len(loss_values))),
    }
