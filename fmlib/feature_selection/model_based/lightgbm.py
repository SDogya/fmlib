"""LightGBM model-based selector with SHAP importance.

Outer folds run sequentially on the driver: the selector builds a bounded local
numeric matrix, optionally tunes LightGBM with Optuna, then trains one model per
fold and aggregates split and SHAP importances.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import ModelConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.local_data import prepare_numeric_frame, root_cause
from fmlib.feature_selection.utils.optuna_space import (
    build_sampler,
    build_search_space,
    resolve_optuna_settings,
    split_parameters,
    suggest_parameter,
)

logger = logging.getLogger(__name__)

try:
    import optuna
except ImportError:
    optuna = None

try:
    import shap
except ImportError:
    shap = None

OPTUNA_MODES = frozenset({"global", "per_fold"})

_DEFAULT_N_TRIALS = 20

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 100, "max": 500},
    "learning_rate": {"type": "float", "min": 0.01, "max": 0.2, "log": True},
    "max_depth": {"type": "int", "min": 3, "max": 8},
    "num_leaves": {"type": "int", "min": 8, "max": 64},
    "subsample": {"type": "float", "min": 0.6, "max": 1.0},
    "colsample_bytree": {"type": "float", "min": 0.6, "max": 1.0},
}


class LightGbmSelector:
    """Select continuous features using LightGBM and SHAP importances.

    The model algorithm follows ``shap_lgbm_spark.py``: tune a binary
    ``LGBMClassifier`` on a stratified hold-out split, execute outer folds
    sequentially on the driver, aggregate split and SHAP importances, then
    keep the cumulative-threshold intersection. ``optuna_mode="global"``
    tunes once on the driver; ``"per_fold"`` tunes independently for each
    fold using only that fold's outer-train rows.

    Spark inputs are stratified before local materialization. Already-local
    pandas inputs use equivalent bounded stratified sampling. Only current
    candidates declared in ``FeatureSchema.continuous`` are evaluated;
    categorical candidates pass through the model stage unchanged. Fold
    training never ships code or packages to Spark executors.

    The Optuna search space defaults to ``DEFAULT_SEARCH_SPACE`` and can be
    overridden per parameter through ``params.parameters``: a mapping such as
    ``{"type": "int", "min": 16, "max": 128}`` replaces the corresponding
    default, a scalar is passed to LightGBM unchanged instead of being tuned.

    Args:
        config: Model-stage settings. Method-specific ``params`` override the
            corresponding tuning and cross-validation defaults.
    """

    method_name = "lightgbm"
    stage_name = "model"

    def __init__(self: LightGbmSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: LightGbmSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Evaluate continuous candidates with LightGBM and SHAP.

        Args:
            context: Shared stage context containing train data, schema, config,
                and the reproducibility seed.
            candidates: Features still under consideration.

        Returns:
            Keep/drop decisions for evaluated continuous features. Features
            outside this selector's scope receive no decision and pass through.

        Raises:
            BackendError: When an optional ML dependency is unavailable.
            ExecutionError: When the input or model execution is invalid.
        """
        if not candidates:
            return []
        if context.schema.task_type != "binary_classification":
            msg = (
                "lightgbm: only task_type='binary_classification' is supported "
                f"by the current LGBM/SHAP algorithm; got {context.schema.task_type!r}."
            )
            raise ExecutionError(msg)

        target_col = context.schema.target
        if not target_col:
            msg = "lightgbm: FeatureSchema.target is required."
            raise ExecutionError(msg)

        train = context.datasets.get("train")
        if train is None:
            msg = "lightgbm: context.datasets must contain a 'train' split."
            raise ExecutionError(msg)

        continuous = set(context.schema.continuous)
        feature_cols = [feature for feature in candidates if feature in continuous]
        if not feature_cols:
            return []

        options = self._resolve_options(context)
        self._load_backends()

        try:
            details = self._select_robust_features(
                df=train,
                target_col=target_col,
                feature_cols=feature_cols,
                n_trials=options["n_trials"],
                n_startup_trials=options["n_startup_trials"],
                sampler=options["sampler"],
                timeout=options["timeout"],
                n_folds=options["n_folds"],
                max_rows_limit=options["max_rows"],
                sample_fraction=options["sample_fraction"],
                lgbm_threshold=options["lgbm_threshold"],
                shap_threshold=options["shap_threshold"],
                seed=context.seed,
                optuna_mode=options["optuna_mode"],
                n_jobs=options["n_jobs"],
                shap_max_rows=options["shap_max_rows"],
                search_space=options["search_space"],
                fixed_params=options["fixed_params"],
                return_importances=True,
                context=context,
            )
        except (BackendError, ExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures
            msg = f"lightgbm: feature selection failed. Root cause: {root_cause(exc)}."
            raise ExecutionError(msg) from exc

        importances = details["importances_df"]
        indexed_importances = importances.set_index("feature")
        lgbm_scores = indexed_importances["lgbm_norm"].to_dict()
        shap_scores = indexed_importances["shap_norm"].to_dict()
        lgbm_cumulative = indexed_importances["lgbm_cumsum"].to_dict()
        shap_cumulative = indexed_importances["shap_cumsum"].to_dict()
        selected = set(details["selected_features"])

        decisions = [
            FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason=(
                    "passed_lgbm_shap_selection"
                    if feature in selected
                    else "failed_lgbm_shap_selection"
                ),
                value=float(
                    max(
                        lgbm_cumulative[feature] / options["lgbm_threshold"],
                        shap_cumulative[feature] / options["shap_threshold"],
                    ),
                ),
                threshold=1.0,
                keep=feature in selected,
            )
            for feature in feature_cols
        ]

        context.scores[self.method_name] = {
            "importances": {key: float(value) for key, value in lgbm_scores.items()},
            "shap_importances": {
                key: float(value) for key, value in shap_scores.items()
            },
            "lgbm_threshold": options["lgbm_threshold"],
            "shap_threshold": options["shap_threshold"],
            "optuna_mode": options["optuna_mode"],
            "search_space": options["search_space"],
            "fixed_params": options["fixed_params"],
            "fold_execution": "driver",
            "global_best_params": details["global_best_params"],
            "fold_best_params": details["fold_best_params"],
        }
        logger.info(
            "LightGbmSelector: evaluated %d features, kept %d",
            len(feature_cols),
            len(selected),
        )
        return decisions

    def _load_backends(
        self: LightGbmSelector,
    ) -> tuple[Any, Any, Any]:
        """Load optional model dependencies."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            msg = (
                "lightgbm: LightGBM is required. Install the lightgbm optional "
                "dependency."
            )
            raise BackendError(msg) from exc

        if shap is None:
            msg = "lightgbm: SHAP is required. Install the shap optional dependency."
            raise BackendError(msg)
        if optuna is None:
            msg = (
                "lightgbm: Optuna is required. Install the optuna optional "
                "dependency."
            )
            raise BackendError(msg)
        return lgb, shap, optuna

    def _resolve_options(
        self: LightGbmSelector,
        context: StageContext,
    ) -> dict[str, Any]:
        """Resolve and validate method options without changing draft defaults."""
        params = self.config.params
        try:
            legacy_n_trials = params.get("n_trials")
            optuna_settings = resolve_optuna_settings(
                params.get("optuna_params", {}),
                method_name=self.method_name,
                n_trials=(
                    _DEFAULT_N_TRIALS
                    if legacy_n_trials is None
                    else int(legacy_n_trials)
                ),
            )
            if "n_jobs" in params:
                n_jobs = int(params["n_jobs"])
            elif "driver_n_jobs" in params:
                n_jobs = int(params["driver_n_jobs"])
            else:
                n_jobs = -1
            options: dict[str, Any] = {
                "lgbm_threshold": float(params.get("lgbm_threshold", 0.85)),
                "shap_threshold": float(params.get("shap_threshold", 0.85)),
                "n_folds": int(
                    params.get("n_folds")
                    or self.config.cross_validation.folds
                ),
                "n_trials": optuna_settings["n_trials"],
                "n_startup_trials": optuna_settings["n_startup_trials"],
                "sampler": optuna_settings["sampler"],
                "timeout": optuna_settings["timeout"],
                "max_rows": min(
                    int(params.get("max_rows", 250_000)),
                    int(context.config.execution.max_local_rows),
                ),
                "sample_fraction": params.get("sample_fraction"),
                "optuna_mode": str(
                    params.get("optuna_mode", "global"),
                ).lower(),
                "n_jobs": n_jobs,
                "shap_max_rows": int(params.get("shap_max_rows", 5_000)),
            }
            fixed, overrides = split_parameters(
                params.get("parameters", {}),
                method_name=self.method_name,
            )
            options["fixed_params"] = fixed
            options["search_space"] = build_search_space(
                DEFAULT_SEARCH_SPACE,
                overrides=overrides,
                fixed=fixed,
            )
            sample_fraction = options["sample_fraction"]
            if sample_fraction is not None:
                options["sample_fraction"] = float(sample_fraction)
        except (TypeError, ValueError) as exc:
            msg = f"lightgbm: invalid numeric model parameter. Root cause: {exc}."
            raise ExecutionError(msg) from exc

        for name in ("lgbm_threshold", "shap_threshold"):
            if not 0.0 < options[name] <= 1.0:
                msg = f"lightgbm: {name} must be in (0, 1]; got {options[name]!r}."
                raise ExecutionError(msg)
        if options["n_folds"] < 2:
            msg = "lightgbm: n_folds must be at least 2."
            raise ExecutionError(msg)
        if options["max_rows"] < 1:
            msg = "lightgbm: max_rows must be at least 1."
            raise ExecutionError(msg)
        if options["optuna_mode"] not in OPTUNA_MODES:
            msg = (
                f"lightgbm: optuna_mode must be one of "
                f"{sorted(OPTUNA_MODES)}."
            )
            raise ExecutionError(msg)
        if options["n_jobs"] == 0 or options["n_jobs"] < -1:
            msg = "lightgbm: n_jobs must be -1 or a positive integer."
            raise ExecutionError(msg)
        if options["shap_max_rows"] < 1:
            msg = "lightgbm: shap_max_rows must be at least 1."
            raise ExecutionError(msg)
        if (
            sample_fraction is not None
            and not 0.0 < options["sample_fraction"] <= 1.0
        ):
            msg = "lightgbm: sample_fraction must be in (0, 1]."
            raise ExecutionError(msg)
        return options

    def _extract_and_prep_data(
        self: LightGbmSelector,
        df: Any,
        target_col: str,
        feature_cols: list[str],
        max_rows: int,
        sample_fraction: float | None,
        seed: int,
        context: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build the bounded local numeric matrix used by the draft algorithm."""
        local = prepare_numeric_frame(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            method_name=self.method_name,
            context=context,
        )
        return (
            local.loc[:, feature_cols].to_numpy(dtype=float),
            local[target_col].to_numpy(),
            list(feature_cols),
        )

    def _select_robust_features(
        self: LightGbmSelector,
        *,
        df: Any,
        target_col: str,
        feature_cols: list[str],
        n_trials: int,
        n_folds: int,
        max_rows_limit: int,
        sample_fraction: float | None,
        lgbm_threshold: float,
        shap_threshold: float,
        seed: int,
        optuna_mode: str,
        n_jobs: int,
        shap_max_rows: int,
        search_space: dict[str, dict[str, Any]] | None = None,
        fixed_params: dict[str, Any] | None = None,
        n_startup_trials: int = 10,
        sampler: str = "TPE",
        timeout: int | None = None,
        return_importances: bool = False,
        context: Any | None = None,
    ) -> list[str] | dict[str, Any]:
        """Tune parameters and execute every outer fold on the driver."""
        try:
            from sklearn.model_selection import StratifiedKFold
        except ImportError as exc:
            msg = (
                "lightgbm: scikit-learn is required. Install the sklearn "
                "optional dependency."
            )
            raise BackendError(msg) from exc

        feature_matrix, target, evaluated = self._extract_and_prep_data(
            df,
            target_col,
            feature_cols,
            max_rows_limit,
            sample_fraction,
            seed,
            context,
        )
        classes, class_counts = np.unique(target, return_counts=True)
        if len(classes) != 2:
            msg = (
                "lightgbm: binary classification requires exactly two target "
                f"classes; got {classes.tolist()}."
            )
            raise ExecutionError(msg)
        if int(class_counts.min()) < n_folds:
            msg = (
                "lightgbm: each target class must contain at least n_folds rows "
                f"after sampling; class counts are {class_counts.tolist()}."
            )
            raise ExecutionError(msg)

        global_best_params: dict[str, Any] | None = None
        if optuna_mode == "global":
            global_best_params = tune_parameters(
                feature_matrix,
                target,
                n_trials=n_trials,
                seed=seed,
                n_jobs=n_jobs,
                search_space=search_space,
                fixed_params=fixed_params,
                n_startup_trials=n_startup_trials,
                sampler=sampler,
                timeout=timeout,
            )
        folds = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        logger.info(
            "LightGbmSelector: running %d folds on driver, optuna_mode=%s",
            n_folds,
            optuna_mode,
        )

        fold_lgbm: list[np.ndarray] = []
        fold_shap: list[np.ndarray] = []
        fold_best_params: dict[str, dict[str, Any]] = {}
        for fold_index, (_, valid_indices) in enumerate(
            folds.split(feature_matrix, target),
            start=1,
        ):
            lgbm_values, shap_values, best_params = self._run_fold(
                feature_matrix,
                target,
                fold_index=fold_index,
                valid_indices=np.asarray(valid_indices, dtype=np.int64),
                seed=seed + fold_index,
                optuna_mode=optuna_mode,
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
                sampler=sampler,
                timeout=timeout,
                n_jobs=n_jobs,
                shap_max_rows=shap_max_rows,
                global_params=global_best_params,
                search_space=search_space,
                fixed_params=fixed_params,
            )
            fold_lgbm.append(lgbm_values)
            fold_shap.append(shap_values)
            fold_best_params[str(fold_index)] = dict(best_params)

        if not fold_lgbm:
            msg = "lightgbm: cross-validation produced no folds."
            raise ExecutionError(msg)

        lgbm_importances = np.sum(fold_lgbm, axis=0) / n_folds
        shap_importances = np.sum(fold_shap, axis=0) / n_folds

        details = self._aggregate_importances(
            evaluated,
            lgbm_importances,
            shap_importances,
            lgbm_threshold,
            shap_threshold,
        )
        details["global_best_params"] = (
            dict(global_best_params)
            if global_best_params is not None
            else None
        )
        details["fold_best_params"] = fold_best_params
        return details if return_importances else details["selected_features"]

    @staticmethod
    def _run_fold(
        feature_matrix: np.ndarray,
        target: np.ndarray,
        *,
        fold_index: int,
        valid_indices: np.ndarray,
        seed: int,
        optuna_mode: str,
        n_trials: int,
        n_jobs: int,
        shap_max_rows: int,
        n_startup_trials: int = 10,
        sampler: str = "TPE",
        timeout: int | None = None,
        global_params: dict[str, Any] | None = None,
        search_space: dict[str, dict[str, Any]] | None = None,
        fixed_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Tune if requested, train one fold, and compute LGBM/SHAP importances.

        Args:
            feature_matrix: Full local feature matrix shared by every fold.
            target: Full local target vector shared by every fold.
            fold_index: One-based fold number, used in error messages.
            valid_indices: Row positions held out as this fold's validation part.
            seed: Deterministic seed for tuning, the model and SHAP sampling.
            optuna_mode: ``"global"`` or ``"per_fold"``.
            n_trials: Optuna trials, used only when ``optuna_mode="per_fold"``.
            n_jobs: Thread count forced onto the model.
            shap_max_rows: Upper bound on rows explained by SHAP.
            global_params: Parameters tuned once, required unless tuning per fold.
            search_space: Parameter specifications tuned by Optuna.
            fixed_params: Scalar parameters passed through unchanged.

        Returns:
            Tuple of ``(lgbm_importances, shap_importances, best_params)``.

        Raises:
            BackendError: When LightGBM or SHAP is unavailable.
            ExecutionError: When tuning, training or SHAP fails for this fold.
        """
        try:
            import lightgbm as lgb
            import shap
        except ImportError as exc:
            msg = (
                "lightgbm: LightGBM and SHAP must be installed on the driver "
                "process running fold execution."
            )
            raise BackendError(msg) from exc

        row_count = len(target)
        train_mask = np.ones(row_count, dtype=bool)
        train_mask[valid_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_matrix = feature_matrix[train_indices]
        train_target = target[train_indices]
        valid_matrix = feature_matrix[valid_indices]

        if optuna_mode == "per_fold":
            best_params = tune_parameters(
                train_matrix,
                train_target,
                n_trials=n_trials,
                seed=seed,
                n_jobs=n_jobs,
                search_space=search_space,
                fixed_params=fixed_params,
                n_startup_trials=n_startup_trials,
                sampler=sampler,
                timeout=timeout,
            )
        elif global_params is not None:
            best_params = _finalize_parameters(
                global_params,
                seed=seed,
                n_jobs=n_jobs,
            )
        else:
            msg = (
                f"lightgbm: fold {fold_index} has no global Optuna "
                "parameters."
            )
            raise ExecutionError(msg)

        try:
            model = lgb.LGBMClassifier(**best_params)
            model.fit(train_matrix, train_target)
            lgbm_importances = np.asarray(
                model.booster_.feature_importance(importance_type="split"),
                dtype=float,
            )

            sample_size = min(len(valid_matrix), shap_max_rows)
            if len(valid_matrix) <= sample_size:
                shap_sample = valid_matrix
            else:
                rng = np.random.default_rng(seed)
                sample_indices = rng.choice(
                    len(valid_matrix),
                    sample_size,
                    replace=False,
                )
                shap_sample = valid_matrix[sample_indices]
            shap_values = shap.TreeExplainer(model).shap_values(shap_sample)
            normalized_shap = normalize_binary_shap_values(shap_values)
            shap_importances = np.abs(normalized_shap).mean(axis=0)
        except Exception as exc:  # noqa: BLE001 - model/SHAP failures
            msg = f"lightgbm: fold {fold_index} failed: {exc}"
            raise ExecutionError(msg) from exc

        return lgbm_importances, shap_importances, dict(best_params)

    @staticmethod
    def _aggregate_importances(
        feature_cols: list[str],
        lgbm_importances: np.ndarray,
        shap_importances: np.ndarray,
        lgbm_threshold: float,
        shap_threshold: float,
    ) -> dict[str, Any]:
        """Normalize importances and apply the draft cumulative intersection."""
        lgbm_total = float(np.sum(lgbm_importances))
        shap_total = float(np.sum(shap_importances))
        if not np.isfinite(lgbm_total) or lgbm_total <= 0.0:
            msg = "lightgbm: split importances have a non-positive total."
            raise ExecutionError(msg)
        if not np.isfinite(shap_total) or shap_total <= 0.0:
            msg = "lightgbm: SHAP importances have a non-positive total."
            raise ExecutionError(msg)

        importances = pd.DataFrame(
            {
                "feature": feature_cols,
                "lgbm_imp": lgbm_importances,
                "shap_imp": shap_importances,
            }
        )
        importances["lgbm_norm"] = importances["lgbm_imp"] / lgbm_total
        importances["shap_norm"] = importances["shap_imp"] / shap_total

        lgbm_sorted = importances.sort_values(
            "lgbm_norm",
            ascending=False,
            kind="stable",
        ).copy()
        lgbm_sorted["lgbm_cumsum"] = lgbm_sorted["lgbm_norm"].cumsum()
        lgbm_cumulative = lgbm_sorted.set_index("feature")["lgbm_cumsum"]
        importances["lgbm_cumsum"] = importances["feature"].map(lgbm_cumulative)
        lgbm_selected = set(
            lgbm_sorted.loc[
                lgbm_sorted["lgbm_cumsum"] <= lgbm_threshold,
                "feature",
            ]
        )

        shap_sorted = importances.sort_values(
            "shap_norm",
            ascending=False,
            kind="stable",
        ).copy()
        shap_sorted["shap_cumsum"] = shap_sorted["shap_norm"].cumsum()
        shap_cumulative = shap_sorted.set_index("feature")["shap_cumsum"]
        importances["shap_cumsum"] = importances["feature"].map(shap_cumulative)
        shap_selected = set(
            shap_sorted.loc[
                shap_sorted["shap_cumsum"] <= shap_threshold,
                "feature",
            ]
        )
        selected = [
            feature
            for feature in feature_cols
            if feature in lgbm_selected and feature in shap_selected
        ]
        return {
            "selected_features": selected,
            "lgbm_selected": [
                feature for feature in feature_cols if feature in lgbm_selected
            ],
            "shap_selected": [
                feature for feature in feature_cols if feature in shap_selected
            ],
            "lgbm_dropped": [
                feature for feature in feature_cols if feature not in lgbm_selected
            ],
            "shap_dropped": [
                feature for feature in feature_cols if feature not in shap_selected
            ],
            "importances_df": importances,
        }


def build_trial_parameters(
    trial: Any,
    *,
    seed: int,
    n_jobs: int,
    search_space: Mapping[str, Mapping[str, Any]] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build LightGBM parameters for one Optuna trial.

    Args:
        trial: Active Optuna trial.
        seed: Deterministic seed forced onto the model.
        n_jobs: Thread count forced onto the model.
        search_space: Parameter specifications to tune. Defaults to
            ``DEFAULT_SEARCH_SPACE``, preserving the established behaviour.
        fixed_params: Scalar parameters passed through unchanged.

    Returns:
        Parameter dictionary for ``lgb.LGBMClassifier``.
    """
    space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    suggested = {
        name: suggest_parameter(trial, name, specification, method_name="lightgbm")
        for name, specification in space.items()
    }
    return _finalize_parameters(
        {**dict(fixed_params or {}), **suggested},
        seed=seed,
        n_jobs=n_jobs,
    )


def tune_parameters(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_trials: int,
    seed: int,
    n_jobs: int,
    search_space: Mapping[str, Mapping[str, Any]] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    n_startup_trials: int = 10,
    sampler: str = "TPE",
    timeout: int | None = None,
) -> dict[str, Any]:
    """Tune one LightGBM parameter set on a stratified 80/20 hold-out.

    ``search_space`` and ``fixed_params`` come from ``model.params.parameters``;
    when omitted, ``DEFAULT_SEARCH_SPACE`` is used. The sampler settings come
    from ``model.params.optuna_params``.
    """
    try:
        import lightgbm as lgb
        import optuna
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        msg = (
            "lightgbm: LightGBM, Optuna, and scikit-learn must be installed "
            "on the process running tuning."
        )
        raise BackendError(msg) from exc

    train_matrix, valid_matrix, train_target, valid_target = train_test_split(
        feature_matrix,
        target,
        test_size=0.2,
        stratify=target,
        random_state=seed,
    )
    space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    study = optuna.create_study(
        direction="maximize",
        sampler=build_sampler(
            optuna,
            sampler_name=sampler,
            search_space=space,
            seed=seed,
            n_startup_trials=n_startup_trials,
            method_name="lightgbm",
        ),
    )

    def objective(trial: Any) -> float:
        trial_params = build_trial_parameters(
            trial,
            seed=seed,
            n_jobs=n_jobs,
            search_space=space,
            fixed_params=fixed_params,
        )
        model = lgb.LGBMClassifier(**trial_params)
        model.fit(
            train_matrix,
            train_target,
            eval_set=[(valid_matrix, valid_target)],
        )
        predictions = model.predict_proba(valid_matrix)[:, 1]
        return float(roc_auc_score(valid_target, predictions))

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except ExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - third-party tuning failures
        msg = f"lightgbm: Optuna tuning failed: {exc}"
        raise ExecutionError(msg) from exc

    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        msg = "lightgbm: Optuna finished without a completed trial. Increase n_trials or the timeout."
        raise ExecutionError(msg)

    return _finalize_parameters(
        {**dict(fixed_params or {}), **study.best_params},
        seed=seed,
        n_jobs=n_jobs,
    )


def normalize_binary_shap_values(shap_values: Any) -> np.ndarray:
    """Normalize SHAP outputs from supported versions for positive class."""
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values
    if hasattr(values, "values"):
        values = values.values

    normalized = np.asarray(values)
    if normalized.ndim == 3:
        normalized = normalized[:, :, 1]
    if normalized.ndim != 2:
        msg = (
            "lightgbm: unsupported SHAP output shape "
            f"{normalized.shape!r}; expected a 2D feature matrix."
        )
        raise ExecutionError(msg)
    return normalized


def _finalize_parameters(
    parameters: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Attach fixed binary-classification and execution parameters."""
    return {
        **dict(parameters),
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": n_jobs,
        "random_state": seed,
    }
