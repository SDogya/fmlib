"""BorutaSHAP model-based feature selector."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fmlib.feature_selection.base import FeatureDecision, StageContext, resolve_step_seed
from fmlib.feature_selection.config import (
    BORUTA_MODEL_TYPES,
    BORUTA_SAMPLERS,
    ModelConfig,
)
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.default_model_param_spaces import (
    BORUTA_LGBM_SEARCH_SPACE,
    BORUTA_RF_SEARCH_SPACE,
)
from fmlib.feature_selection.utils.lama_boost_defaults import (
    apply_boost_heuristics,
    fit_lgbm_with_early_stopping,
    split_lgbm_early_stopping,
)
from fmlib.feature_selection.utils.local_data import prepare_numeric_frame, root_cause
from fmlib.feature_selection.utils.optuna_space import (
    build_sampler,
    resolve_optuna_settings,
    resolve_tuning_space,
    suggest_parameter,
)
from fmlib.feature_selection.utils.task_runtime import (
    TaskRuntime,
    binary_task,
    lgbm_estimator_class,
    lgbm_objective_params,
    optuna_direction,
    require_min_class_count,
    resolve_task,
    rf_estimator_class,
    score_model,
)

logger = logging.getLogger(__name__)

try:
    import optuna
except ImportError:
    optuna = None

try:
    from BorutaShap import BorutaShap
except Exception as exc:  # noqa: BLE001 - optional package may fail on incompatible numpy
    BorutaShap = None
    _boruta_import_error: BaseException | None = exc
else:
    _boruta_import_error = None


@dataclass(frozen=True)
class _Backends:
    """Loaded third-party classes and functions."""

    boruta_class: Any
    model_class: Any
    optuna_module: Any
    train_test_split: Any


class BorutaShapSelector:
    """Run Optuna-tuned BorutaSHAP as the final continuous-feature selector.

    The ML flow follows ``BorutaSHAP_dep.py``: tune LightGBM or RandomForest on
    a stratified hold-out split, pass the optimized model to BorutaShap, fit on
    the full bounded sample, and optionally resolve tentative features with
    ``TentativeRoughFix``.

    Args:
        config: Model-stage method and BorutaSHAP parameters.
    """

    method_name = "boruta_shap"
    stage_name = "model"

    def __init__(self: BorutaShapSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: BorutaShapSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Evaluate current continuous candidates with BorutaSHAP."""
        if not candidates:
            return []

        target_col = context.schema.target
        if not target_col:
            msg = "boruta_shap: FeatureSchema.target is required."
            raise ExecutionError(msg)
        train = context.datasets.get("train")
        if train is None:
            msg = "boruta_shap: context.datasets must contain a 'train' split."
            raise ExecutionError(msg)

        continuous = set(context.schema.continuous)
        feature_cols = [
            feature for feature in candidates if feature in continuous
        ]
        if not feature_cols:
            return []

        options = self._resolve_options(context)
        backends = self._load_backends(
            options["model_type"],
            require_optuna=bool(options["search_space"]),
            task_type=context.schema.task_type,
        )
        try:
            details = self._run_boruta_selection(
                train=train,
                target_col=target_col,
                feature_cols=feature_cols,
                options=options,
                seed=options["seed"],
                backends=backends,
                context=context,
            )
        except (BackendError, ExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures
            msg = (
                "boruta_shap: feature selection failed. "
                f"Root cause: {root_cause(exc)}."
            )
            raise ExecutionError(msg) from exc

        accepted = set(details["accepted"])
        tentative = set(details["tentative"])
        decisions = [
            FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason=(
                    "boruta_accepted"
                    if feature in accepted
                    else (
                        "boruta_tentative"
                        if feature in tentative
                        else "boruta_rejected"
                    )
                ),
                value=None,
                threshold=None,
                keep=feature in accepted,
            )
            for feature in feature_cols
        ]
        context.scores[self.method_name] = {
            "accepted": list(details["accepted"]),
            "rejected": list(details["rejected"]),
            "tentative": list(details["tentative"]),
            "model_type": options["model_type"],
            "best_auc": (
                None
                if details["best_auc"] is None
                else float(details["best_auc"])
            ),
            "best_params": {
                key: _json_value(value)
                for key, value in details["best_params"].items()
            },
            "optuna_enabled": options["optuna_enabled"],
            "optuna_trials": (
                0
                if not options["search_space"]
                else options["n_trials"]
            ),
            "boruta_trials": options["boruta_trials"],
        }
        logger.info(
            "BorutaShapSelector: evaluated %d features, kept %d",
            len(feature_cols),
            len(accepted),
        )
        return decisions

    def _resolve_options(
        self: BorutaShapSelector,
        context: StageContext,
    ) -> dict[str, Any]:
        """Resolve legacy-compatible options and execution capacity limits."""
        params = self.config.params
        optuna_params = params.get("optuna_params", {})
        raw_parameters = params.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            msg = "boruta_shap: params.parameters must be a mapping."
            raise ExecutionError(msg)
        # Legacy aliases stay supported as the fallback for optuna_params.n_trials.
        legacy_n_trials = optuna_params.get("niter") if isinstance(optuna_params, Mapping) else None
        if legacy_n_trials is None:
            legacy_n_trials = params.get("n_trials", params.get("optuna_trials", 20))
        try:
            optuna_settings = resolve_optuna_settings(
                optuna_params,
                method_name=self.method_name,
                n_trials=int(legacy_n_trials),
            )
            model_type = str(params.get("model_type", "lgbm")).lower()
            sampler_name = optuna_settings["sampler"]
            filtered = {
                name: value
                for name, value in raw_parameters.items()
                if not (model_type == "lgbm" and str(name) == "bootstrap_type")
            }
            defaults: Mapping[str, Mapping[str, Any]] = {}
            if sampler_name != "GRID":
                defaults = (
                    BORUTA_LGBM_SEARCH_SPACE
                    if model_type == "lgbm"
                    else BORUTA_RF_SEARCH_SPACE
                )
            fixed_params, search_space = resolve_tuning_space(
                filtered,
                defaults=defaults,
                enabled=optuna_settings["enabled"],
                method_name=self.method_name,
            )
            options: dict[str, Any] = {
                "model_type": model_type,
                "max_rows": min(
                    int(
                        params.get(
                            "max_rows",
                            params.get("max_rows_limit", 700_000),
                        ),
                    ),
                    int(context.config.execution.max_local_rows),
                ),
                "sample_fraction": params.get("sample_fraction"),
                "n_trials": optuna_settings["n_trials"],
                "n_startup_trials": optuna_settings["n_startup_trials"],
                "sampler": sampler_name,
                "timeout": optuna_settings["timeout"],
                "optuna_enabled": optuna_settings["enabled"],
                "boruta_trials": int(params.get("boruta_trials", 50)),
                "tentative_fix_method": params.get(
                    "tentative_fix_method",
                    "rough",
                ),
                "parameters": {
                    name: spec
                    for name, spec in filtered.items()
                    if isinstance(spec, Mapping)
                },
                "fixed_params": fixed_params,
                "search_space": search_space,
                "seed": resolve_step_seed(params, context),
            }
            if options["sample_fraction"] is not None:
                options["sample_fraction"] = float(
                    options["sample_fraction"],
                )
        except (TypeError, ValueError) as exc:
            msg = (
                "boruta_shap: invalid numeric or parameter-grid option. "
                f"Root cause: {exc}."
            )
            raise ExecutionError(msg) from exc

        if options["model_type"] not in BORUTA_MODEL_TYPES:
            msg = (
                f"boruta_shap: model_type must be one of "
                f"{sorted(BORUTA_MODEL_TYPES)}."
            )
            raise ExecutionError(msg)
        if options["sampler"] not in BORUTA_SAMPLERS:
            msg = (
                f"boruta_shap: sampler must be one of "
                f"{sorted(BORUTA_SAMPLERS)}."
            )
            raise ExecutionError(msg)
        # n_trials and n_startup_trials are validated by resolve_optuna_settings.
        for name in ("max_rows", "boruta_trials"):
            if options[name] < 1:
                msg = f"boruta_shap: {name} must be at least 1."
                raise ExecutionError(msg)
        sample_fraction = options["sample_fraction"]
        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            msg = "boruta_shap: sample_fraction must be in (0, 1]."
            raise ExecutionError(msg)
        if options["tentative_fix_method"] not in {None, "rough"}:
            msg = (
                "boruta_shap: tentative_fix_method must be 'rough' or null."
            )
            raise ExecutionError(msg)
        if options["optuna_enabled"] and options["sampler"] == "GRID":
            finite = bool(options["parameters"]) and all(
                isinstance(spec.get("values"), (list, tuple))
                and bool(spec["values"])
                for spec in options["parameters"].values()
            )
            if not finite:
                msg = (
                    "boruta_shap: GRID sampler requires every custom parameter "
                    "to define a non-empty 'values' list."
                )
                raise ExecutionError(msg)
        return options

    def _load_backends(
        self: BorutaShapSelector,
        model_type: str,
        *,
        require_optuna: bool = True,
        task_type: str = "binary_classification",
    ) -> _Backends:
        """Load optional BorutaSHAP, Optuna, sklearn, and model dependencies."""
        if BorutaShap is None:
            msg = (
                "boruta_shap: BorutaShap is required. Install the BorutaShap "
                "optional dependency."
            )
            if _boruta_import_error is not None:
                msg = f"{msg} Root cause: {root_cause(_boruta_import_error)}."
            raise BackendError(msg)
        train_test_split: Any = None
        if require_optuna:
            if optuna is None:
                msg = (
                    "boruta_shap: Optuna is required. Install the optuna optional "
                    "dependency."
                )
                raise BackendError(msg)
            try:
                from sklearn.model_selection import train_test_split
            except ImportError as exc:
                msg = (
                    "boruta_shap: scikit-learn is required. Install the sklearn "
                    "optional dependency."
                )
                raise BackendError(msg) from exc

        if model_type == "lgbm":
            model_class = lgbm_estimator_class(task_type)
        else:
            model_class = rf_estimator_class(task_type)

        return _Backends(
            boruta_class=BorutaShap,
            model_class=model_class,
            optuna_module=optuna,
            train_test_split=train_test_split,
        )

    def _run_boruta_selection(
        self: BorutaShapSelector,
        *,
        train: Any,
        target_col: str,
        feature_cols: list[str],
        options: dict[str, Any],
        seed: int,
        backends: _Backends,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """Tune the model and execute the legacy BorutaSHAP flow."""
        local = prepare_numeric_frame(
            train,
            target_col=target_col,
            feature_cols=feature_cols,
            max_rows=options["max_rows"],
            sample_fraction=options["sample_fraction"],
            seed=seed,
            method_name=self.method_name,
            context=context,
        )
        all_null = [
            feature
            for feature in feature_cols
            if local[feature].isna().all()
        ]
        if all_null:
            msg = (
                "boruta_shap: all-null features after sampling: "
                f"{all_null}."
            )
            raise ExecutionError(msg)

        features = local.loc[:, feature_cols]
        raw_target = local[target_col].to_numpy()
        task_type = (
            context.schema.task_type
            if context is not None
            else "binary_classification"
        )
        task = resolve_task(task_type, raw_target, method_name=self.method_name)
        target = task.encoded_target
        if task.is_regression:
            if len(target) < 4:
                msg = (
                    "boruta_shap: regression sample is too small for an 80/20 "
                    f"hold-out; got {len(target)} rows."
                )
                raise ExecutionError(msg)
        else:
            require_min_class_count(
                task,
                min_count=2,
                method_name=self.method_name,
            )

        fixed_params = dict(options["fixed_params"])
        search_space = {
            name: dict(spec) for name, spec in options["search_space"].items()
        }
        if options["model_type"] == "lgbm":
            fixed_params, search_space = apply_boost_heuristics(
                fixed_params,
                search_space,
                n_rows=len(target),
                library="lightgbm",
                task_type=task.task_type,
            )
        if search_space:
            test_rows = math.ceil(len(target) * 0.2)
            train_rows = len(target) - test_rows
            if task.stratify:
                n_classes = int(task.n_classes or 0)
                if test_rows < n_classes or train_rows < n_classes:
                    msg = (
                        "boruta_shap: the bounded sample is too small for a stratified "
                        f"80/20 hold-out; got {len(target)} rows."
                    )
                    raise ExecutionError(msg)
            elif test_rows < 1 or train_rows < 1:
                msg = (
                    "boruta_shap: the bounded sample is too small for an 80/20 "
                    f"hold-out; got {len(target)} rows."
                )
                raise ExecutionError(msg)

            split_kwargs: dict[str, Any] = {
                "test_size": 0.2,
                "random_state": seed,
            }
            if task.stratify:
                split_kwargs["stratify"] = target
            (
                train_features,
                valid_features,
                train_target,
                valid_target,
            ) = backends.train_test_split(
                features,
                target,
                **split_kwargs,
            )
            sampler = build_sampler(
                backends.optuna_module,
                sampler_name=options["sampler"],
                search_space=search_space,
                seed=seed,
                n_startup_trials=options["n_startup_trials"],
                method_name=self.method_name,
            )
            study = backends.optuna_module.create_study(
                direction=optuna_direction(task),
                sampler=sampler,
            )

            def objective(trial: Any) -> float:
                suggested = {
                    name: suggest_parameter(
                        trial,
                        name,
                        specification,
                        method_name=self.method_name,
                    )
                    for name, specification in search_space.items()
                }
                trial_params = {**fixed_params, **suggested}
                if options["model_type"] == "lgbm":
                    ctor_params, stopping_rounds = split_lgbm_early_stopping(
                        trial_params,
                    )
                    model = self._build_model(
                        backends.model_class,
                        options["model_type"],
                        ctor_params,
                        seed,
                        task=task,
                    )
                    fit_lgbm_with_early_stopping(
                        model,
                        train_features,
                        train_target,
                        eval_set=[(valid_features, valid_target)],
                        early_stopping_rounds=stopping_rounds,
                    )
                else:
                    model = self._build_model(
                        backends.model_class,
                        options["model_type"],
                        trial_params,
                        seed,
                        task=task,
                    )
                    model.fit(train_features, train_target)
                return score_model(task, model, valid_features, valid_target)

            try:
                study.optimize(
                    objective,
                    n_trials=options["n_trials"],
                    timeout=options["timeout"],
                    show_progress_bar=False,
                    n_jobs=1,
                )
                best_params = {**fixed_params, **study.best_params}
                best_auc = float(study.best_value)
                if not math.isfinite(best_auc):
                    msg = (
                        "boruta_shap: Optuna did not produce a finite best metric."
                    )
                    raise ExecutionError(msg)
                final_model = self._build_model(
                    backends.model_class,
                    options["model_type"],
                    best_params,
                    seed,
                    task=task,
                )
            except ExecutionError:
                raise
            except Exception as exc:  # noqa: BLE001 - third-party model failures
                msg = (
                    "boruta_shap: Optuna tuning failed. "
                    f"Root cause: {root_cause(exc)}."
                )
                raise ExecutionError(msg) from exc
        else:
            best_params = dict(fixed_params)
            best_auc = None
            final_model = self._build_model(
                backends.model_class,
                options["model_type"],
                best_params,
                seed,
                task=task,
            )

        try:
            feature_selector = backends.boruta_class(
                model=final_model,
                importance_measure="shap",
                classification=not task.is_regression,
            )
            feature_selector.fit(
                X=features,
                y=target,
                n_trials=options["boruta_trials"],
                random_state=seed,
                verbose=False,
            )
            if options["tentative_fix_method"] == "rough":
                feature_selector.TentativeRoughFix()
        except Exception as exc:  # noqa: BLE001 - third-party Boruta failures
            msg = (
                "boruta_shap: BorutaShap fitting failed. "
                f"Root cause: {root_cause(exc)}."
            )
            raise ExecutionError(msg) from exc

        accepted_set = set(getattr(feature_selector, "accepted", []))
        rejected_set = set(getattr(feature_selector, "rejected", []))
        tentative_set = set(getattr(feature_selector, "tentative", []))
        accepted = [
            feature for feature in feature_cols if feature in accepted_set
        ]
        tentative = [
            feature
            for feature in feature_cols
            if feature in tentative_set
            and feature not in accepted_set
            and feature not in rejected_set
        ]
        rejected = [
            feature
            for feature in feature_cols
            if feature in rejected_set
            or (
                feature not in accepted_set
                and feature not in tentative_set
            )
        ]
        return {
            "accepted": accepted,
            "rejected": rejected,
            "tentative": tentative,
            "best_auc": best_auc,
            "best_params": best_params,
        }

    @staticmethod
    def _build_search_space(
        model_type: str,
        overrides: Mapping[str, dict[str, Any]],
        sampler_name: str,
        fixed: Mapping[str, Any],
        *,
        enabled: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Resolve the Optuna space: YAML mappings replace the fallback.

        ``GRID`` enumerates an explicit product, so built-in ranges are not
        used: only configured entries take part. ``bootstrap_type`` is dropped
        for LightGBM because it is not a LightGBM constructor argument.
        """
        clean_overrides = {
            name: specification
            for name, specification in overrides.items()
            if not (model_type == "lgbm" and name == "bootstrap_type")
        }
        defaults: Mapping[str, Mapping[str, Any]] = {}
        if sampler_name != "GRID":
            defaults = (
                BORUTA_LGBM_SEARCH_SPACE
                if model_type == "lgbm"
                else BORUTA_RF_SEARCH_SPACE
            )
        _fixed, space = resolve_tuning_space(
            {**fixed, **clean_overrides},
            defaults=defaults,
            enabled=enabled,
            method_name="boruta_shap",
        )
        return space

    @staticmethod
    def _build_model(
        model_class: Any,
        model_type: str,
        parameters: Mapping[str, Any],
        seed: int,
        task: TaskRuntime | None = None,
    ) -> Any:
        """Build the same model shape for Optuna and final Boruta fitting."""
        resolved = task if task is not None else binary_task()
        common = {
            **dict(parameters),
            "n_jobs": -1,
            "random_state": seed,
        }
        if model_type == "lgbm":
            common, _stopping_rounds = split_lgbm_early_stopping(common)
            common.update(
                {
                    **lgbm_objective_params(resolved),
                    "verbosity": -1,
                    "bagging_seed": seed,
                    "feature_fraction_seed": seed,
                    "data_random_seed": seed,
                    "extra_seed": seed,
                    "deterministic": True,
                    "force_row_wise": True,
                },
            )
            common.pop("force_col_wise", None)
            return model_class(**common)
        return model_class(**common)


def _json_value(value: Any) -> Any:
    """Recursively convert model parameters to JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
