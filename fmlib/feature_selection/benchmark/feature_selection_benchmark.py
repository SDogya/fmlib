"""Standalone CatBoost benchmark for feature-selection approaches.

The script intentionally lives outside ``fmlib``. It tunes CatBoost on the
fixed train/validation split, then trains fold models and evaluates their
ensemble on the fixed test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, NoReturn


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError as error:
        message = "OmegaConf is required: install it together with catboost and optuna."
        raise RuntimeError(message) from error

    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        message = f"Config must contain a mapping at its root: {path}"
        raise ValueError(message)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        message = f"Resolved config must contain a mapping at its root: {path}"
        raise ValueError(message)
    return resolved


def _runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import optuna
        import pandas as pd
        from catboost import CatBoostClassifier, Pool
        from sklearn import metrics, model_selection
    except ImportError as error:
        package = getattr(error, "name", "runtime dependency")
        message = (
            f"Missing {package}. Install pandas, scikit-learn, catboost, optuna, and OmegaConf "
            "in the environment used to run this script."
        )
        raise RuntimeError(message) from error
    return np, optuna, pd, CatBoostClassifier, Pool, (metrics, model_selection)


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        message = f"{name} must be a mapping."
        raise ValueError(message)
    return value


def _required_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        message = f"{name} must be a list."
        raise ValueError(message)
    return value


def _raise_value_error(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_config(config: dict[str, Any]) -> None:
    benchmark = _required_mapping(config.get("benchmark"), "benchmark")
    optuna_config = _required_mapping(config.get("optuna"), "optuna")
    catboost_config = _required_mapping(config.get("catboost"), "catboost")
    datasets = _required_list(config.get("datasets"), "datasets")

    if not datasets:
        _raise_value_error("datasets must not be empty.")
    if int(benchmark.get("n_splits", 0)) < 2:
        _raise_value_error("benchmark.n_splits must be at least 2.")
    run_baseline = benchmark.get("run_baseline", True)
    if not isinstance(run_baseline, bool):
        _raise_value_error("benchmark.run_baseline must be a boolean.")
    if int(optuna_config.get("n_trials", 0)) < 1:
        _raise_value_error("optuna.n_trials must be at least 1.")
    _required_mapping(catboost_config.get("search_space", {}), "catboost.search_space")
    _required_mapping(catboost_config.get("fixed_params", {}), "catboost.fixed_params")

    names: set[str] = set()
    for index, dataset in enumerate(datasets):
        dataset = _required_mapping(dataset, f"datasets[{index}]")
        required = {
            "name",
            "train_path",
            "valid_path",
            "test_path",
            "target",
            "technical_columns",
            "categorical_columns",
            "continuous_columns",
            "feature_sets",
        }
        missing = sorted(required - dataset.keys())
        if missing:
            _raise_value_error(f"datasets[{index}] is missing fields: {missing}")
        name = str(dataset["name"])
        if name in names:
            _raise_value_error(f"Dataset name is duplicated: {name}")
        names.add(name)
        for field in ("technical_columns", "categorical_columns", "continuous_columns"):
            _required_list(dataset[field], f"{name}.{field}")
        group_column = dataset.get("group_column")
        if group_column is not None and (not isinstance(group_column, str) or not group_column.strip()):
            _raise_value_error(f"{name}.group_column must be a non-empty string.")
        feature_sets = _required_mapping(dataset["feature_sets"], f"{name}.feature_sets")
        if not run_baseline and not feature_sets:
            _raise_value_error(f"{name}.feature_sets must not be empty when baseline is disabled.")
        if "baseline" in feature_sets:
            _raise_value_error(f"{name}.feature_sets must not define reserved name 'baseline'.")
        for method, dropped in feature_sets.items():
            if isinstance(dropped, list):
                continue
            if not isinstance(dropped, str) or not dropped.strip():
                _raise_value_error(f"{name}.feature_sets.{method} must be a list or a non-empty file path.")


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def _read_frame(path: Path, pd: Any) -> Any:
    if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".csv.gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return pd.read_feather(path)
    _raise_value_error(f"Unsupported dataset format for {path}; use Parquet, CSV, or Feather.")


def _as_unique_strings(values: list[Any], name: str) -> list[str]:
    result = [str(value) for value in values]
    duplicates = sorted({value for value in result if result.count(value) > 1})
    if duplicates:
        _raise_value_error(f"{name} contains duplicate columns: {duplicates}")
    return result


def _feature_sets(
    dataset: dict[str, Any],
    config_dir: Path,
    include_baseline: bool = True,
) -> dict[str, list[str]]:
    methods = {"baseline": []} if include_baseline else {}
    for method, value in dataset["feature_sets"].items():
        name = f"{dataset['name']}.feature_sets.{method}"
        if isinstance(value, list):
            dropped = value
        else:
            path = _resolve_path(value, config_dir)
            lines = path.read_text(encoding="utf-8").splitlines()
            dropped = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        methods[str(method)] = _as_unique_strings(dropped, name)
    return methods


def _validate_frames(
    dataset: dict[str, Any],
    frames: dict[str, Any],
    feature_sets: dict[str, list[str]],
    n_splits: int,
) -> dict[str, Any]:
    name = str(dataset["name"])
    target = str(dataset["target"])
    technical = _as_unique_strings(dataset["technical_columns"], f"{name}.technical_columns")
    categorical = _as_unique_strings(dataset["categorical_columns"], f"{name}.categorical_columns")
    continuous = _as_unique_strings(dataset["continuous_columns"], f"{name}.continuous_columns")
    id_columns = _as_unique_strings(dataset.get("id_columns", []), f"{name}.id_columns")
    group_column = dataset.get("group_column")
    group_column = str(group_column).strip() if group_column is not None else None
    if group_column is not None and group_column not in id_columns:
        _raise_value_error(f"{name}.group_column must be listed in id_columns.")

    feature_overlap = sorted(set(categorical) & set(continuous))
    if feature_overlap:
        _raise_value_error(f"{name}: categorical and continuous columns overlap: {feature_overlap}")
    technical_features = sorted(set(technical) & (set(categorical) | set(continuous)))
    if technical_features:
        _raise_value_error(f"{name}: technical columns must not be used as features: {technical_features}")
    forbidden = sorted({target} & (set(technical) | set(categorical) | set(continuous)))
    if forbidden:
        _raise_value_error(f"{name}: target must not be listed as a technical or feature column.")

    declared_features = set(categorical) | set(continuous)
    if not declared_features:
        _raise_value_error(f"{name}: no feature columns are declared.")
    for method, dropped in feature_sets.items():
        unknown = sorted(set(dropped) - declared_features)
        if unknown:
            _raise_value_error(f"{name}/{method}: dropped columns are not declared features: {unknown}")
        if set(dropped) == declared_features:
            _raise_value_error(f"{name}/{method}: all features were dropped.")

    required = declared_features | set(technical) | set(id_columns) | {target}
    for split, frame in frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            _raise_value_error(f"{name}/{split}: missing columns: {missing}")
        if frame.empty:
            _raise_value_error(f"{name}/{split}: split is empty.")
        classes = frame[target].dropna().unique()
        if len(classes) != 2:
            _raise_value_error(f"{name}/{split}: target must contain exactly two non-null classes.")
        if frame[target].isna().any():
            _raise_value_error(f"{name}/{split}: target contains null values.")
        if group_column is not None:
            if frame[group_column].isna().any():
                _raise_value_error(f"{name}/{split}: group column contains null values.")
            group_values = set(frame[group_column].unique())
            if group_values != {0, 1}:
                _raise_value_error(f"{name}/{split}: group column must contain exactly values 0 and 1.")

    train_counts = frames["train"][target].value_counts()
    if int(train_counts.min()) < n_splits:
        _raise_value_error(
            f"{name}/train: every target class needs at least n_splits={n_splits} rows; counts={train_counts.to_dict()}"
        )

    if group_column is not None:
        for group_value, group_name in ((0, "target"), (1, "control")):
            group_targets = frames["test"].loc[frames["test"][group_column] == group_value, target]
            if group_targets.nunique() != 2:
                _raise_value_error(f"{name}/test: {group_name} group must contain both target classes.")

    if bool(dataset.get("enforce_disjoint_ids", False)):
        if not id_columns:
            _raise_value_error(f"{name}: enforce_disjoint_ids requires id_columns.")
        split_keys = {
            split: set(map(tuple, frame[id_columns].drop_duplicates().itertuples(index=False, name=None)))
            for split, frame in frames.items()
        }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = split_keys[left] & split_keys[right]
            if overlap:
                _raise_value_error(f"{name}: {left}/{right} contain {len(overlap)} overlapping IDs.")

    return {
        "target": target,
        "technical_columns": technical,
        "categorical_columns": categorical,
        "continuous_columns": continuous,
        "id_columns": id_columns,
        "group_column": group_column,
    }


def _prepare_features(
    frame: Any,
    features: list[str],
    categorical: list[str],
    continuous: list[str],
) -> Any:
    prepared = frame.loc[:, features].copy()
    if continuous:
        prepared[continuous] = prepared[continuous].astype("float32")
    for column in categorical:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def _prepare_dataset(
    frames: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any | None]:
    target = schema["target"]
    categorical = schema["categorical_columns"]
    continuous = schema["continuous_columns"]
    features = categorical + continuous
    targets = {split: frame[target].to_numpy(copy=True) for split, frame in frames.items()}
    test_groups = frames["test"][schema["group_column"]].to_numpy(copy=True) if schema["group_column"] is not None else None
    prepared_frames = {split: _prepare_features(frame, features, categorical, continuous) for split, frame in frames.items()}
    return prepared_frames, targets, test_groups


def _build_method_pools(
    prepared_frames: dict[str, Any],
    targets: dict[str, Any],
    features: list[str],
    categorical: list[str],
    pool_class: Any,
) -> dict[str, Any]:
    pools = {}
    for split, frame in prepared_frames.items():
        selected = frame.loc[:, features]
        pools[split] = pool_class(
            data=selected,
            label=targets[split],
            cat_features=categorical,
            feature_names=features,
        )
    return pools


def _suggest_parameters(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, specification in search_space.items():
        specification = _required_mapping(specification, f"catboost.search_space.{name}")
        parameter_type = specification.get("type")
        if parameter_type == "int":
            parameters[name] = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification.get("step", 1)),
                log=bool(specification.get("log", False)),
            )
        elif parameter_type == "float":
            parameters[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                step=specification.get("step"),
                log=bool(specification.get("log", False)),
            )
        elif parameter_type == "categorical":
            parameters[name] = trial.suggest_categorical(name, _required_list(specification["choices"], f"{name}.choices"))
        else:
            _raise_value_error(f"Unsupported search-space type for {name}: {parameter_type}")
    return parameters


def _model_parameters(
    tuned: dict[str, Any],
    catboost_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    parameters = dict(_required_mapping(catboost_config.get("fixed_params", {}), "catboost.fixed_params"))
    parameters.update(tuned)
    parameters.update(
        {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": int(catboost_config.get("thread_count", -1)),
        }
    )
    return parameters


def _fit_model(
    classifier: Any,
    parameters: dict[str, Any],
    train_pool: Any,
    eval_pool: Any,
    early_stopping_rounds: int | None,
) -> Any:
    model = classifier(**parameters)
    fit_kwargs: dict[str, Any] = {
        "X": train_pool,
        "eval_set": eval_pool,
        "use_best_model": True,
    }
    if early_stopping_rounds is not None:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model.fit(**fit_kwargs)
    return model


def _folds_fingerprint(folds: list[tuple[Any, Any]]) -> str:
    digest = hashlib.sha256()
    for train_indices, validation_indices in folds:
        digest.update(",".join(map(str, train_indices)).encode())
        digest.update(b"|")
        digest.update(",".join(map(str, validation_indices)).encode())
        digest.update(b";")
    return digest.hexdigest()


def _schema_fingerprint(schema: dict[str, Any], dropped: list[str]) -> str:
    payload = {
        "target": schema["target"],
        "categorical_columns": sorted(schema["categorical_columns"]),
        "continuous_columns": sorted(schema["continuous_columns"]),
        "technical_columns": sorted(schema["technical_columns"]),
        "dropped_columns": sorted(dropped),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _trial_records(study: Any) -> list[dict[str, Any]]:
    records = []
    for trial in study.trials:
        records.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            }
        )
    return records


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("catboost", "numpy", "optuna", "pandas", "PyYAML", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "result"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    message = f"Object of type {type(value).__name__} is not JSON serializable."
    raise TypeError(message)


def _aggregate_metrics(
    np: Any,
    metrics: Any,
    fold_validation_auc: list[float],
    fold_test_auc: list[float],
    valid_targets: Any,
    valid_predictions: list[Any],
    test_targets: Any,
    test_predictions: list[Any],
    test_groups: Any | None = None,
) -> dict[str, Any]:
    ensemble_valid = np.mean(np.vstack(valid_predictions), axis=0)
    ensemble_test = np.mean(np.vstack(test_predictions), axis=0)
    payload = {
        "fold_validation_auc": fold_validation_auc,
        "fold_validation_auc_mean": float(np.mean(fold_validation_auc)),
        "fold_validation_auc_std": float(np.std(fold_validation_auc)),
        "fold_test_auc": fold_test_auc,
        "fold_test_auc_mean_diagnostic": float(np.mean(fold_test_auc)),
        "fold_test_auc_std_diagnostic": float(np.std(fold_test_auc)),
        "ensemble_valid_auc": float(metrics.roc_auc_score(valid_targets, ensemble_valid)),
        "ensemble_test_auc": float(metrics.roc_auc_score(test_targets, ensemble_test)),
        "ensemble_test_auc_target_group": None,
        "ensemble_test_auc_control_group": None,
    }
    if test_groups is not None:
        group_values = np.asarray(test_groups)
        target_values = np.asarray(test_targets)
        target_mask = group_values == 0
        control_mask = group_values == 1
        payload["ensemble_test_auc_target_group"] = float(
            metrics.roc_auc_score(target_values[target_mask], ensemble_test[target_mask])
        )
        payload["ensemble_test_auc_control_group"] = float(
            metrics.roc_auc_score(target_values[control_mask], ensemble_test[control_mask])
        )
    return payload


def _run_method(
    dataset: dict[str, Any],
    method: str,
    dropped: list[str],
    pools: dict[str, Any],
    targets: dict[str, Any],
    test_groups: Any | None,
    schema: dict[str, Any],
    folds: list[tuple[Any, Any]],
    config: dict[str, Any],
    dependencies: tuple[Any, Any, Any, Any, Any, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    np, optuna, _, classifier, _, sklearn_modules = dependencies
    metrics, _ = sklearn_modules
    benchmark_config = config["benchmark"]
    optuna_config = config["optuna"]
    catboost_config = config["catboost"]
    seed = int(benchmark_config.get("seed", 42))
    features = [column for column in schema["categorical_columns"] + schema["continuous_columns"] if column not in set(dropped)]
    categorical = [column for column in schema["categorical_columns"] if column in features]
    search_space = _required_mapping(catboost_config.get("search_space", {}), "catboost.search_space")
    early_stopping = catboost_config.get("early_stopping_rounds")
    early_stopping = int(early_stopping) if early_stopping is not None else None

    def objective(trial: Any) -> float:
        tuned = _suggest_parameters(trial, search_space)
        parameters = _model_parameters(tuned, catboost_config, seed)
        model = _fit_model(
            classifier,
            parameters,
            pools["train"],
            pools["valid"],
            early_stopping,
        )
        predictions = model.predict_proba(pools["valid"])[:, 1]
        return float(metrics.roc_auc_score(targets["valid"], predictions))

    sampler = optuna.samplers.TPESampler(seed=seed)
    storage = optuna_config.get("storage")
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{dataset['name']}__{method}",
        storage=storage,
        load_if_exists=bool(optuna_config.get("load_if_exists", False)),
    )
    started_at = time.perf_counter()
    study.optimize(
        objective,
        n_trials=int(optuna_config["n_trials"]),
        timeout=optuna_config.get("timeout_seconds"),
        n_jobs=int(optuna_config.get("n_jobs", 1)),
        show_progress_bar=bool(optuna_config.get("show_progress_bar", False)),
    )
    tuning_seconds = time.perf_counter() - started_at

    fold_validation_auc = []
    fold_test_auc = []
    valid_predictions = []
    test_predictions = []
    best_iterations = []
    training_started_at = time.perf_counter()
    for fold_number, (fit_indices, score_indices) in enumerate(folds):
        parameters = _model_parameters(study.best_params, catboost_config, seed + fold_number)
        fit_pool = pools["train"].slice(fit_indices.tolist())
        score_pool = pools["train"].slice(score_indices.tolist())
        model = _fit_model(
            classifier,
            parameters,
            fit_pool,
            score_pool,
            early_stopping,
        )
        current_test_predictions = model.predict_proba(pools["test"])[:, 1]
        fold_validation_auc.append(float(model.get_best_score()["validation"]["AUC"]))
        fold_test_auc.append(float(metrics.roc_auc_score(targets["test"], current_test_predictions)))
        valid_predictions.append(model.predict_proba(pools["valid"])[:, 1])
        test_predictions.append(current_test_predictions)
        best_iterations.append(int(model.get_best_iteration()))
    training_seconds = time.perf_counter() - training_started_at

    metrics_payload = _aggregate_metrics(
        np,
        metrics,
        fold_validation_auc,
        fold_test_auc,
        targets["valid"],
        valid_predictions,
        targets["test"],
        test_predictions,
        test_groups,
    )
    result = {
        "dataset": str(dataset["name"]),
        "method": method,
        "main_metric": "ensemble_test_auc",
        "metrics": metrics_payload,
        "feature_count": len(features),
        "categorical_feature_count": len(categorical),
        "continuous_feature_count": len(features) - len(categorical),
        "kept_columns": features,
        "dropped_columns": dropped,
        "schema_sha256": _schema_fingerprint(schema, dropped),
        "folds_sha256": _folds_fingerprint(folds),
        "seed": seed,
        "best_params": study.best_params,
        "best_tuning_valid_auc": float(study.best_value),
        "best_iterations": best_iterations,
        "timing_seconds": {
            "tuning": tuning_seconds,
            "final_training": training_seconds,
            "total": tuning_seconds + training_seconds,
        },
        "trials": _trial_records(study),
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    summary = {
        "dataset": str(dataset["name"]),
        "method": method,
        "feature_count": len(features),
        "dropped_feature_count": len(dropped),
        "best_tuning_valid_auc": result["best_tuning_valid_auc"],
        "validation_auc_mean": metrics_payload["fold_validation_auc_mean"],
        "validation_auc_std": metrics_payload["fold_validation_auc_std"],
        "ensemble_valid_auc": metrics_payload["ensemble_valid_auc"],
        "test_auc_mean_diagnostic": metrics_payload["fold_test_auc_mean_diagnostic"],
        "test_auc_std_diagnostic": metrics_payload["fold_test_auc_std_diagnostic"],
        "ensemble_test_auc": metrics_payload["ensemble_test_auc"],
        "ensemble_test_auc_target_group": metrics_payload["ensemble_test_auc_target_group"],
        "ensemble_test_auc_control_group": metrics_payload["ensemble_test_auc_control_group"],
        "total_seconds": result["timing_seconds"]["total"],
    }
    return result, summary


def run(config_path: Path) -> list[dict[str, Any]]:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    _validate_config(config)
    dependencies = _runtime_dependencies()
    _, _, pd, _, pool_class, sklearn_modules = dependencies
    _, model_selection = sklearn_modules
    benchmark_config = config["benchmark"]
    seed = int(benchmark_config.get("seed", 42))
    n_splits = int(benchmark_config["n_splits"])
    run_baseline = bool(benchmark_config.get("run_baseline", True))
    output_dir = _resolve_path(str(benchmark_config.get("output_dir", "benchmark_results")), config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    summaries: list[dict[str, Any]] = []
    baseline_scores: dict[str, float] = {}
    for dataset in config["datasets"]:
        feature_sets = _feature_sets(dataset, config_path.parent, include_baseline=run_baseline)
        frames = {
            split: _read_frame(_resolve_path(str(dataset[f"{split}_path"]), config_path.parent), pd)
            for split in ("train", "valid", "test")
        }
        schema = _validate_frames(dataset, frames, feature_sets, n_splits)
        prepared_frames, targets, test_groups = _prepare_dataset(frames, schema)
        del frames
        splitter = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(targets["train"], targets["train"]))
        for method, dropped in feature_sets.items():
            features = [
                column for column in schema["categorical_columns"] + schema["continuous_columns"] if column not in set(dropped)
            ]
            categorical = [column for column in schema["categorical_columns"] if column in features]
            pools = _build_method_pools(
                prepared_frames,
                targets,
                features,
                categorical,
                pool_class,
            )
            result, summary = _run_method(
                dataset,
                method,
                dropped,
                pools,
                targets,
                test_groups,
                schema,
                folds,
                config,
                dependencies,
            )
            result["config_sha256"] = config_sha256
            if method == "baseline":
                baseline_scores[str(dataset["name"])] = summary["ensemble_test_auc"]
            baseline_score = baseline_scores.get(str(dataset["name"]))
            summary["delta_test_auc_vs_baseline"] = (
                summary["ensemble_test_auc"] - baseline_score if baseline_score is not None else None
            )
            result["delta_test_auc_vs_baseline"] = summary["delta_test_auc_vs_baseline"]
            result_path = output_dir / f"{_safe_name(str(dataset['name']))}__{_safe_name(method)}.json"
            with result_path.open("w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2, default=_json_default)
            summaries.append(summary)
            del pools

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the benchmark YAML config.")
    args = parser.parse_args()
    try:
        summaries = run(args.config)
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"Benchmark failed: {error}\n")
        return 1
    sys.stdout.write(f"{json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

