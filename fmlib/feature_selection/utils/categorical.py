"""Categorical preparation shared by local model-based selectors.

The feature-selection result always refers to source feature names.  This
module may build temporary model columns (multiclass target encoding), but
keeps their source mapping so selectors can aggregate importances back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.utils.local_data import prepare_mixed_frame
from fmlib.feature_selection.utils.task_runtime import TaskRuntime

CATEGORICAL_MODES = frozenset(
    {
        "ordinal_campaign",
        "max_cardinality",
        "top_n",
        "target_encoding",
        "skip",
        "native",
    },
)

_MISSING = "__FMLIB_MISSING__"
_OTHER = "__FMLIB_OTHER__"


@dataclass(frozen=True)
class CategoricalHandling:
    """Resolved categorical handling settings for one model step."""

    mode: str = "native"
    max_cardinality: int | None = None
    top_n: int | None = None
    target_encoding_folds: int = 5
    target_encoding_smoothing: float = 20.0


@dataclass
class CategoricalSample:
    """Bounded local frame plus categorical candidates eligible for a model."""

    frame: pd.DataFrame
    feature_cols: list[str]
    categorical_cols: list[str]
    dropped_cardinality: dict[str, int]
    cardinality: dict[str, int]


@dataclass
class EncodedCategoricalFrame:
    """Temporary model matrix and the mapping back to source features."""

    features: pd.DataFrame
    model_features: list[str]
    categorical_model_features: list[str]
    source_by_model_feature: dict[str, str]


def resolve_categorical_handling(
    params: Mapping[str, Any],
    *,
    method_name: str,
) -> CategoricalHandling:
    """Parse the per-method ``categorical_handling`` mapping.

    Missing configuration intentionally defaults to native categories.  The
    validation in :mod:`fmlib.feature_selection.config` catches YAML mistakes;
    this runtime check keeps direct Python construction safe as well.
    """
    raw = params.get("categorical_handling", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        msg = f"{method_name}: params.categorical_handling must be a mapping."
        raise ExecutionError(msg)
    mode = str(raw.get("mode", "native")).lower()
    if mode not in CATEGORICAL_MODES:
        msg = f"{method_name}: unsupported categorical_handling.mode={mode!r}."
        raise ExecutionError(msg)
    te = raw.get("target_encoding", {})
    if te is None:
        te = {}
    if not isinstance(te, Mapping):
        msg = f"{method_name}: categorical_handling.target_encoding must be a mapping."
        raise ExecutionError(msg)
    max_cardinality = raw.get("max_cardinality")
    top_n = raw.get("top_n")
    folds = te.get("folds", 5)
    smoothing = te.get("smoothing", 20.0)
    for name, value in (("max_cardinality", max_cardinality), ("top_n", top_n)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            msg = f"{method_name}: categorical_handling.{name} must be a positive integer."
            raise ExecutionError(msg)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        msg = f"{method_name}: categorical_handling.target_encoding.folds must be at least 2."
        raise ExecutionError(msg)
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or float(smoothing) < 0:
        msg = f"{method_name}: categorical_handling.target_encoding.smoothing must be non-negative."
        raise ExecutionError(msg)
    if mode == "max_cardinality" and max_cardinality is None:
        msg = f"{method_name}: categorical_handling.max_cardinality is required for mode='max_cardinality'."
        raise ExecutionError(msg)
    if mode == "top_n" and top_n is None:
        msg = f"{method_name}: categorical_handling.top_n is required for mode='top_n'."
        raise ExecutionError(msg)
    return CategoricalHandling(
        mode=mode,
        max_cardinality=max_cardinality,
        top_n=top_n,
        target_encoding_folds=folds,
        target_encoding_smoothing=float(smoothing),
    )


def prepare_categorical_sample(
    data: Any,
    *,
    target_col: str,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    extra_cols: Sequence[str] = (),
    max_rows: int,
    sample_fraction: float | None,
    seed: int,
    method_name: str,
    handling: CategoricalHandling,
    context: Any | None = None,
) -> CategoricalSample:
    """Materialize a bounded mixed frame and apply the cardinality gate."""
    features = list(feature_cols)
    categorical_set = set(categorical_cols)
    cats = [name for name in features if name in categorical_set]
    frame = prepare_mixed_frame(
        data,
        target_col=target_col,
        feature_cols=features,
        categorical_cols=cats,
        extra_cols=extra_cols,
        max_rows=max_rows,
        sample_fraction=sample_fraction,
        seed=seed,
        method_name=method_name,
        context=context,
    )
    cardinality = {
        name: int(_normalise_category(frame[name]).nunique(dropna=False))
        for name in cats
    }
    dropped: dict[str, int] = {}
    if handling.mode == "max_cardinality":
        assert handling.max_cardinality is not None
        dropped = {
            name: value
            for name, value in cardinality.items()
            if value > handling.max_cardinality
        }
        features = [name for name in features if name not in dropped]
        cats = [name for name in cats if name not in dropped]
    return CategoricalSample(
        frame=frame,
        feature_cols=features,
        categorical_cols=cats,
        dropped_cardinality=dropped,
        cardinality=cardinality,
    )


def encode_categorical_frame(
    sample: CategoricalSample,
    *,
    target: np.ndarray,
    task: TaskRuntime,
    handling: CategoricalHandling,
    seed: int,
    train_indices: np.ndarray | None = None,
) -> EncodedCategoricalFrame:
    """Build a model matrix, fitting transformations only on ``train_indices``.

    Rows outside ``train_indices`` are transformed with mappings learned from
    train rows.  For target encoding, train rows get OOF values and outside
    rows use the mapping fit on all train rows.
    """
    frame = sample.frame
    n_rows = len(frame)
    train = (
        np.arange(n_rows, dtype=np.int64)
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64)
    )
    train_set = set(train.tolist())
    source_by_model: dict[str, str] = {}
    parts: list[pd.DataFrame] = []
    numeric = [name for name in sample.feature_cols if name not in set(sample.categorical_cols)]
    if numeric:
        numeric_frame = frame.loc[:, numeric].copy()
        parts.append(numeric_frame)
        source_by_model.update({name: name for name in numeric})

    if handling.mode == "skip":
        pass
    elif handling.mode == "target_encoding":
        encoded, mapping = _target_encode(
            frame,
            categorical_cols=sample.categorical_cols,
            target=np.asarray(target),
            task=task,
            train_indices=train,
            folds=handling.target_encoding_folds,
            smoothing=handling.target_encoding_smoothing,
            seed=seed,
        )
        parts.append(encoded)
        source_by_model.update(mapping)
    else:
        transformed = pd.DataFrame(index=frame.index)
        for column in sample.categorical_cols:
            values = _normalise_category(frame[column])
            fitted = values.iloc[train]
            if handling.mode == "ordinal_campaign":
                levels = sorted(fitted.drop_duplicates().tolist())
                codes = {value: float(index) for index, value in enumerate(levels)}
                transformed[column] = values.map(codes).astype(float)
            elif handling.mode == "top_n":
                assert handling.top_n is not None
                counts = fitted.value_counts(dropna=False)
                levels = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), str(value)))
                kept = set(levels[: handling.top_n])
                transformed[column] = values.where(values.isin(kept), _OTHER)
                transformed[column] = pd.Categorical(transformed[column], categories=[*levels[: handling.top_n], _OTHER])
            else:  # native and max_cardinality
                levels = sorted(fitted.drop_duplicates().tolist())
                transformed[column] = pd.Categorical(values, categories=levels)
        parts.append(transformed)
        source_by_model.update({name: name for name in sample.categorical_cols})

    features = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
    model_features = list(features.columns)
    categorical_model = [
        name
        for name in model_features
        if isinstance(features[name].dtype, pd.CategoricalDtype)
    ]
    # Keep this assertion explicit: it detects an accidental target/feature
    # position mismatch before it reaches a third-party model.
    if len(features) != n_rows or not train_set.issubset(set(range(n_rows))):
        msg = "categorical handling produced an invalid row mapping."
        raise ExecutionError(msg)
    return EncodedCategoricalFrame(
        features=features,
        model_features=model_features,
        categorical_model_features=categorical_model,
        source_by_model_feature=source_by_model,
    )


def _normalise_category(values: pd.Series) -> pd.Series:
    """Represent null as one deterministic, model-safe categorical level."""
    return values.astype("string").fillna(_MISSING).astype(str)


def _target_encode(
    frame: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    target: np.ndarray,
    task: TaskRuntime,
    train_indices: np.ndarray,
    folds: int,
    smoothing: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return leakage-safe OOF encodings and their source-column mapping."""
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as exc:
        msg = "target_encoding: scikit-learn is required."
        raise ExecutionError(msg) from exc

    train_target = np.asarray(target)[train_indices]
    if task.is_regression:
        effective_folds = min(folds, len(train_indices))
        if effective_folds < 2:
            msg = "target_encoding: at least two train rows are required."
            raise ExecutionError(msg)
        splitter: Any = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(train_indices))
    else:
        _, counts = np.unique(train_target, return_counts=True)
        effective_folds = min(folds, int(counts.min()))
        if effective_folds < 2:
            msg = "target_encoding: every target class needs at least two train rows."
            raise ExecutionError(msg)
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(train_indices, train_target))

    encoded = pd.DataFrame(index=frame.index)
    source_by_model: dict[str, str] = {}
    class_count = 1 if task.is_regression or task.is_binary else int(task.n_classes or 0)
    for column in categorical_cols:
        values = _normalise_category(frame[column])
        names = (
            [f"__fmlib_te__{column}"]
            if class_count == 1
            else [f"__fmlib_te__{column}__class_{index}" for index in range(class_count)]
        )
        result = np.full((len(frame), class_count), np.nan, dtype=float)
        for train_part_pos, valid_part_pos in splits:
            fit_rows = train_indices[np.asarray(train_part_pos, dtype=np.int64)]
            valid_rows = train_indices[np.asarray(valid_part_pos, dtype=np.int64)]
            result[valid_rows] = _encode_with_mapping(
                values,
                target,
                fit_rows,
                valid_rows,
                task,
                smoothing,
                class_count,
            )
        outside = np.setdiff1d(np.arange(len(frame)), train_indices, assume_unique=False)
        if len(outside):
            result[outside] = _encode_with_mapping(
                values,
                target,
                train_indices,
                outside,
                task,
                smoothing,
                class_count,
            )
        encoded.loc[:, names] = result
        source_by_model.update({name: column for name in names})
    return encoded, source_by_model


def _encode_with_mapping(
    values: pd.Series,
    target: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    task: TaskRuntime,
    smoothing: float,
    class_count: int,
) -> np.ndarray:
    """Encode ``apply_rows`` from category statistics of ``fit_rows`` only."""
    fit_values = values.iloc[fit_rows].to_numpy()
    fit_target = np.asarray(target)[fit_rows]
    apply_values = values.iloc[apply_rows].to_numpy()
    result = np.empty((len(apply_rows), class_count), dtype=float)
    if task.is_regression:
        prior = float(np.mean(fit_target))
        stats = pd.DataFrame({"value": fit_values, "target": fit_target}).groupby("value")["target"].agg(["sum", "count"])
        mapped = ((stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)).to_dict()
        result[:, 0] = [float(mapped.get(value, prior)) for value in apply_values]
        return result
    if task.is_binary:
        classes = np.unique(np.asarray(target))
        positive = classes[-1]
        binary = (fit_target == positive).astype(float)
        prior = float(binary.mean())
        stats = pd.DataFrame({"value": fit_values, "target": binary}).groupby("value")["target"].agg(["sum", "count"])
        mapped = ((stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)).to_dict()
        result[:, 0] = [float(mapped.get(value, prior)) for value in apply_values]
        return result
    for class_index in range(class_count):
        binary = (fit_target == class_index).astype(float)
        prior = float(binary.mean())
        stats = pd.DataFrame({"value": fit_values, "target": binary}).groupby("value")["target"].agg(["sum", "count"])
        mapped = ((stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)).to_dict()
        result[:, class_index] = [float(mapped.get(value, prior)) for value in apply_values]
    return result
