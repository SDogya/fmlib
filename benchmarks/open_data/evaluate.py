"""Downstream scoring: is the selected feature set actually as good?

A feature-selection run is only meaningful against what happens next. For every
candidate feature set this trains the same LightGBM with the same seed and
reports ROC-AUC on the held-out ``test`` split -- the split the pipeline never
reads -- so the number answers "did we lose anything by dropping 90% of the
columns", not "did the filters agree with each other".
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass
class Evaluation:
    """One trained downstream model.

    Args:
        label: Which feature set this was ("baseline" or a pipeline name).
        n_features: How many features the model saw.
        auc_valid: ROC-AUC on the validation split (used for early stopping).
        auc_test: ROC-AUC on the held-out test split -- the headline number.
        best_iteration: Trees kept after early stopping.
        fit_seconds: Wall-clock training time.
    """

    label: str
    n_features: int
    auc_valid: float
    auc_test: float
    best_iteration: int
    fit_seconds: float

    def to_dict(self: Evaluation) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)


def evaluate_feature_set(
    frames: dict[str, pd.DataFrame],
    features: Sequence[str],
    *,
    target: str,
    categorical: Sequence[str],
    label: str,
    seed: int = 42,
    n_jobs: int = 8,
    params: Optional[dict[str, Any]] = None,
) -> Evaluation:
    """Train LightGBM on ``features`` and score it on valid and test.

    Args:
        frames: ``train`` / ``valid`` / ``test`` frames sharing one schema.
        features: Columns the model is allowed to use.
        target: Target column name.
        categorical: Which of ``features`` are categorical.
        label: Name recorded on the result.
        seed: Model seed.
        n_jobs: Thread count, held at the benchmark's 8-core budget.
        params: Overrides for the default LightGBM parameters.

    Returns:
        The fitted model's ``Evaluation``.

    Raises:
        ValueError: If ``features`` is empty -- an empty selection has no AUC.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    features = list(features)
    if not features:
        msg = f"{label}: cannot evaluate an empty feature set."
        raise ValueError(msg)

    categorical_used = [name for name in categorical if name in set(features)]
    prepared = {
        split: _prepare(frame, features, categorical_used)
        for split, frame in frames.items()
    }

    model_params = {**_DEFAULT_PARAMS, **(params or {})}
    model_params.update({"random_state": seed, "n_jobs": n_jobs})

    started = time.perf_counter()
    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        prepared["train"],
        frames["train"][target],
        eval_set=[(prepared["valid"], frames["valid"][target])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=categorical_used or "auto",
    )
    fit_seconds = time.perf_counter() - started

    scores = {
        split: float(
            roc_auc_score(
                frames[split][target],
                model.predict_proba(prepared[split])[:, 1],
            ),
        )
        for split in ("valid", "test")
    }
    evaluation = Evaluation(
        label=label,
        n_features=len(features),
        auc_valid=round(scores["valid"], 5),
        auc_test=round(scores["test"], 5),
        best_iteration=int(getattr(model, "best_iteration_", 0) or 0),
        fit_seconds=round(fit_seconds, 2),
    )
    logger.info(
        "evaluated %s: %d features, test AUC %.5f (%.1fs)",
        label,
        evaluation.n_features,
        evaluation.auc_test,
        evaluation.fit_seconds,
    )
    return evaluation


def _prepare(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    """Project to ``features`` and give LightGBM the dtypes it can use natively."""
    prepared = frame.loc[:, list(features)].copy()
    categorical_set = set(categorical)
    for name in prepared.columns:
        if name in categorical_set:
            prepared[name] = prepared[name].astype("category")
        elif not pd.api.types.is_numeric_dtype(prepared[name]):
            prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    return prepared


def selection_quality(
    selected: Sequence[str],
    informative: Sequence[str],
    drifting: Sequence[str],
) -> dict[str, Any]:
    """Score a selection against known ground truth.

    Only meaningful for the synthetic dataset, where the informative and
    drifting columns are named by construction.

    Args:
        selected: Features the pipeline kept.
        informative: Features that genuinely carry signal.
        drifting: Features whose distribution shifts across months.

    Returns:
        Recall of the informative set, precision of the selection against it,
        and how many drifting features survived (fewer is better).
    """
    if not informative:
        return {}
    selected_set = set(selected)
    informative_set = set(informative)
    hits = len(selected_set & informative_set)
    return {
        "informative_recall": round(hits / len(informative_set), 4),
        "informative_precision": round(
            hits / len(selected_set) if selected_set else 0.0,
            4,
        ),
        "drifting_kept": len(selected_set & set(drifting)),
        "drifting_total": len(drifting),
    }


def gini(auc: float) -> float:
    """Gini coefficient, the form scorecard reviews usually ask for."""
    return round(2.0 * auc - 1.0, 5)


def summarize(evaluations: Sequence[Evaluation]) -> pd.DataFrame:
    """Tabulate evaluations with the delta against the baseline."""
    frame = pd.DataFrame([item.to_dict() for item in evaluations])
    if frame.empty:
        return frame
    baseline = frame.loc[frame["label"] == "baseline", "auc_test"]
    if not baseline.empty:
        frame["auc_test_delta"] = (frame["auc_test"] - float(baseline.iloc[0])).round(5)
    frame["gini_test"] = frame["auc_test"].map(gini)
    return frame
