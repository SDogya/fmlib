"""Shared contracts and stub helpers for feature selection stages."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from fmlib.feature_selection.utils.verbose import VerboseRecorder, default_verbose_recorder


@dataclass(frozen=True)
class FeatureDecision:
    """Decision about a single feature produced by a selection stage.

    Args:
        feature: Feature name.
        stage: Pipeline stage name (``preprocessing``, ``statistics``,
            ``model``, or ``precise``).
        method: Selector method name within the stage.
        reason: Machine-readable reason code.
        value: Measured metric value, if any.
        threshold: Threshold used for the decision, if any.
        keep: Whether the feature remains a candidate after this decision.
    """

    feature: str
    stage: str
    method: str
    reason: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    keep: bool = False

    def to_dict(self: "FeatureDecision") -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "feature": self.feature,
            "stage": self.stage,
            "method": self.method,
            "reason": self.reason,
            "value": self.value,
            "threshold": self.threshold,
            "keep": self.keep,
        }

    @classmethod
    def from_dict(cls: type["FeatureDecision"], payload: dict[str, Any]) -> "FeatureDecision":
        """Build from a dictionary."""
        return cls(
            feature=payload["feature"],
            stage=payload["stage"],
            method=payload["method"],
            reason=payload["reason"],
            value=payload.get("value"),
            threshold=payload.get("threshold"),
            keep=payload.get("keep", False),
        )


@dataclass
class StageContext:
    """Runtime context passed to selectors.

    Args:
        spark: Active Spark session (duck-typed in the skeleton).
        datasets: Mapping of split name to DataFrame-like objects.
        schema: Validated feature schema.
        config: Validated pipeline configuration.
        seed: Root reproducibility seed.
        candidates: Current candidate feature names.
        scores: Mapping of selector method names to their importance scores.
        datasets_mode: Whether input was a single frame or split mapping.
        decisions: Decisions accumulated across completed pipeline stages.
        verbose_log: Per-method verbose recorder. Silent unless ``execution.verbose``.
        step_index: 0-based position of the current pipeline step.
        run_seed: Seed for this step. ``None`` uses ``seed``.
        output_dir: Optional directory for per-method intermediate artifacts.
        local_numeric_sample: Cached driver-local numeric frame shared by
            LightGBM and BorutaSHAP when sample knobs match.
    """

    spark: Any
    datasets: dict[str, Any]
    schema: Any
    config: Any
    seed: int
    candidates: list[str]
    scores: dict[str, Any] = field(default_factory=dict)
    datasets_mode: str = "mapping"
    decisions: list[FeatureDecision] = field(default_factory=list)
    verbose_log: VerboseRecorder = field(default_factory=default_verbose_recorder)
    step_index: int = 0
    run_seed: Optional[int] = None
    output_dir: Optional[Path] = None
    local_numeric_sample: Optional[Any] = None


def step_seed(context: StageContext) -> int:
    """Seed for the current pipeline step."""
    if context.run_seed is None:
        return context.seed
    return context.run_seed


def persist_step_artifact(
    context: StageContext,
    remaining: Sequence[str],
    *,
    stage_name: str,
    method_name: str,
) -> None:
    """Write a per-method artifact when ``context.output_dir`` is set."""
    output_dir = context.output_dir
    if output_dir is None:
        return
    from fmlib.feature_selection.result import _save_intermediate_result

    _save_intermediate_result(
        remaining=list(remaining),
        decisions=context.decisions,
        stage_name=stage_name,
        method_name=method_name,
        output_dir=output_dir,
        context=context,
    )


class Selector(Protocol):
    """Protocol for a single feature selection method."""

    method_name: str

    def select(
        self: Selector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Evaluate candidates and return keep/drop decisions.

        Args:
            context: Shared stage context.
            candidates: Features still under consideration.

        Returns:
            Decisions for dropped (and optionally kept) features. Dropped features
            must have ``keep=False`` and are removed before the next selector.
        """
        ...


def derive_seed(root_seed: int, *parts: str) -> int:
    """Derive a stable integer seed from a root seed and stage identifiers.

    Args:
        root_seed: Root reproducibility seed from execution config.
        *parts: Stage/method identifiers mixed into the seed.

    Returns:
        Deterministic 32-bit integer seed.
    """
    payload = f"{root_seed}:" + ":".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stub_drop_features(
    candidates: Sequence[str],
    *,
    seed: int,
    stage: str,
    method: str,
    n_min: int = 10,
    n_max: int = 20,
) -> list[FeatureDecision]:
    """Drop a random subset of candidate features (skeleton stub).

    Never drops all candidates when more than one remains. If there is a single
    candidate, returns an empty decision list (feature is kept).

    Args:
        candidates: Current candidate feature names.
        seed: Root seed; a derived seed is used for sampling.
        stage: Stage name recorded in decisions.
        method: Method name recorded in decisions.
        n_min: Minimum number of features to drop when enough candidates exist.
        n_max: Maximum number of features to drop.

    Returns:
        Drop decisions with ``reason=\"stub_random_drop\"``.
    """
    if len(candidates) <= 1:
        return []

    rng = random.Random(derive_seed(seed, stage, method))  # noqa: S311 - deterministic stub sampling
    max_drop = len(candidates) - 1
    lower = min(n_min, max_drop)
    upper = min(n_max, max_drop)
    if lower > upper:
        lower = upper
    k = rng.randint(lower, upper)
    to_drop = set(rng.sample(list(candidates), k=k))

    return [
        FeatureDecision(
            feature=name,
            stage=stage,
            method=method,
            reason="stub_random_drop",
            value=None,
            threshold=None,
            keep=False,
        )
        for name in candidates
        if name in to_drop
    ]


def apply_drop_decisions(
    candidates: Sequence[str],
    decisions: Sequence[FeatureDecision],
) -> list[str]:
    """Remove dropped features from candidates while preserving order.

    Args:
        candidates: Current candidate names.
        decisions: Decisions from a selector; only ``keep=False`` entries drop.

    Returns:
        Remaining candidates in original order.
    """
    dropped = {decision.feature for decision in decisions if not decision.keep}
    return [name for name in candidates if name not in dropped]
