"""Pipeline steps for pre-statistics drop and sample helpers."""

from __future__ import annotations

from typing import Any, Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    StageContext,
    apply_drop_decisions,
    derive_seed,
    step_seed,
)
from fmlib.feature_selection.config import (
    FeatureDropConfig,
    FeatureSelectionConfig,
    RandomFeatureDropConfig,
    RowSampleConfig,
)
from fmlib.feature_selection.debug import debug_span
from fmlib.feature_selection.utils.feature_drop import apply_feature_drop_file
from fmlib.feature_selection.utils.preprocessing import (
    apply_random_feature_drop,
    apply_row_sample,
)

_PREPROCESSING_STAGE = "preprocessing"


class FeatureDropStep:
    """Drop columns listed in a text file and update schema."""

    method_name = "feature_drop"

    def __init__(self: FeatureDropStep, settings: FeatureDropConfig) -> None:
        self.settings = settings

    def run(
        self: FeatureDropStep,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[str]:
        feature_drop = self.settings
        debug = context.debug
        start: dict[str, Any] = {}
        if debug.enabled(self.method_name):
            start = {
                "n_candidates_in": len(candidates),
                "strict": feature_drop.strict,
                "step_index": context.step_index,
                "datasets": debug.snapshot_datasets(context.datasets, count_rows=False),
            }
        with debug_span(debug, self.method_name, **start) as span:
            datasets, schema, report = apply_feature_drop_file(
                context.datasets,
                context.schema,
                feature_drop.path or "",
                strict=feature_drop.strict,
            )
            context.datasets = datasets
            context.schema = schema
            if span is not None:
                span.update(
                    n_requested=len(report.requested),
                    n_dropped=len(report.dropped),
                    n_unknown=len(report.unknown),
                    n_candidates_out=len(schema.candidate_features()),
                    datasets=debug.snapshot_datasets(context.datasets, count_rows=False),
                )
        decisions = [
            FeatureDecision(
                feature=feature,
                stage=_PREPROCESSING_STAGE,
                method=self.method_name,
                reason="listed_in_drop_file",
                keep=False,
            )
            for feature in report.dropped
        ]
        context.decisions.extend(decisions)
        remaining = apply_drop_decisions(candidates, decisions)
        remaining = [
            name for name in remaining if name in schema.candidate_features()
        ]
        context.candidates = remaining
        context.scores[self.method_name] = {
            "path": report.path,
            "requested": list(report.requested),
            "dropped": list(report.dropped),
            "unknown": list(report.unknown),
            "strict": feature_drop.strict,
        }
        return remaining


class RandomFeatureDropStep:
    """Drop a deterministic random subset of schema candidates."""

    method_name = "random_feature_drop"

    def __init__(
        self: RandomFeatureDropStep,
        settings: RandomFeatureDropConfig,
    ) -> None:
        self.settings = settings

    def run(
        self: RandomFeatureDropStep,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[str]:
        random_drop = self.settings
        debug = context.debug
        start: dict[str, Any] = {}
        if debug.enabled(self.method_name):
            start = {
                "n_candidates_in": len(candidates),
                "n_features": random_drop.n_features,
                "step_index": context.step_index,
                "datasets": debug.snapshot_datasets(context.datasets, count_rows=False),
            }
        used_seed = step_seed(context)
        with debug_span(debug, self.method_name, **start) as span:
            datasets, schema, report = apply_random_feature_drop(
                context.datasets,
                context.schema,
                n_features=random_drop.n_features,
                seed=used_seed,
            )
            context.datasets = datasets
            context.schema = schema
            if span is not None:
                span.update(
                    n_dropped=len(report.dropped),
                    n_candidates_out=len(schema.candidate_features()),
                    datasets=debug.snapshot_datasets(context.datasets, count_rows=False),
                )
        decisions = [
            FeatureDecision(
                feature=feature,
                stage=_PREPROCESSING_STAGE,
                method=self.method_name,
                reason="random_test_drop",
                keep=False,
            )
            for feature in report.dropped
        ]
        context.decisions.extend(decisions)
        remaining = apply_drop_decisions(candidates, decisions)
        remaining = [
            name for name in remaining if name in schema.candidate_features()
        ]
        context.candidates = remaining
        context.scores[self.method_name] = {
            "n_features": random_drop.n_features,
            "dropped": list(report.dropped),
            "seed": derive_seed(used_seed, "preprocessing", "random_feature_drop"),
        }
        return remaining


class RowSampleStep:
    """Cap row counts on every split; does not drop features."""

    method_name = "row_sample"

    def __init__(self: RowSampleStep, settings: RowSampleConfig) -> None:
        self.settings = settings

    def run(
        self: RowSampleStep,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[str]:
        row_sample = self.settings
        debug = context.debug
        start: dict[str, Any] = {}
        if debug.enabled(self.method_name):
            start = {
                "max_rows": row_sample.max_rows,
                "stratified": row_sample.stratified,
                "step_index": context.step_index,
                "datasets": debug.snapshot_datasets(context.datasets, count_rows=False),
            }
        used_seed = step_seed(context)
        with debug_span(debug, self.method_name, **start) as span:
            datasets, report = apply_row_sample(
                context.datasets,
                context.schema,
                max_rows=row_sample.max_rows or 0,
                stratified=row_sample.stratified,
                seed=used_seed,
            )
            context.datasets = datasets
            if span is not None:
                span.update(
                    max_rows=report.max_rows,
                    stratified=report.stratified,
                    splits={
                        split.split: {
                            "seed": split.seed,
                            "original_rows": split.original_rows,
                            "sampled_rows": split.sampled_rows,
                        }
                        for split in report.splits
                    },
                    datasets=debug.snapshot_datasets(context.datasets, count_rows=False),
                )
        context.scores[self.method_name] = {
            "max_rows": report.max_rows,
            "stratified": report.stratified,
            "splits": {
                split.split: {
                    "seed": split.seed,
                    "original_rows": split.original_rows,
                    "sampled_rows": split.sampled_rows,
                }
                for split in report.splits
            },
        }
        context.candidates = list(candidates)
        return list(candidates)


def build_utils_steps(config: FeatureSelectionConfig) -> list[Any]:
    """Enabled utils in fixed order: feature_drop, random_feature_drop, row_sample."""
    preprocessing = config.preprocessing
    steps: list[Any] = []
    if preprocessing.feature_drop.enabled:
        steps.append(FeatureDropStep(preprocessing.feature_drop))
    if preprocessing.random_feature_drop.enabled:
        steps.append(RandomFeatureDropStep(preprocessing.random_feature_drop))
    if preprocessing.row_sample.enabled:
        steps.append(RowSampleStep(preprocessing.row_sample))
    return steps
