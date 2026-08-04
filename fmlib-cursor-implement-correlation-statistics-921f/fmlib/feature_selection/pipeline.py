"""Feature selection pipeline facade."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from fmlib.feature_selection.backends.spark import (
    SPARK_CAPABILITIES,
    ensure_dataframe,
    ensure_spark_session,
    get_columns,
)
from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
from fmlib.feature_selection.model_based.stage import ModelBasedStage
from fmlib.feature_selection.precise.stage import PreciseStage
from fmlib.feature_selection.result import DroppedFeature, SelectionResult
from fmlib.feature_selection.schema import FeatureSchema, ensure_no_feature_leak
from fmlib.feature_selection.statistics.stage import StatisticsStage

_DATASET_KEYS = ("train", "valid", "test")


class FeatureSelectionPipeline:
    """Configurable multi-stage feature selection pipeline.

    The skeleton wires statistical, model-based and optional precise stages.
    Selector bodies currently drop a deterministic random subset of columns.

    Args:
        config: Validated pipeline configuration.
    """

    def __init__(self: FeatureSelectionPipeline, config: FeatureSelectionConfig) -> None:
        config.validate()
        self.config = config
        self._statistics = StatisticsStage()
        self._model = ModelBasedStage()
        self._precise = PreciseStage()

    def fit_select(
        self: FeatureSelectionPipeline,
        spark: Any,
        *,
        schema: FeatureSchema,
        data: Any = None,
        datasets: Optional[Mapping[str, Any]] = None,
    ) -> SelectionResult:
        """Fit selection stages and return a reproducible artifact.

        Pass either a single ``data`` DataFrame (with ``schema.split``) or a
        ``datasets`` mapping with a required ``train`` key. Inputs are not mutated.

        Args:
            spark: Active Spark session (duck-typed in the skeleton).
            schema: Feature role description.
            data: Single Spark DataFrame with an optional split column.
            datasets: Mapping of ``train`` / ``valid`` / ``test`` DataFrames.

        Returns:
            ``SelectionResult`` with selected/dropped features and metadata.

        Raises:
            ConfigError: On mutually exclusive or incomplete input forms.
            SchemaError: On invalid schema or missing columns.
            BackendError: When ``spark`` / DataFrames are not usable.
        """
        ensure_spark_session(spark)
        resolved, datasets_mode = self._resolve_datasets(data=data, datasets=datasets, schema=schema)
        self._validate_inputs(resolved, schema=schema, datasets_mode=datasets_mode)

        candidates = schema.candidate_features()
        ensure_no_feature_leak(candidates, schema)
        if not candidates:
            msg = "FeatureSchema has no candidate features. Provide non-empty categorical and/or continuous lists."
            raise SchemaError(
                msg,
            )

        seed = self.config.execution.seed
        context = StageContext(
            spark=spark,
            datasets=resolved,
            schema=schema,
            config=self.config,
            seed=seed,
            candidates=list(candidates),
        )

        all_decisions = []
        stage_backends = {
            "statistics": SPARK_CAPABILITIES.name,
            "model": SPARK_CAPABILITIES.name,
            "precise": SPARK_CAPABILITIES.name,
        }

        remaining, decisions = self._statistics.run(context, candidates)
        all_decisions.extend(decisions)

        remaining, decisions = self._model.run(context, remaining)
        all_decisions.extend(decisions)

        remaining, decisions = self._precise.run(context, remaining)
        all_decisions.extend(decisions)

        dropped = [
            DroppedFeature(
                feature=item.feature,
                stage=item.stage,
                method=item.method,
                reason=item.reason,
                value=item.value,
                threshold=item.threshold,
            )
            for item in all_decisions
            if not item.keep
        ]

        return SelectionResult(
            selected_features=list(remaining),
            dropped_features=dropped,
            schema=schema,
            config=self.config.to_dict(),
            seed=seed,
            stage_backends=stage_backends,
            warnings=["Selectors are stubs that randomly drop features; replace with real algorithms."],
            datasets_mode=datasets_mode,
        )

    def transform(self: FeatureSelectionPipeline, data: Any, result: SelectionResult) -> Any:
        """Apply a fitted ``SelectionResult`` to a DataFrame-like object.

        Args:
            data: Input DataFrame-like object.
            result: Previously fitted selection artifact.

        Returns:
            Projection onto selected and service columns.
        """
        return result.apply(data)

    @staticmethod
    def _resolve_datasets(
        *,
        data: Any,
        datasets: Optional[Mapping[str, Any]],
        schema: FeatureSchema,
    ) -> tuple[dict[str, Any], str]:
        if data is not None and datasets is not None:
            msg = (
                "Pass either data=... or datasets=..., not both. "
                "Use a single DataFrame with FeatureSchema.split, or a train/valid/test mapping."
            )
            raise ConfigError(
                msg,
            )
        if data is None and datasets is None:
            msg = "One of data=... or datasets=... is required."
            raise ConfigError(
                msg,
            )

        if data is not None:
            ensure_dataframe(data, name="data")
            if schema.split is None:
                msg = (
                    "FeatureSchema.split is required when passing a single data DataFrame. "
                    "Set split or use datasets={'train': ...}."
                )
                raise SchemaError(
                    msg,
                )
            return {"train": data}, "single"

        assert datasets is not None
        unknown = sorted(set(datasets) - set(_DATASET_KEYS))
        if unknown:
            msg = f"Unsupported datasets keys: {unknown}. Allowed: {list(_DATASET_KEYS)}."
            raise ConfigError(
                msg,
            )
        if "train" not in datasets:
            msg = "datasets must contain a 'train' DataFrame."
            raise ConfigError(msg)
        resolved: dict[str, Any] = {}
        for key in _DATASET_KEYS:
            if key in datasets:
                resolved[key] = ensure_dataframe(datasets[key], name=f"datasets[{key!r}]")
        return resolved, "mapping"

    @staticmethod
    def _validate_inputs(
        datasets: Mapping[str, Any],
        *,
        schema: FeatureSchema,
        datasets_mode: str,
    ) -> None:
        require_split = datasets_mode == "single"
        reference_columns = get_columns(datasets["train"])
        schema.validate_against_columns(reference_columns, require_split=require_split)

        for name, frame in datasets.items():
            columns = set(get_columns(frame))
            missing = [col for col in schema.all_declared_columns() if col not in columns]
            # split column is only required on the single-DF path
            if datasets_mode == "mapping":
                missing = [col for col in missing if col != schema.split]
            if missing:
                msg = f"datasets[{name!r}] is missing declared columns: {missing}. Align all splits to the same feature schema."
                raise SchemaError(
                    msg,
                )
