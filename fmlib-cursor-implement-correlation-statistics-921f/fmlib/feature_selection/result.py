"""Selection result artifact and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from fmlib.feature_selection.exceptions import SchemaError
from fmlib.feature_selection.schema import FeatureSchema

FORMAT_VERSION = 1


@dataclass(frozen=True)
class DroppedFeature:
    """Record of a feature excluded by the pipeline.

    Args:
        feature: Feature name.
        stage: Stage that dropped the feature.
        method: Method within the stage.
        reason: Machine-readable reason.
        value: Measured metric, if any.
        threshold: Threshold used, if any.
    """

    feature: str
    stage: str
    method: str
    reason: str
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self: DroppedFeature) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[DroppedFeature], payload: dict[str, Any]) -> DroppedFeature:
        """Build from a dictionary."""
        return cls(
            feature=payload["feature"],
            stage=payload["stage"],
            method=payload["method"],
            reason=payload["reason"],
            value=payload.get("value"),
            threshold=payload.get("threshold"),
        )


@dataclass
class SelectionResult:
    """Artifact produced by ``FeatureSelectionPipeline.fit_select``.

    Args:
        selected_features: Features kept after all stages, in stable order.
        dropped_features: Dropped features with stage/method/reason metadata.
        schema: Input feature schema.
        config: Normalized config snapshot as a dictionary.
        seed: Root reproducibility seed.
        stage_backends: Backend name used per stage.
        warnings: Non-fatal diagnostics.
        format_version: Artifact format version.
        datasets_mode: ``single`` or ``mapping`` input mode.
        scores: Optional feature scores/importances by method.
    """

    selected_features: list[str]
    dropped_features: list[DroppedFeature]
    schema: FeatureSchema
    config: dict[str, Any]
    seed: int
    stage_backends: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    format_version: int = FORMAT_VERSION
    datasets_mode: str = "single"
    scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self: SelectionResult) -> dict[str, Any]:
        """Serialize the artifact to a JSON-compatible dictionary.

        Returns:
            Dictionary including ``format_version``.
        """
        return {
            "format_version": self.format_version,
            "selected_features": list(self.selected_features),
            "dropped_features": [item.to_dict() for item in self.dropped_features],
            "schema": self.schema.to_dict(),
            "config": self.config,
            "seed": self.seed,
            "stage_backends": dict(self.stage_backends),
            "warnings": list(self.warnings),
            "datasets_mode": self.datasets_mode,
            "scores": dict(self.scores),
        }

    def save(self: SelectionResult, path: Union[str, Path]) -> None:
        """Save the artifact as JSON.

        Args:
            path: Destination file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    @classmethod
    def load(cls: type[SelectionResult], path: Union[str, Path]) -> SelectionResult:
        """Load an artifact from JSON.

        Args:
            path: Source file path.

        Returns:
            Restored ``SelectionResult``.

        Raises:
            SchemaError: If ``format_version`` is unsupported.
        """
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls: type[SelectionResult], payload: dict[str, Any]) -> SelectionResult:
        """Build result from a dictionary.

        Args:
            payload: Serialized artifact.

        Returns:
            ``SelectionResult`` instance.
        """
        version = payload.get("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            message = f"Unsupported SelectionResult format_version={version}. Supported version: {FORMAT_VERSION}."
            raise SchemaError(message)
        return cls(
            selected_features=list(payload.get("selected_features", [])),
            dropped_features=[DroppedFeature.from_dict(item) for item in payload.get("dropped_features", [])],
            schema=FeatureSchema.from_dict(payload["schema"]),
            config=dict(payload.get("config", {})),
            seed=int(payload["seed"]),
            stage_backends=dict(payload.get("stage_backends", {})),
            warnings=list(payload.get("warnings", [])),
            format_version=version,
            datasets_mode=str(payload.get("datasets_mode", "single")),
            scores=dict(payload.get("scores", {})),
        )

    def columns_to_keep(self: SelectionResult) -> list[str]:
        """Return selected features plus schema service columns.

        Returns:
            Ordered unique column names to project on apply.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for name in list(self.selected_features) + self.schema.service_columns():
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def apply(self: SelectionResult, data: Any) -> Any:
        """Project ``data`` to selected and service columns.

        Supports Spark-like objects with ``select`` / ``columns`` and simple
        testing doubles that expose ``columns`` as a mutable sequence.

        Args:
            data: Input DataFrame-like object.

        Returns:
            Projected DataFrame-like object.

        Raises:
            SchemaError: If a selected feature is missing from ``data``.
        """
        keep = self.columns_to_keep()
        available = _get_columns(data)
        missing = [name for name in self.selected_features if name not in available]
        if missing:
            message = f"Cannot apply SelectionResult: selected features missing from data: {missing}."
            raise SchemaError(message)

        if hasattr(data, "select") and callable(data.select):
            return data.select(*[name for name in keep if name in available])

        # Testing double / list-columns fallback
        projected = [name for name in keep if name in available]
        if hasattr(data, "columns"):
            clone = type(data)(columns=projected) if _accepts_columns_kwarg(data) else data
            if clone is data and isinstance(getattr(data, "columns", None), list):
                data.columns = projected
                return data
            return clone
        return projected


def _get_columns(data: Any) -> Sequence[str]:
    columns = getattr(data, "columns", None)
    if columns is None:
        msg = "Data object has no columns attribute; cannot apply SelectionResult."
        raise SchemaError(msg)
    return list(columns)


def _accepts_columns_kwarg(data: Any) -> bool:
    try:
        type(data)(columns=["__probe__"])
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 - probe only
        return False
    return True
