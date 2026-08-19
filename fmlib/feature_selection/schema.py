"""Feature schema describing candidates and service columns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from fmlib.feature_selection.exceptions import SchemaError

TASK_TYPES = frozenset({"binary_classification", "classification", "regression"})
SPLIT_VALUES = frozenset({"train", "valid", "test"})


@dataclass(frozen=True)
class FeatureSchema:
    """Description of feature roles for the selection pipeline.

    Feature roles are separated from the physical DataFrame schema. Candidates
    are ``categorical`` and ``continuous`` only; target and service columns are
    never selection candidates.

    Args:
        categorical: Categorical feature column names.
        continuous: Continuous feature column names.
        target: Target column name (required for model and precise stages).
        task_type: ``binary_classification``, ``classification``, or ``regression``.
        time: Optional time column for month-over-month PSI / time-based CV.
        split: Optional split column with values ``train`` / ``valid`` / ``test``.
        fold: Optional CV fold column inside the train split.
        id_columns: Identifier columns excluded from selection.
    """

    categorical: tuple[str, ...]
    continuous: tuple[str, ...]
    target: str
    task_type: str
    time: Optional[str] = None
    split: Optional[str] = None
    fold: Optional[str] = None
    id_columns: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self: FeatureSchema) -> None:
        object.__setattr__(self, "categorical", tuple(self.categorical))
        object.__setattr__(self, "continuous", tuple(self.continuous))
        object.__setattr__(self, "id_columns", tuple(self.id_columns))
        self._validate_internal()

    def _validate_internal(self: FeatureSchema) -> None:
        if self.task_type not in TASK_TYPES:
            message = f"Unsupported task_type={self.task_type!r}. Expected one of: {sorted(TASK_TYPES)}."
            raise SchemaError(message)

        categorical_set = set(self.categorical)
        continuous_set = set(self.continuous)
        overlap = categorical_set & continuous_set
        if overlap:
            message = f"categorical and continuous must be disjoint; overlap={sorted(overlap)}."
            raise SchemaError(message)

        if len(self.categorical) != len(categorical_set):
            msg = "categorical contains duplicate column names."
            raise SchemaError(msg)
        if len(self.continuous) != len(continuous_set):
            msg = "continuous contains duplicate column names."
            raise SchemaError(msg)
        if len(self.id_columns) != len(set(self.id_columns)):
            msg = "id_columns contains duplicate column names."
            raise SchemaError(msg)

        candidates = set(self.candidate_features())
        if self.target in candidates:
            msg = "target must not appear in categorical or continuous."
            raise SchemaError(msg)

        service = set(self.service_columns()) - {self.target}
        leaked = candidates & service
        if leaked:
            message = (
                f"Service columns must not appear among candidates: {sorted(leaked)}. Remove them from categorical/continuous."
            )
            raise SchemaError(message)

    def candidate_features(self: FeatureSchema) -> list[str]:
        """Return selection candidates in stable order (categorical then continuous).

        Returns:
            Ordered list of candidate feature names.
        """
        return list(self.categorical) + list(self.continuous)

    def service_columns(self: FeatureSchema) -> list[str]:
        """Return non-candidate service columns that should be preserved on apply.

        Returns:
            Ordered unique service column names (target, time, split, fold, ids).
        """
        columns: list[str] = [self.target]
        for name in (self.time, self.split, self.fold):
            if name is not None:
                columns.append(name)
        columns.extend(self.id_columns)
        seen: set[str] = set()
        ordered: list[str] = []
        for name in columns:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def all_declared_columns(self: FeatureSchema) -> list[str]:
        """Return every column declared by the schema.

        Returns:
            Ordered unique column names from candidates and service fields.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for name in self.candidate_features() + self.service_columns():
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def validate_against_columns(
        self: FeatureSchema,
        columns: Sequence[str],
        *,
        require_split: bool = False,
    ) -> None:
        """Validate that declared columns exist in a DataFrame column list.

        Args:
            columns: Physical column names available on the input.
            require_split: When True, ``split`` must be set and present.

        Raises:
            SchemaError: If required columns are missing or split is required but absent.
        """
        available = set(columns)
        missing = [name for name in self.all_declared_columns() if name not in available]

        if missing:
            message = f"Declared columns missing from DataFrame: {missing}. Align FeatureSchema with the input schema."
            raise SchemaError(message)
            
        if require_split:
            if self.split is None:
                msg = (
                    "FeatureSchema.split is required when a single DataFrame is passed. "
                    "Set split or pass datasets={'train': ...} instead."
                )
                raise SchemaError(
                    msg,
                )
            if self.split not in available:
                msg = f"split column {self.split!r} is missing from DataFrame."
                raise SchemaError(msg)

    def to_dict(self: FeatureSchema) -> dict:
        """Serialize schema to a plain dictionary.

        Returns:
            JSON-compatible dictionary.
        """
        return {
            "categorical": list(self.categorical),
            "continuous": list(self.continuous),
            "target": self.target,
            "task_type": self.task_type,
            "time": self.time,
            "split": self.split,
            "fold": self.fold,
            "id_columns": list(self.id_columns),
        }

    @classmethod
    def from_dict(cls: type[FeatureSchema], payload: dict) -> FeatureSchema:
        """Build schema from a dictionary.

        Args:
            payload: Mapping with schema fields.

        Returns:
            Validated ``FeatureSchema`` instance.
        """
        return cls(
            categorical=tuple(payload.get("categorical", ())),
            continuous=tuple(payload.get("continuous", ())),
            target=payload["target"],
            task_type=payload["task_type"],
            time=payload.get("time"),
            split=payload.get("split"),
            fold=payload.get("fold"),
            id_columns=tuple(payload.get("id_columns", ())),
        )


def ensure_no_feature_leak(
    candidates: Iterable[str],
    schema: FeatureSchema,
) -> None:
    """Assert candidates do not include target or service columns.

    Args:
        candidates: Candidate feature names.
        schema: Feature schema.

    Raises:
        SchemaError: If a service/target column leaked into candidates.
    """
    forbidden = set(schema.service_columns())
    leaked = sorted(set(candidates) & forbidden)
    if leaked:
        message = f"Candidates include forbidden service columns: {leaked}."
        raise SchemaError(message)
