"""Exceptions for the feature selection module."""


class FeatureSelectionError(Exception):
    """Base error for the feature selection pipeline.

    Args:
        message: Human-readable description with a fixable action when possible.
    """

    def __init__(self: "FeatureSelectionError", message: str) -> None:
        super().__init__(message)
        self.message = message


class SchemaError(FeatureSelectionError):
    """Raised when FeatureSchema or input columns are invalid."""


class ConfigError(FeatureSelectionError):
    """Raised when FeatureSelectionConfig is invalid or incompatible."""


class CapacityError(FeatureSelectionError):
    """Raised when local materialization limits are exceeded."""


class BackendError(FeatureSelectionError):
    """Raised when a required backend or optional dependency is unavailable."""


class ExecutionError(FeatureSelectionError):
    """Raised when a stage fails during execution."""
