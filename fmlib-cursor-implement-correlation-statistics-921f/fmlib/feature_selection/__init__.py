"""Feature selection package public API."""

from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.pipeline import FeatureSelectionPipeline
from fmlib.feature_selection.result import SelectionResult
from fmlib.feature_selection.schema import FeatureSchema

__all__ = [
    "FeatureSchema",
    "FeatureSelectionConfig",
    "FeatureSelectionPipeline",
    "SelectionResult",
]
