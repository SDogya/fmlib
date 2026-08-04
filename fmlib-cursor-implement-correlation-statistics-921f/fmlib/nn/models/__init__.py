from fmlib.utils.deprecation import deprecation_warning

from .dlt import DLTBody, DLTClassification, DLTEmbedding, LegacyDLTmodelCLS
from .feature_transformer import FeatureTransformer, UpliftFeatureTransformer
from .ivan import IvanModel
from .sequence_classification import SequenceClassificationModel
from .sequence_representation import SequenceRepresentationModel
from .trivan import Trivan

DLTmodelCLS = deprecation_warning("Please use modular variant.")(LegacyDLTmodelCLS)

__all__ = [
    "DLTBody",
    "DLTClassification",
    "DLTEmbedding",
    "DLTmodelCLS",
    "FeatureTransformer",
    "IvanModel",
    "SequenceClassificationModel",
    "SequenceRepresentationModel",
    "Trivan",
    "UpliftFeatureTransformer",
]
