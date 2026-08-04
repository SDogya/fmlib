from fmlib.utils.deprecation import deprecation_warning

from .dlt import make_for_dlt
from .feature_transformer import make_for_feature_transformer
from .legacy import SequentialTransform, TabularTransform, TrivanTransform, UpliftTabularTransform
from .modular import ChangePaddingTransform as ModularChangePaddingTransform
from .sequence_representation import make_for_sequence_representation
from .uplift_feature_transformer import make_for_uplift_feature_transformer

ChangePaddingTransform = deprecation_warning()(ModularChangePaddingTransform)

__all__ = [
    "ChangePaddingTransform",
    "SequentialTransform",
    "TabularTransform",
    "TrivanTransform",
    "UpliftTabularTransform",
    "make_for_dlt",
    "make_for_feature_transformer",
    "make_for_sequence_representation",
    "make_for_uplift_feature_transformer",
]
