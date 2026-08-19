from typing import Callable

from fmlib.data.io import general_collate
from fmlib.data.io.partitioning.replicas import ReplicasInfo, ReplicasInfoProtocol

from .batches import GeneralCollateFn

DEFAULT_COLLATE_FN: GeneralCollateFn = general_collate

DEFAULT_MASK_POSTFIX: str = "_mask"


def default_make_mask_name(postfix: str) -> Callable[[str], str]:
    def function(name: str) -> str:
        return f"{name}{postfix}"

    return function


DEFAULT_MAKE_MASK_NAME: Callable[[str], str] = default_make_mask_name(DEFAULT_MASK_POSTFIX)

DEFAULT_REPLICAS_INFO: ReplicasInfoProtocol = ReplicasInfo()
DEFAULT_WRITE_EVERY_N_BATCH: int = 8

DEFAULT_FRAGMENT_POSTFIX: str = ""
DEFAULT_TEMPLATE_FRAGMENT_NAME: str = "part_{part:06d}_replica_{replica:06d}{postfix}.parquet"

DEFAULT_CHECKPOINT_NAME: str = "model.safetensors"
DEFAULT_TEMPLATE_CHECKPOINT_DIR: str = "checkpoint_{epoch:06d}"
