from functools import lru_cache
from typing import Callable


def is_mask(name: str) -> bool:
    @lru_cache
    def _get_default_mask_postfix() -> str:
        from .io import DEFAULT_MASK_POSTFIX

        return DEFAULT_MASK_POSTFIX

    default_mask_postfix = _get_default_mask_postfix()
    return name.endswith(default_mask_postfix)


def get_mask_name(name: str) -> str:
    @lru_cache
    def _get_default_make_mask_name() -> Callable[[str], str]:
        from .io import DEFAULT_MAKE_MASK_NAME

        return DEFAULT_MAKE_MASK_NAME

    if is_mask(name):
        return name
    else:
        default_make_mask_name = _get_default_make_mask_name()
        return default_make_mask_name(name)


DEFAULT_GET_MASK_NAME: Callable[[str], str] = get_mask_name
