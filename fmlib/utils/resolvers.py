import sys
import traceback as tb
import warnings
from typing import Any, Callable, Optional

import hydra.utils
from omegaconf import DictConfig, OmegaConf


def get_resolver_name(resolver: Callable, name: Optional[str]) -> str:
    return str(resolver.__name__) if (name is None) else name


def register_new_resolver(name: Optional[str] = None, **kwargs) -> Callable:
    def register_new_resolver_impl(resolver: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            resolver_name: str = get_resolver_name(resolver, name)
            try:
                result: Any = resolver(*args, **kwargs)
                return result
            except Exception as exc:
                msg: str = f'Call to resolver "{resolver_name=}" with {args=} and {kwargs=} failed.'
                warnings.warn(msg, stacklevel=2)
                tb.print_tb(exc.__traceback__)
                exc_info: Any = sys.exc_info()
                tb.print_exception(*exc_info)
                del exc_info
                raise exc

        resolver_name: str = get_resolver_name(resolver, name)
        OmegaConf.register_new_resolver(resolver_name, wrapper, **kwargs)
        return wrapper

    return register_new_resolver_impl


@register_new_resolver("instantiate")
def instantiate(config: DictConfig) -> Any:
    return hydra.utils.instantiate(config)
