import warnings
from typing import Callable, cast


def _get_class_name(obj: object | type) -> str:
    if isinstance(obj, type):
        return obj.__name__
    else:
        return obj.__class__.__name__


def deprecation_warning(message: str | None = None) -> Callable:
    """
    Wraps some callable entity to show a deprecation warning on usage.
    """

    def deprecation_warning_impl(callable_entity: Callable, message: str | None = message) -> Callable:
        class_name: str = _get_class_name(callable_entity)
        if message is None:
            message = "Calling to a deprecated entity: {class_name}."
        msg: str = cast(str, message.format(class_name=class_name))

        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return callable_entity(*args, **kwargs)

        return wrapper

    return deprecation_warning_impl
