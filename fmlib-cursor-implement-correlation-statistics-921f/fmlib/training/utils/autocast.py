import warnings
from contextlib import contextmanager

import accelerate


@contextmanager
def autocast(accelerator: accelerate.Accelerator, autocast_handler: accelerate.AutocastKwargs = None):
    with accelerator.autocast(autocast_handler):
        yield


@contextmanager
def no_autocast(accelerator: accelerate.Accelerator, autocast_handler: accelerate.AutocastKwargs = None):
    if autocast_handler is not None:
        warnings.warn("Autocast handler is not None.", stacklevel=2)
    yield


def make_autocast_wrapper(do_autocast: bool = False):
    """
    Создает контекстный менеджер для использования `autocast` или `no_autocast`.
    """
    return autocast if do_autocast else no_autocast
