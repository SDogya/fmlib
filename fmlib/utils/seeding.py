import warnings
from typing import Optional

from fmlib.constants.random import DEFAULT_SEED


def seed_numpy(seed: int) -> None:
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        warnings.warn("Unable to load numpy.", stacklevel=2)


def seed_torch(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        warnings.warn("Unable to load torch.", stacklevel=2)


def seed_python(seed: int) -> None:
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_accelerate(seed: int) -> None:
    try:
        from accelerate.utils import set_seed

        set_seed(seed)
    except ImportError:
        warnings.warn("Unable to load accelerate.", stacklevel=2)


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    seed_numpy(seed=seed)
    seed_torch(seed=seed)
    seed_python(seed=seed)
    seed_accelerate(seed=seed)


def force_seed_everything(seed: Optional[int]) -> None:
    """
    Заставляет проставить глобальные seed'ы.

    Аргументы:
        seed (int | None): seed для глобальных ГПСЧ.
            Если `None` - не инициализирует и пишет warning.
    """
    if seed:
        seed_everything(seed=seed)
    else:
        warnings.warn("NOT seeding everything.", stacklevel=2)
