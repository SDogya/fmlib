import warnings
from contextlib import contextmanager

import torch


@contextmanager
def mode_context(model: torch.nn.Module, training: bool = False, verbose: bool = True):
    was_training: bool = model.training
    if verbose and was_training != training:
        warnings.warn(
            """Changing models training mode from """
            f"""{was_training} to {training}.""",
            stacklevel=2,
        )
    model.train(training)
    assert model.training == training

    yield

    if model.training != training:
        msg: str = (
            """Model training mode changed """
            """during context manager."""
        )
        raise RuntimeError(msg)

    model.train(was_training)
    assert model.training == was_training
