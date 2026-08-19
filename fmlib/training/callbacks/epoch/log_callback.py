from typing import Any, Dict, Self

import accelerate

from fmlib.training.training_loop import TrainingState


class LoggingEpochCallback:
    """
    Коллбек для выведения промежуточных результатов во время обучения.
    """

    def __init__(self: Self) -> None:
        pass

    def reset(self: Self) -> Self:
        return self

    def __call__(self: Self, state: TrainingState, metrics: Dict[str, Any], epoch: int = 0) -> Dict[str, Any]:
        accelerator: accelerate.Accelerator = state["accelerator"]
        accelerator.print(metrics)
        accelerator.log(metrics)
        return {}

    def finalize(self: Self, state: TrainingState) -> Dict[str, Any]:
        return {}
