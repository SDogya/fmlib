import torch
from tqdm.auto import tqdm

from fmlib.training.training_types import EvaluationState, MetricsType
from fmlib.training.utils.exit import check_exit
from fmlib.utils.to_device import to_device


def evaluation_loop(state: EvaluationState) -> MetricsType:
    """
    Функция, отвечающая за шаг тестирования/инференса модели.
    """

    pipeline = state["pipeline"].eval()

    autocast = state["autocast"]
    accelerator = state["accelerator"]
    callbacks = state["evaluation_step_callbacks"].reset()

    data_iterable = tqdm(
        iterable=state["evaluation_data"],
        position=0,  # Second after epoch bar
        leave=False,  # Space needed for validation bar
        desc="Evaluation",  # Text for training bar
    )

    with torch.no_grad(), autocast(accelerator):
        for batch in data_iterable:
            device_batch = to_device(batch, accelerator.device)

            transformed, outputs = pipeline(device_batch)
            transformed, outputs = transformed[-1], outputs[-1]

            callbacks(
                state=state,
                outputs=outputs,
                transformed=transformed,
            )

            check_exit(accelerator)

    return callbacks.finalize(state)
