import accelerate
import torch
from tqdm.auto import tqdm

from fmlib.training.training_types import MetricsType, TrainingState
from fmlib.training.utils.exit import check_exit
from fmlib.training.utils.check_reset import check_reset_loss
from fmlib.utils.to_device import to_device
from fmlib.utils import metrics_dict


def training_loop(state: TrainingState) -> MetricsType:
    """
    Функция, обеспечиваюбщая общий цикл обучения модели.
    """
    accelerator: accelerate.Accelerator = state["accelerator"]
    callbacks = state["epoch_callbacks"].reset()

    epoch_iterable = tqdm(
        iterable=state["epoch_iterable"],
        position=0,  # Second after epoch bar
        leave=True,  # Space needed for validation bar
        desc="Epoch",  # Text for epoch bar
        disable=(not accelerator.is_main_process),
    )

    for epoch in epoch_iterable:
        training_metrics: MetricsType = training_iteration(state, epoch)
        validation_metrics: MetricsType = validation_iteration(state, epoch)
        all_metrics: MetricsType = metrics_dict(training_metrics, validation_metrics)

        callbacks(state, all_metrics, epoch=epoch)

        check_exit(accelerator)

    accelerator.end_training()
    return callbacks.finalize(state)


def training_iteration(state: TrainingState, epoch: int = 0) -> MetricsType:
    """
    Функция, отвечающая за один шаг обучения модели.
    """
    pipeline = state["pipeline"].train()

    autocast = state["autocast"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    accelerator: accelerate.Accelerator = state["accelerator"]

    callbacks = state["training_step_callbacks"].reset()

    data_iterable = tqdm(
        iterable=state["training_data"],
        position=1,  # Second after epoch bar
        leave=False,  # Space needed for validation bar
        desc="Training",  # Text for training bar
        disable=(not accelerator.is_main_process),
    )

    check_reset_loss(state)
    optimizer.zero_grad()
    accelerator.wait_for_everyone()

    for batch in data_iterable:
        device_batch = to_device(batch, accelerator.device)

        with accelerator.accumulate(pipeline):
            with autocast(accelerator):
                transformed, outputs, loss = pipeline(device_batch)

            assert torch.numel(loss) == 1

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        external: MetricsType = accelerator.unwrap_model(pipeline).losses.get_stats()
        callbacks(
            state=state,
            epoch=epoch,
            outputs=outputs,
            transformed=transformed,
            external=external
        )

        check_exit(accelerator)

    accelerator.wait_for_everyone()

    return callbacks.finalize(state, epoch)


def validation_iteration(state: TrainingState, epoch: int = 0) -> MetricsType:
    """
    Функция, отвечающая за один шаг валидации модели.
    """

    pipeline = state["pipeline"].eval()

    accelerator: accelerate.Accelerator = state["accelerator"]
    callbacks = state["validation_step_callbacks"].reset()

    data_iterable = tqdm(
        iterable=state["validation_data"],
        position=1,  # Second after epoch bar
        leave=False,  # Space needed for validation bar
        desc="Validation",  # Text for training bar
        disable=(not accelerator.is_main_process),
    )
    check_reset_loss(state)

    with torch.no_grad():
        for batch in data_iterable:
            device_batch = to_device(batch, accelerator.device)
            transformed, outputs, _ = pipeline(device_batch)

            external: MetricsType = accelerator.unwrap_model(pipeline).losses.get_stats()
            callbacks(
                state=state,
                epoch=epoch,
                outputs=outputs,
                transformed=transformed,
                external=external
            )

            check_exit(accelerator)

    accelerator.wait_for_everyone()

    return callbacks.finalize(state, epoch)
