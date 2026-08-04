import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.constants.device import DEFAULT_DEVICE


def to_device(batch: GeneralBatch, device: torch.device = DEFAULT_DEVICE) -> GeneralBatch:
    """
    Перемещает все тензоры в заданный девайс.

    Аргументы:
        batch (GeneralBatch): Объект, содержащий тензоры.
        device (torch.device, optional): Девайс, в который будут перемещены тензоры.
    """
    result: GeneralBatch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            assert isinstance(value, dict)
            result[key] = to_device(value, device)

    return result
