import torch

from fmlib.constants.device import DEFAULT_DEVICE

DeviceType = str | torch.device


def make_seeded_generator(seed: int = 777, device: DeviceType = DEFAULT_DEVICE) -> torch.Generator:
    return torch.Generator(device).manual_seed(seed)
