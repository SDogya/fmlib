import torch

DEFAULT_SEED: int = 777
DEFAULT_GENERATOR: torch.Generator = torch.Generator().manual_seed(DEFAULT_SEED)
