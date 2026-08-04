from typing import Self

import torch

from fmlib.nn.utils.initialization import nba_init_weights


class DLTClassificationHead(torch.nn.Module):
    """
    Классификационная голова для DLT.
    """

    def __init__(self: Self, hid_dim: int = 512, output_dim: int = 2):
        super().__init__()
        self.pad_tensor = torch.nn.Parameter(torch.zeros(hid_dim))
        self.out_layer = torch.nn.Linear(hid_dim, output_dim)

        self.apply(nba_init_weights)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pad_tensor
        x = self.out_layer(x)
        return x
