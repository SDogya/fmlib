from typing import Dict, Self

import torch
import torch.utils.data as data

from .implementation.named_columns import NamedColumns

Batch = Dict[str, torch.Tensor]


class Dataset(data.Dataset):
    def __init__(self: Self, named_columns: NamedColumns) -> None:
        super().__init__()

        self.named_columns: NamedColumns = named_columns

    @property
    def device(self: Self) -> torch.device:
        return self.named_columns.device

    @property
    def length(self: Self) -> int:
        return self.named_columns.length

    def __len__(self: Self) -> int:
        return self.length

    def __getitem__(self: Self, index: int) -> Batch:
        on_device: torch.Tensor = torch.asarray(index, device=self.device, dtype=torch.int64)
        return self.named_columns[on_device[None]]
