from typing import Protocol, Self, Tuple

import torch

OutputType = Tuple[torch.BoolTensor, torch.Tensor]


class ColumnProtocol(Protocol):
    def __len__(self: Self) -> int: ...

    @property
    def length(self: Self) -> int: ...

    @property
    def device(self: Self) -> torch.device: ...

    def __getitem__(self: Self, indices: torch.LongTensor) -> OutputType: ...
