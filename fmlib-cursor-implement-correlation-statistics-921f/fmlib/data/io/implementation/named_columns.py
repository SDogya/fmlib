from typing import Callable, Dict, Self, Sequence, Tuple

import torch

from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME

from .column_protocol import ColumnProtocol

Batch = Dict[str, torch.Tensor]


def deduce_device(columns: Sequence[ColumnProtocol]) -> torch.device:
    assert len(columns) > 0
    device: torch.device = columns[0].device

    def is_correct_device(column: ColumnProtocol) -> bool:
        return column.device == device

    if not all(map(is_correct_device, columns)):
        msg: str = "Columns must be all on the same device."
        raise RuntimeError(msg)
    return device


def deduce_length(columns: Sequence[ColumnProtocol]) -> int:
    assert len(columns) > 0
    length: int = columns[0].length

    def is_correct_length(column: ColumnProtocol) -> bool:
        return column.length == length

    if not all(map(is_correct_length, columns)):
        msg: str = "Columns must have the same lengths."
        raise RuntimeError(msg)
    assert length > 0
    return length


def deduce_length_device(columns: Dict[str, ColumnProtocol]) -> Tuple[int, torch.device]:
    raw: list[ColumnProtocol] = [*columns.values()]
    columns_length: int = deduce_length(raw)
    columns_device: torch.device = deduce_device(raw)
    del raw
    return (columns_length, columns_device)


class NamedColumns:
    def __init__(
        self: Self,
        columns: Dict[str, ColumnProtocol],
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
    ) -> None:
        self.columns_length: int
        self.columns_device: torch.device
        self.columns_length, self.columns_device = deduce_length_device(columns)

        self.columns: Dict[str, ColumnProtocol] = columns
        self.make_mask_name: Callable[[str], str] = make_mask_name

    @property
    def length(self: Self) -> int:
        return self.columns_length

    @property
    def device(self: Self) -> torch.device:
        return self.columns_device

    def __len__(self: Self) -> int:
        return self.columns_length

    def __getitem__(self: Self, indices: torch.LongTensor) -> Batch:
        indices = indices.to(device=self.device)
        result: Batch = {}
        for name, column in self.columns.items():
            result[self.make_mask_name(name)], result[name] = column[indices]
        return result
