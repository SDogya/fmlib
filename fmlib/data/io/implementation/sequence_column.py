from typing import Any, Dict, List, Optional, Self, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc
import torch

from fmlib.constants.device import DEFAULT_DEVICE
from fmlib.constants.metadata import DEFAULT_PADDING
from fmlib.data.io.metadata import Metadata, get_padding, get_sequence_length, list_sequential
from fmlib.data.utils.typing.dtype import pyarrow_to_torch

from .column_protocol import OutputType
from .indexing import get_mask, get_offsets
from .utils.make_mutable import copy_if_immutable


def _get_sequence_length(lengths: torch.LongTensor, length: Optional[int] = None) -> int:
    if length is None:
        length = torch.max(lengths.detach()).cpu().item()
    if length < 1:
        msg: str = f"Length must be positive. Got {length}."
        raise ValueError(msg)
    return length


TabularType = Union[pa.RecordBatch, pa.Table]
ArrayLike = Union[pa.Array, pa.ChunkedArray]


class SequenceColumn:
    def __init__(
        self: Self,
        data: torch.Tensor,
        lengths: torch.LongTensor,
        sequence_length: Optional[int] = None,
        padding: Any = DEFAULT_PADDING,
    ) -> None:
        self.padding: Any = padding
        self.data: torch.Tensor = data
        self.offsets: torch.LongTensor = get_offsets(lengths)
        self.sequence_length: int = _get_sequence_length(lengths, sequence_length)
        assert self.length == torch.numel(lengths)

    @property
    def length(self: Self) -> int:
        return torch.numel(self.offsets) - 1

    def __len__(self: Self) -> int:
        return self.length

    @property
    def device(self: Self) -> torch.device:
        assert self.data.device == self.offsets.device
        return self.offsets.device

    @property
    def dtype(self: Self) -> torch.dtype:
        return self.data.dtype

    def __getitem__(self: Self, indices: torch.LongTensor) -> OutputType:
        indices = indices.to(device=self.device)
        mask: torch.BoolTensor
        output: torch.LongTensor
        mask, output = get_mask(indices, self.offsets, self.sequence_length)
        unmasked_values: torch.Tensor = torch.take(self.data, output)
        masked_values: torch.Tensor = torch.where(mask, unmasked_values, self.padding)
        assert masked_values.device == self.device
        assert masked_values.dtype == self.dtype
        return (mask, masked_values)


def to_torch(array: ArrayLike, device: torch.device = DEFAULT_DEVICE) -> Tuple[torch.LongTensor, torch.Tensor]:
    flatten: ArrayLike = pc.list_flatten(array)
    lengths: ArrayLike = pc.list_value_length(array).cast(pa.int64())

    # Copying to be mutable
    flatten_torch: torch.Tensor = torch.asarray(
        copy_if_immutable(flatten.to_numpy()),
        device=device,
        dtype=pyarrow_to_torch(flatten.type),
    )

    # Copying to be mutable
    lengths_torch: torch.Tensor = torch.asarray(
        copy_if_immutable(lengths.to_numpy()),
        device=device,
        dtype=torch.int64,
    )
    return (lengths_torch, flatten_torch)


def to_sequence_columns(
    data: TabularType,
    metadata: Metadata,
    device: torch.device = DEFAULT_DEVICE,
) -> Dict[str, SequenceColumn]:
    normal: List[str] = list_sequential(metadata)
    result: Dict[str, SequenceColumn] = {}

    column_name: str
    for column_name in normal:
        lengths: torch.LongTensor
        data: torch.Tensor
        lengths, torch_array = to_torch(data.column(column_name), device=device)
        result[column_name] = SequenceColumn(
            data=torch_array,
            lengths=lengths,
            padding=get_padding(metadata, column_name),
            sequence_length=get_sequence_length(metadata, column_name),
        )
    return result
