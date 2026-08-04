from typing import Any, Dict, Optional, Self, Union

import pyarrow as pa
import torch

from fmlib.constants.device import DEFAULT_DEVICE
from fmlib.constants.metadata import DEFAULT_PADDING
from fmlib.data.io.metadata import Metadata, list_not_sequential
from fmlib.data.utils.typing.dtype import pyarrow_to_torch

from .column_protocol import OutputType
from .utils.make_mutable import copy_if_immutable

ArrayLike = Union[pa.Array, pa.ChunkedArray]
TabularType = Union[pa.RecordBatch, pa.Table]


class FlatColumn:
    def __init__(
        self: Self,
        data: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        padding: Any = DEFAULT_PADDING,
    ) -> None:
        self.padding: Any = padding
        self.data: torch.Tensor = data
        self.mask: Optional[torch.BoolTensor] = mask

    @property
    def length(self: Self) -> int:
        result: int = torch.numel(self.data)
        if self.mask is not None:
            assert result == torch.numel(self.mask)
        return result

    def __len__(self: Self) -> int:
        return self.length

    @property
    def device(self: Self) -> torch.device:
        result: int = self.data.device
        if self.mask is not None:
            assert result == self.mask.device
        return result

    def _get_mask(self: Self, indices: torch.LongTensor) -> torch.BoolTensor:
        mask: torch.BoolTensor
        mask = torch.ones_like(indices, dtype=torch.bool) if self.mask is None else self.mask[indices]
        return mask

    def __getitem__(self: Self, indices: torch.LongTensor) -> OutputType:
        indices = indices.to(device=self.device)
        mask: torch.BoolTensor = self._get_mask(indices)
        output: torch.Tensor = torch.where(mask, self.data[indices], self.padding)
        return (mask, output)


def to_torch(array: ArrayLike, device: torch.device = DEFAULT_DEVICE, padding: Any = DEFAULT_PADDING) -> OutputType:
    dtype: torch.dtype = pyarrow_to_torch(array.type)

    mask_torch: Optional[torch.BoolTensor] = None
    if array.null_count > 0:
        mask_torch = torch.asarray(
            copy_if_immutable(array.is_valid().to_numpy(zero_copy_only=False)),
            device=device,
            dtype=torch.bool,
        )

    array_torch: torch.Tensor = torch.asarray(
        copy_if_immutable(array.fill_null(padding).to_numpy()),
        device=device,
        dtype=dtype,
    )
    return (mask_torch, array_torch)


def to_flat_columns(
    data: TabularType,
    metadata: Metadata,
    device: torch.device = DEFAULT_DEVICE,
    padding: Any = DEFAULT_PADDING,
) -> Dict[str, FlatColumn]:
    result: Dict[str, FlatColumn] = {}
    for column_name in list_not_sequential(metadata):
        mask, torch_array = to_torch(data.column(column_name), device, padding)
        result[column_name] = FlatColumn(data=torch_array, mask=mask, padding=padding)
    return result
