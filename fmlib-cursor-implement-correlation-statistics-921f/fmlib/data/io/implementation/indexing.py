from typing import Tuple, Union

import torch


def raw_get_offsets(lengths: torch.LongTensor) -> torch.LongTensor:
    zero: torch.LongTensor = torch.zeros((1,), device=lengths.device, dtype=torch.int64)
    cumsum: torch.LongTensor = torch.cumsum(lengths, dim=-1)
    return torch.cat([zero, cumsum])


def get_offsets(lengths: torch.LongTensor) -> torch.LongTensor:
    if lengths.ndim != 1:
        msg: str = f"Lengths must be strictly 1D. Got {lengths.ndim}D."
        raise ValueError(msg)
    min_length: int = torch.min(lengths.detach()).cpu().item()
    if min_length < 0:
        msg: str = f"There is a negative length. Got {min_length}."
        raise ValueError(msg)
    return raw_get_offsets(lengths)


LengthType = Union[int, torch.LongTensor]


def raw_get_mask(
    indices: torch.LongTensor,
    offsets: torch.LongTensor,
    length: LengthType,
    random_slicing: bool = False,
) -> Tuple[torch.BoolTensor, torch.LongTensor]:
    length: torch.LongTensor = torch.asarray(length, dtype=torch.int64, device=indices.device)
    last: torch.LongTensor = offsets[indices + 1]
    first: torch.LongTensor = offsets[indices + 0]
    start_tail: torch.LongTensor = last - length
    if random_slicing:
        cond: torch.BoolTensor = (last - first) > length
        range_len = (last - length - first + 1).clamp(min=1)
        rand_shift = (torch.rand_like(first, dtype=torch.float32) * range_len).long()
        start_random = first + rand_shift
        start = torch.where(cond, start_random, start_tail)
    else:
        start = start_tail
    arange: torch.Longtensor = torch.arange(length, dtype=torch.int64, device=offsets.device)
    raw_indices: torch.LongTensor = start[:, None] + arange[None, :]
    mask: torch.BoolTensor = (first[:, None] <= raw_indices) & (raw_indices < last[:, None])
    assert torch.all(torch.sum(mask, dim=-1, dtype=torch.int64) == torch.minimum(last - first, length)).cpu().item()
    # We are indexing `first` because of the data locality & tests
    output_indices: torch.LongTensor = torch.where(mask, raw_indices, first[:, None])
    assert torch.all(first <= torch.min(output_indices, dim=-1).values).cpu().item()
    assert torch.all((torch.max(output_indices, dim=-1).values < last) | (last == first)).cpu().item()
    return (mask, output_indices)


def get_mask(
    indices: torch.LongTensor,
    offsets: torch.LongTensor,
    length: LengthType,
    random_slicing: bool = False
) -> Tuple[torch.BoolTensor, torch.LongTensor]:
    if torch.asarray(length).cpu().item() < 1:
        msg: str = f"Length must be a positive number. Got {length}"
        raise ValueError(msg)
    if torch.numel(indices) < 1:
        msg: str = f"Indices must be non-empty. Got {torch.numel(indices)}."
        raise IndexError(msg)
    if indices.device != offsets.device:
        msg: str = f"Devices must match. Got {indices.device} vs {offsets.device}"
        raise RuntimeError(msg)
    if offsets.ndim != 1:
        msg: str = f"Offsets must be strictly 1D. Got {offsets.ndim}D."
        raise ValueError(msg)
    min_index: int = torch.max(indices.detach()).cpu().item()
    if min_index < 0:
        msg: str = f"Index is too small. Got {min_index}."
        raise IndexError(msg)
    max_index: int = torch.max(indices.detach()).cpu().item()
    if torch.numel(offsets) < max_index:
        msg: str = f"Index is too large. Got {max_index}."
        raise IndexError(msg)
    if not torch.all(offsets[:-1] <= offsets[1:]).cpu().item():
        msg: str = "Offset sequence is not monothonous."
        raise ValueError(msg)
    return raw_get_mask(indices, offsets, length, random_slicing)
