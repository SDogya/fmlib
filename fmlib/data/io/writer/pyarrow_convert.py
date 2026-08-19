from functools import lru_cache
from typing import Dict, List, Tuple, Union

import numpy as np
import pyarrow as pa
import torch

from fmlib.data.utils.typing.dtype import to_pyarrow

ArrayLike = Union[pa.Array, pa.ChunkedArray]
TensorLike = Union[np.ndarray, torch.Tensor]
Batch = Dict[str, TensorLike]


def last_size(shape: Tuple[int, ...]) -> int:
    @lru_cache
    def _last_size(shape: Tuple[int, ...]) -> int:
        result: int = 1
        for dim in shape[1:]:
            result *= dim
        return result

    return _last_size(shape)


def to_numpy(data: TensorLike) -> np.ndarray:
    result: np.ndarray
    if torch.is_tensor(data):
        assert isinstance(data, torch.Tensor)
        result = data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        result = data
    else:
        msg: str = f"Expected a tensor or numpy.ndarray. Got {type(data)}."
        raise TypeError(msg)
    result = result.reshape(result.shape[0], last_size(result.shape))
    return result


def is_nd(shape: Tuple[int, ...]) -> bool:
    @lru_cache
    def _is_nd(shape: Tuple[int, ...]) -> bool:
        if len(shape) > 1:
            return last_size(shape) > 1
        return False

    return _is_nd(shape)


def tensor_to_pyarrow(tensor: TensorLike) -> ArrayLike:
    numpy_data: np.ndarray = to_numpy(tensor)
    dtype: pa.DataType = to_pyarrow(numpy_data.dtype)

    result: ArrayLike
    if is_nd(numpy_data.shape):
        length: int = last_size(numpy_data.shape)
        result = pa.FixedSizeListArray.from_arrays(pa.array(numpy_data.ravel(), type=dtype), list_size=length)
        assert result.type == pa.list_(dtype, length)
    else:
        result = pa.array(numpy_data.ravel(), type=dtype)
        assert result.type == dtype
    assert len(result) == tensor.shape[0]
    return result


def validate_batch_size(batch: Batch) -> int:
    keys: List[str] = sorted(batch.keys())

    batch_size: int = len(batch[keys[0]])

    for key in keys:
        curr_size: int = len(batch[key])
        if curr_size != batch_size:
            msg: str = f"Invalid batch size in {key}. Got {curr_size} vs {batch_size}."
            raise ValueError(msg)

    return batch_size


def batch_to_pyarrow(batch: Batch) -> pa.Table:
    result: Dict[str, ArrayLike] = {}

    batch_size: int = validate_batch_size(batch)

    key: str
    values: TensorLike
    for key, values in batch.items():
        result[key] = tensor_to_pyarrow(values)
        assert batch_size == len(result[key])

    assert batch_size == validate_batch_size(result)

    return pa.table(result)
