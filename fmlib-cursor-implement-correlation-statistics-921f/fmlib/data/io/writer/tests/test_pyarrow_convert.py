from typing import Tuple

import numpy as np
import pyarrow as pa
import pytest
import torch

from fmlib.constants.random import DEFAULT_SEED
from fmlib.data.io.writer.pyarrow_convert import ArrayLike, Batch, batch_to_pyarrow, is_nd, last_size, tensor_to_pyarrow
from fmlib.data.utils.typing.dtype import to_pyarrow


def test_last_size():
    assert last_size((1, 1, 1)) == 1
    assert last_size((1, 2, 1, 1)) == 2
    assert last_size((2, 2, 3, 4)) == 24


def test_is_nd():
    assert is_nd((1, 2))
    assert not is_nd((700,))
    assert not is_nd((5, 1))
    assert is_nd((5, 1, 2))
    assert is_nd((5, 2, 1))
    assert not is_nd((5, 1, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [(7,), (5, 1), (3, 1, 1)])
def test_convert_1d(dtype: torch.dtype, size: Tuple[int, ...]):
    gen: torch.Generator = torch.Generator().manual_seed(DEFAULT_SEED)

    torch_data: torch.Tensor = torch.rand(size, dtype=dtype, generator=gen)
    numpy_data: np.ndarray = torch_data.numpy()

    assert not is_nd(torch_data.shape)

    torch_result: ArrayLike = tensor_to_pyarrow(torch_data)
    numpy_result: ArrayLike = tensor_to_pyarrow(numpy_data)

    assert len(torch_result) == size[0]
    assert len(numpy_result) == size[0]
    assert torch_result.type == to_pyarrow(dtype)
    assert numpy_result.type == to_pyarrow(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [(7, 2), (5, 1, 2), (3, 2, 1)])
def test_convert_nd(dtype: torch.dtype, size: Tuple[int, ...]):
    gen: torch.Generator = torch.Generator().manual_seed(DEFAULT_SEED)

    torch_data: torch.Tensor = torch.rand(size, dtype=dtype, generator=gen)

    numpy_data: np.ndarray = torch_data.numpy()

    assert is_nd(torch_data.shape)

    torch_result: ArrayLike = tensor_to_pyarrow(torch_data)
    numpy_result: ArrayLike = tensor_to_pyarrow(numpy_data)

    length: int = last_size(size)

    assert len(torch_result) == size[0]
    assert len(numpy_result) == size[0]
    assert torch_result.type == pa.list_(to_pyarrow(dtype), length)
    assert numpy_result.type == pa.list_(to_pyarrow(dtype), length)

    assert np.allclose(numpy_data.reshape(size[0], -1), np.asarray(numpy_result.to_pylist()))
    assert np.allclose(numpy_data.reshape(size[0], -1), np.asarray(torch_result.to_pylist()))


def test_dict_convert():
    batch_size: int = 7

    gen: torch.Generator = torch.Generator().manual_seed(DEFAULT_SEED)

    sim_batch: Batch = {
        "tensor_1": torch.rand(
            (batch_size,),
            generator=gen,
            dtype=torch.float32,
        ),
        "tensor_2": torch.rand(
            (batch_size, 2),
            generator=gen,
            dtype=torch.float64,
        ),
        "array_1": torch.rand(
            (batch_size,),
            generator=gen,
            dtype=torch.float64,
        ).numpy(),
        "array_2": torch.rand(
            (batch_size, 3),
            generator=gen,
            dtype=torch.float32,
        ).numpy(),
    }

    result: pa.Table = batch_to_pyarrow(sim_batch)

    key: str
    tensor: ArrayLike
    for key, tensor in sim_batch.items():
        test: np.ndarray = np.asarray(result.column(key).to_pylist())
        assert np.allclose(np.asarray(tensor), test)
