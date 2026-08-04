import numpy as np
import pyarrow as pa
import torch

from ..dtype import (
    numpy_to_pyarrow,
    numpy_to_torch,
    pyarrow_to_numpy,
    pyarrow_to_torch,
    to_numpy,
    to_pyarrow,
    to_torch,
    torch_to_numpy,
    torch_to_pyarrow,
)

TORCH_DTYPE_LIST: list[torch.dtype] = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
    torch.int32,
    torch.uint32,
    torch.int64,
    torch.uint64,
    torch.float32,
    torch.float64,
]


def test_transitivity():
    torch_dtype: torch.dtype

    for torch_dtype in TORCH_DTYPE_LIST:
        np_dtype: np.dtype = torch_to_numpy(torch_dtype)
        assert torch_dtype == numpy_to_torch(np_dtype)

        pa_dtype: pa.DataType = torch_to_pyarrow(torch_dtype)
        assert torch_dtype == pyarrow_to_torch(pa_dtype)

        pa_dtype: pa.DataType = numpy_to_pyarrow(np_dtype)
        assert torch_dtype == pyarrow_to_torch(pa_dtype)
        assert np_dtype == pyarrow_to_numpy(pa_dtype)


TORCH_VS_PA: list[tuple[torch.dtype, pa.DataType]] = [
    (torch.int8, pa.int8()),
    (torch.uint32, pa.uint32()),
    (torch.int64, pa.int64()),
    (torch.float32, pa.float32()),
    (torch.float64, pa.float64()),
]


def test_torch_vs_pa():
    pa_dtype: pa.DataType
    torch_dtype: torch.dtype
    for torch_dtype, pa_dtype in TORCH_VS_PA:
        assert torch_dtype == pyarrow_to_torch(pa_dtype)
        assert pa_dtype == torch_to_pyarrow(torch_dtype)


TORCH_VS_NP: list[tuple[torch.dtype, np.dtype]] = [
    (torch.int8, np.int8),
    (torch.uint32, np.uint32),
    (torch.int64, np.int64),
    (torch.float32, np.float32),
    (torch.float64, np.float64),
]


def test_torch_vs_np():
    np_dtype: pa.DataType
    torch_dtype: torch.dtype
    for torch_dtype, np_dtype in TORCH_VS_NP:
        assert torch_dtype == numpy_to_torch(np_dtype)
        assert np_dtype == torch_to_numpy(torch_dtype)


def test_any_transitivity():
    torch_dtype: torch.dtype
    for torch_dtype in TORCH_DTYPE_LIST:
        np_dtype: np.dtype = torch_to_numpy(torch_dtype)
        pa_dtype: pa.DataType = numpy_to_pyarrow(np_dtype)

        assert to_torch(np_dtype) == torch_dtype
        assert to_torch(pa_dtype) == torch_dtype
        assert to_torch(np_dtype) == to_torch(pa_dtype)

        assert to_numpy(pa_dtype) == np_dtype
        assert to_numpy(torch_dtype) == np_dtype
        assert to_numpy(pa_dtype) == to_numpy(torch_dtype)

        assert to_pyarrow(np_dtype) == pa_dtype
        assert to_pyarrow(torch_dtype) == pa_dtype
        assert to_pyarrow(np_dtype) == to_pyarrow(torch_dtype)
