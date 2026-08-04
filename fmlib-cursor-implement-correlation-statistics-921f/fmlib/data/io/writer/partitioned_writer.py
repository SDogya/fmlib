import warnings
from typing import Any, Dict, Optional, Self, Union

import numpy as np
import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
import torch

from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.constants.io import (
    DEFAULT_FRAGMENT_POSTFIX,
    DEFAULT_REPLICAS_INFO,
    DEFAULT_TEMPLATE_FRAGMENT_NAME,
    DEFAULT_WRITE_EVERY_N_BATCH,
)
from fmlib.data.io.partitioning.replicas import ReplicasInfoProtocol
from fmlib.data.utils.prepare_directory import prepare_directory

from .pyarrow_convert import batch_to_pyarrow

ArrayLike = Union[pa.Array, pa.ChunkedArray]
TensorLike = Union[np.ndarray, torch.Tensor]
Batch = Dict[str, TensorLike]


def validate_write_every_n_batch(write_every_n_batch: int) -> int:
    if write_every_n_batch <= 0:
        msg: str = f"Invalid `write_every_n_batch`. Got {write_every_n_batch}."
        raise ValueError(msg)
    if (write_every_n_batch == 1) or (write_every_n_batch >= 128):
        warnings.warn(f"Suspicious `write_every_n_batch`. Got {write_every_n_batch}.", stacklevel=2)
    return write_every_n_batch


class PartitionedParquetWriter:
    """
    TODO: Add docstring
    """

    def __init__(
        self: Self,
        base_path: str,
        filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
        write_every_n_batch: int = DEFAULT_WRITE_EVERY_N_BATCH,
        writer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        if writer_kwargs is None:
            writer_kwargs = {}
        self.base_path: str = prepare_directory(path=base_path, strict=True, exists_ok=True, filesystem=filesystem)
        self.filesystem: fs.FileSystem = filesystem

        self.counter: int = 0
        self.schema: Optional[pa.Schema] = None
        self.writer: Optional[pq.ParquetWriter] = None
        self.write_every_n_batch: int = validate_write_every_n_batch(write_every_n_batch)

        self.replicas_info: ReplicasInfoProtocol = replicas_info
        self.writer_kwargs: Dict[str, Any] = writer_kwargs

        self.template_fragment_name: str = DEFAULT_TEMPLATE_FRAGMENT_NAME

    def make_partition_name(self: Self, postfix: str) -> str:
        result: str = self.template_fragment_name.format(
            replica=self.replicas_info.curr_replica,
            part=self.counter,
            postfix=postfix,
        )
        return result

    def make_path(self: Self, postfix: str) -> str:
        name: str = self.make_partition_name(postfix)
        full_path: str = f"{self.base_path}/{name}"
        return full_path

    def make_writer(self: Self, schema: pa.Schema, full_path: str) -> pq.ParquetWriter:
        if self.filesystem.get_file_info(full_path).type != fs.FileType.NotFound:
            warnings.warn(f"File already exists: {full_path}. Overwriting it.", stacklevel=2)
            self.filesystem.delete_file(full_path)
        assert self.filesystem.get_file_info(full_path).type == fs.FileType.NotFound
        result: pq.ParquetWriter = pq.ParquetWriter(
            where=full_path,
            schema=schema,
            filesystem=self.filesystem,
            **self.writer_kwargs,
        )
        return result

    def get_or_make_writer(self: Self, schema: pa.Schema, full_path: str) -> pq.ParquetWriter:
        if self.writer is None:
            self.writer = self.make_writer(schema, full_path)
        else:
            assert self.writer.schema.equals(schema)
        return self.writer

    def write(self: Self, data: Batch, postfix: str = DEFAULT_FRAGMENT_POSTFIX) -> None:
        table: pa.Table = batch_to_pyarrow(data)

        if self.counter % self.write_every_n_batch == 0:
            self.clean_up_writer()

        if self.schema is None:
            self.schema = table.schema

        full_path: str = self.make_path(postfix)
        writer: pq.ParquetWriter = self.get_or_make_writer(self.schema, full_path)

        writer.write(table)

        self.counter = self.counter + 1
        return self.counter

    def __call__(self: Self, data: Batch, postfix: str = DEFAULT_FRAGMENT_POSTFIX) -> None:
        return self.write(data, postfix=postfix)

    def clean_up_writer(self: Self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.writer = None

    def close(self: Self) -> None:
        self.clean_up_writer()
        assert self.writer is None

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(self: Self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self: Self) -> None:
        if self.writer is not None:
            warnings.warn("Writer is not closed.", stacklevel=2)
        self.close()
