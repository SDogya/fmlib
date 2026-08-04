import os
import tempfile
from typing import Any, Dict

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
import torch

from fmlib.data.io.implementation.flat_column import to_flat_columns
from fmlib.data.io.implementation.named_columns import NamedColumns
from fmlib.data.io.implementation.sequence_column import to_sequence_columns

from ..iterable_dataset import IterableDataset as TorchIterableDataset
from .sincos_dataset import BatchGenerator, make_metadata, write_dataset


@pytest.mark.parametrize("seed", [42, 777])
@pytest.mark.parametrize("batch_size", [5, 7, 9])
@pytest.mark.parametrize("batch_count", [3, 7, 8, 25])
def test_dataset_pipeline(seed: int, batch_size: int, batch_count: int):
    gtr_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )
    batch_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path: str = os.path.join(temp_dir, "partitioned")
        os.makedirs(path)
        write_dataset(batch_generator, path, batch_count)
        dataset: ds.Dataset = ds.dataset(path, format="parquet")

        data_table: pa.Table = dataset.to_table()
        metadata: Any = make_metadata(batch_generator)
        named_columns: NamedColumns = NamedColumns(
            columns={
                **to_flat_columns(data_table, metadata),
                **to_sequence_columns(data_table, metadata),
            },
        )
        torch_iterable_dataset: TorchIterableDataset = TorchIterableDataset(named_columns, batch_size)

        batch: Dict[str, torch.Tensor]
        for batch in torch_iterable_dataset:
            gtr_batch: Dict[str, torch.Tensor] = gtr_generator.generate_padded()

            key: str
            for key in metadata:
                shape: Any = gtr_batch[key].shape
                element: torch.Any = batch[key].reshape(shape)
                assert torch.allclose(element, gtr_batch[key])
