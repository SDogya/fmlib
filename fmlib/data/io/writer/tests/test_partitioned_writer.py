import os
import tempfile
from typing import Dict

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
import torch

from fmlib.data.io.tests.sincos_dataset import BatchGenerator

from ..partitioned_writer import PartitionedParquetWriter


@pytest.mark.parametrize("seed", [42, 777])
@pytest.mark.parametrize("batch_size", [1, 32, 64])
@pytest.mark.parametrize("batch_count", [1, 8, 16, 20])
def test_pyarrow_writer(seed: int, batch_size: int, batch_count: int) -> None:
    params: dict = {
        "batch_size": batch_size,
        "max_frequency": 10,
        "mean_length": 5,
    }
    batch_gen: BatchGenerator = BatchGenerator(generator=torch.Generator().manual_seed(seed), **params)
    gtr_gen: BatchGenerator = BatchGenerator(generator=torch.Generator().manual_seed(seed), **params)
    with tempfile.TemporaryDirectory() as temp_dir:
        path: str = os.path.join(temp_dir, "partitioned")
        with PartitionedParquetWriter(path) as writer:
            for _ in range(batch_count):
                batch: Dict[str, torch.Tensor] = batch_gen.generate_padded()
                writer.write(batch)
        dataset: ds.Dataset = ds.dataset(path, format="parquet")

        batch: pa.RecordBatch
        for batch in dataset.to_batches(batch_size=batch_size):
            gtr_batch: Dict[str, torch.Tensor] = gtr_gen.generate_padded()

            column: str
            for column in gtr_batch:
                raw_values: list = batch.column(column).to_pylist()
                gtr_values: np.ndarray = gtr_batch[column].cpu().numpy()
                col_values: np.ndarray = np.asarray(raw_values).reshape(gtr_values.shape)

                assert np.allclose(col_values, gtr_values)
