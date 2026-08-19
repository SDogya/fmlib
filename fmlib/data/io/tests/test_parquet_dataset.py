import re
import warnings
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fmlib.data.io.parquet_dataset import ParquetDataset
from fmlib.data.io.utils.compute_length import (
    compute_fixed_size_batches_length,
    compute_fixed_size_generic_length,
)


@dataclass
class FakeReplicasInfo:
    num_replicas: int = 1
    curr_replica: int = 0


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("num_replicas", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 7, 19])
@pytest.mark.parametrize("partition_size", [1, 3, 5, 17])
@pytest.mark.parametrize("partition_count", [1, 2, 7, 19])
def test_parquet_dataset_length(
    seed: int, num_replicas: int, batch_size: int, partition_size: int, partition_count: int
) -> None:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    fragment_sizes: list[int] = torch.randint(low=1, high=31, size=(partition_count,), generator=generator).tolist()

    with TemporaryDirectory() as tmpdir:
        for i, fragment_size in enumerate(fragment_sizes):
            test_data: torch.Tensor = torch.arange(fragment_size)
            table: pa.Table = pa.table({"test_data": test_data.tolist()})
            pq.write_table(table, f"{tmpdir}/fragment_{i}.parquet")

        fake_replicas_info: FakeReplicasInfo = FakeReplicasInfo(num_replicas=num_replicas)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message=re.escape("Suboptimal parameters:") + ".*",
            )
            dataset: ParquetDataset = ParquetDataset(
                source=tmpdir,
                metadata={"test_data": {}},
                partition_size=partition_size,
                batch_size=batch_size,
                replicas_info=fake_replicas_info,
            )

        last_idx = 0
        for i, _ in enumerate(dataset):
            last_idx = i

        length: int = last_idx + 1
        assert length == len(dataset)

        generic_length: int = compute_fixed_size_generic_length(
            iterable=dataset.iterator,
            num_replicas=num_replicas,
            batch_size=batch_size,
        )

        assert length == generic_length

        batches_length: int = compute_fixed_size_batches_length(
            iterable=dataset.iterator,
            num_replicas=num_replicas,
            batch_size=batch_size,
        )

        assert length == batches_length
