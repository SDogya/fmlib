from typing import Dict

import pytest
import torch

from ...utils.batching import UniformBatching
from ..implementation.flat_column import FlatColumn
from ..implementation.named_columns import NamedColumns
from ..implementation.sequence_column import SequenceColumn
from ..iterable_dataset import IterableDataset
from .sincos_dataset import BatchGenerator

Batch = Dict[str, torch.Tensor]


@pytest.mark.parametrize("seed", [1, 7, 19])
@pytest.mark.parametrize("batch_size", [1, 256, 384])
@pytest.mark.parametrize("sub_batch_size", [1, 128, 256])
def test_sincos_sub_batch(seed: int, batch_size: int, sub_batch_size: int):
    batch_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )

    gtr_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )

    batch: Batch = batch_generator.generate_batch()
    sin_col: SequenceColumn = SequenceColumn(
        data=batch["sin_sin"],
        lengths=batch["sin_length"],
        sequence_length=batch_generator.max_length,
        padding=-2,
    )
    phase_col: FlatColumn = FlatColumn(data=batch["phase"])
    frequency_col: FlatColumn = FlatColumn(data=batch["frequency"])
    named: NamedColumns = NamedColumns(
        {
            "sin_sin": sin_col,
            "phase": phase_col,
            "frequency": frequency_col,
        },
    )

    dataset: IterableDataset = IterableDataset(named, batch_size=sub_batch_size)

    gtr_batch: Batch = gtr_generator.generate_padded()
    batching: UniformBatching = UniformBatching(batch_size, sub_batch_size)
    assert len(dataset) == batching.batch_count

    for i, s in enumerate(dataset):
        first, last = batching.get_limits(i)
        assert torch.allclose(s["sin_sin"], gtr_batch["sin_sin"][first:last, ...])
        assert torch.allclose(s["phase"][:, None], gtr_batch["phase"][first:last, ...])
        assert torch.allclose(s["frequency"][:, None], gtr_batch["frequency"][first:last, ...])
