from typing import Dict

import pytest
import torch

from ..dataset import Dataset
from ..implementation.flat_column import FlatColumn
from ..implementation.named_columns import NamedColumns
from ..implementation.sequence_column import SequenceColumn
from .sincos_dataset import BatchGenerator

Batch = Dict[str, torch.Tensor]


@pytest.mark.parametrize("seed", [1, 3, 7, 19])
@pytest.mark.parametrize("batch_size", [1, 33, 77])
def test_sincos_one_batch(seed: int, batch_size: int):
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

    dataset: Dataset = Dataset(named)
    dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset)

    assert len(dataset) == batch_size
    gtr_batch: Batch = gtr_generator.generate_padded()

    for i, s in enumerate(dataloader):
        assert torch.allclose(s["sin_sin"], gtr_batch["sin_sin"][i, ...])
        assert torch.allclose(s["phase"][:, None], gtr_batch["phase"][i, ...])
        assert torch.allclose(s["frequency"][:, None], gtr_batch["frequency"][i, ...])
