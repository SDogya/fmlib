import math
from datetime import datetime

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.modular.cut_sequences import CutSequences

START_TIMESTAMP: int = int(datetime(2025, 1, 1).timestamp())
END_TIMESTAMP: int = int(datetime(2026, 1, 1).timestamp())


def generate_sequences(
    seed: int, padding: int, batch_size: int, max_seq_len: int, max_day_delta: int, max_evt_len: int
) -> tuple[GeneralBatch, GeneralBatch]:
    generator = torch.Generator().manual_seed(seed)
    event_lengths = torch.randint(
        low=max(1, math.floor(0.5 * max_evt_len)),
        high=math.ceil(max_evt_len * 1.5),
        size=(batch_size,),
        generator=generator,
    ).tolist()

    alphas, betas, timestamps = [], [], []
    gtr_alphas, gtr_betas, gtr_timestamps = [], [], []
    timestamp_delta = max_day_delta * 24 * 60 * 60
    for i in range(batch_size):
        length = event_lengths[i]

        curr_alphas = torch.randint(1, 100, size=(length,), generator=generator)
        curr_betas = 1.0 + torch.randn(size=(length,), generator=generator)
        curr_timestamps = torch.randint(START_TIMESTAMP, END_TIMESTAMP, size=(length,), generator=generator)
        curr_timestamps, _ = torch.sort(curr_timestamps, dim=-1)

        alphas.append(curr_alphas)
        betas.append(curr_betas)
        timestamps.append(curr_timestamps)

        max_timestamp = torch.max(curr_timestamps).detach().cpu().item()
        cutoff = max_timestamp - timestamp_delta
        gtr_mask = ((length - max_seq_len) <= torch.arange(length)) & (cutoff <= curr_timestamps)

        curr_gtr_alphas, curr_gtr_betas = curr_alphas[gtr_mask], curr_betas[gtr_mask]
        curr_gtr_timestamps = curr_timestamps[gtr_mask]

        assert max_timestamp == torch.max(curr_gtr_timestamps).detach().cpu().item()
        assert torch.all(cutoff <= curr_gtr_timestamps).detach().cpu().item()
        assert torch.numel(curr_gtr_timestamps) <= max_seq_len

        gtr_alphas.append(curr_gtr_alphas)
        gtr_betas.append(curr_gtr_betas)
        gtr_timestamps.append(curr_gtr_timestamps)

    torch_alphas = pad_sequence(alphas, batch_first=True, padding_value=padding)
    torch_betas = pad_sequence(betas, batch_first=True, padding_value=padding)
    torch_timestamps = pad_sequence(timestamps, batch_first=True, padding_value=padding)

    torch_gtr_alphas = pad_sequence(gtr_alphas, batch_first=True, padding_value=padding)
    torch_gtr_betas = pad_sequence(gtr_betas, batch_first=True, padding_value=padding)
    torch_gtr_timestamps = pad_sequence(gtr_timestamps, batch_first=True, padding_value=padding)

    assert torch_alphas.shape == torch_timestamps.shape
    assert torch_betas.shape == torch_timestamps.shape

    torch_batch = {"alpha": torch_alphas, "beta": torch_betas, "date_stamp": torch_timestamps}
    gtr_batch = {"alpha": torch_gtr_alphas, "beta": torch_gtr_betas, "date_stamp": torch_gtr_timestamps}

    return (torch_batch, gtr_batch)


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("padding", [0, -1])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("max_seq_len", [10, 100, 300])
@pytest.mark.parametrize("max_day_delta", [10, 200, 300])
@pytest.mark.parametrize("max_evt_len", [100, 250, 500, 1_000])
def test_cut_sequences(
    seed: int, padding: int, batch_size: int, max_seq_len: int, max_day_delta: int, max_evt_len: int
) -> None:
    torch_batch, gtr_batch = generate_sequences(seed, padding, batch_size, max_seq_len, max_day_delta, max_evt_len)

    timestamp_delta = max_day_delta * 24 * 60 * 60
    module = CutSequences(
        sequences=["alpha", "beta"], max_seq_len=max_seq_len, max_time_delta=timestamp_delta, padding_value=padding
    )

    result = module(torch_batch)

    assert set(result.keys()) == gtr_batch.keys()

    for name in sorted(result.keys()):
        assert torch.equal(result[name], gtr_batch[name])


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("padding", [0, -1])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("max_seq_len", [100, 300])
@pytest.mark.parametrize("max_day_delta", [100, 300])
@pytest.mark.parametrize("max_evt_len", [100, 300, 1_000])
def test_compiled_cut_sequences(
    seed: int, padding: int, batch_size: int, max_seq_len: int, max_day_delta: int, max_evt_len: int
) -> None:
    torch_batch, gtr_batch = generate_sequences(seed, padding, batch_size, max_seq_len, max_day_delta, max_evt_len)

    timestamp_delta = max_day_delta * 24 * 60 * 60
    module = CutSequences(
        sequences=["alpha", "beta"], max_seq_len=max_seq_len, max_time_delta=timestamp_delta, padding_value=padding
    )

    compiled = torch.compile(module, dynamic=True)

    result = compiled(torch_batch)

    assert set(result.keys()) == gtr_batch.keys()

    for name in sorted(result.keys()):
        assert torch.equal(result[name], gtr_batch[name])
