import datetime

import numpy as np
import pytest
import torch
import torch.nn.functional as func
from torch.nn.utils.rnn import pad_sequence

from fmlib.nn.transforms.modular.patch_and_mask import PatchAndMaskImpl

END_DATE: int = int(datetime.datetime(2025, 12, 1).timestamp())
START_DATE: int = int(datetime.datetime(2025, 9, 1).timestamp())


def get_timestamps_and_masks(seed: int, batch_size: int, max_seq_len: int, padding_value: int = 0):
    gen = torch.Generator().manual_seed(seed)
    pad_counts = torch.randint(low=0, high=max_seq_len, size=(batch_size, 1), generator=gen)
    timestamps = torch.randint(low=START_DATE, high=END_DATE, size=(batch_size, max_seq_len), generator=gen)

    timestamps, _ = torch.sort(timestamps, dim=-1)
    mask = pad_counts <= torch.arange(max_seq_len)[None, :]
    timestamps = torch.where(mask, timestamps, padding_value)

    return (timestamps, mask)


@pytest.mark.parametrize("seed", [1, 33, 42, 777])
@pytest.mark.parametrize("batch_size", [1, 17, 19, 101])
@pytest.mark.parametrize("max_seq_len", [11, 29, 117, 511])
def test_get_patches(seed: int, batch_size: int, max_seq_len: int) -> None:
    def get_patches(dates: torch.LongTensor, padding_value=0) -> tuple[torch.LongTensor, torch.BoolTensor]:
        def make_patch(x: torch.Tensor) -> torch.Tensor:
            return torch.unique_consecutive(x[x != padding_value])

        raw_patches: list[torch.Tensor] = [make_patch(x) for x in dates.detach().cpu()]
        raw_patches_mask: list[torch.Tensor] = [torch.ones(x.shape).bool() for x in raw_patches]
        patches: torch.LongTensor = pad_sequence(raw_patches, batch_first=True)
        patches_mask: torch.BoolTensor = pad_sequence(raw_patches_mask, batch_first=True)

        return (patches, patches_mask)

    patch_and_mask = PatchAndMaskImpl(padding_value=0)
    time_constant: int = patch_and_mask.time_constant
    timestamps, _ = get_timestamps_and_masks(seed, batch_size, max_seq_len)

    dates = timestamps // time_constant
    gtr_patches, gtr_mask = get_patches(dates)
    res_patches, res_mask = patch_and_mask.get_patches(dates)

    assert torch.equal(gtr_patches, res_patches)
    assert torch.equal(gtr_mask, res_mask)


@pytest.mark.parametrize("seed", [1, 77, 42, 333])
@pytest.mark.parametrize("batch_size", [1, 17, 19, 101])
@pytest.mark.parametrize("max_seq_len", [11, 29, 117, 511])
def test_get_patches_pos(seed: int, batch_size: int, max_seq_len: int) -> None:
    def get_patches_pos(patches: torch.Tensor, time_constant: int) -> torch.LongTensor:
        patches_dates = np.array((patches * time_constant).detach().cpu().numpy(), dtype="datetime64[s]")
        raw_tensor_0 = (patches_dates - patches_dates.astype("datetime64[Y]")).astype("timedelta64[D]")
        patches_tensor_0 = torch.from_numpy((raw_tensor_0 + 1).astype("int"))
        patches_tensor_1 = torch.from_numpy(np.vectorize(lambda x: x.weekday() + 370)(patches_dates.astype(datetime.datetime)))
        patches_pos = torch.stack([patches_tensor_0, patches_tensor_1], dim=-1).long()
        return patches_pos

    patch_and_mask = PatchAndMaskImpl(padding_value=0)
    time_constant: int = patch_and_mask.time_constant
    timestamps, _ = get_timestamps_and_masks(seed, batch_size, max_seq_len)

    dates = timestamps // time_constant
    patches, _ = patch_and_mask.get_patches(dates)

    gtr_pos_tensor = get_patches_pos(patches, time_constant)
    res_pos_tensor = patch_and_mask.get_patches_pos(patches)

    assert torch.equal(gtr_pos_tensor, res_pos_tensor)


@pytest.mark.parametrize("seed", [14, 17, 19, 23])
@pytest.mark.parametrize("batch_size", [1, 17, 19, 177])
@pytest.mark.parametrize("max_seq_len", [11, 29, 117, 511, 1001])
def test_get_patch_and_mask(seed: int, batch_size: int, max_seq_len: int) -> None:
    def patch_and_mask_creation(inp_tensor, time_constant=60 * 60 * 24):
        timestamps = inp_tensor // time_constant
        encoder_mask = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)
        encoder_mask = torch.where(encoder_mask == 0, 1, 0)
        encoder_mask = torch.tril(encoder_mask)

        patches = [torch.unique_consecutive(x[x != 0]) for x in timestamps.detach().cpu()]
        patches_mask = [torch.ones(x.shape) for x in patches]
        patches = pad_sequence(patches, batch_first=True)
        patches_mask = pad_sequence(patches_mask, batch_first=True)

        patches_dates = np.array((patches * time_constant).detach().cpu().numpy(), dtype="datetime64[s]")
        patches_tensor_0 = torch.from_numpy(
            ((patches_dates - patches_dates.astype("datetime64[Y]")).astype("timedelta64[D]") + 1).astype("int")
        )
        patches_tensor_1 = torch.from_numpy(np.vectorize(lambda x: x.weekday() + 370)(patches_dates.astype(datetime.datetime)))
        patches_pos = torch.stack([patches_tensor_0, patches_tensor_1], dim=-1).long().to(timestamps.device)

        patches = patches.to(timestamps.device)
        encoder_cross_mask = patches.unsqueeze(-1) - timestamps.unsqueeze(-2)
        encoder_cross_mask = torch.where(encoder_cross_mask == 0, 1, 0)

        patches_shift = func.pad(patches, (0, 1), "constant", 0)
        decoder_cross_mask = timestamps.unsqueeze(-1) - patches_shift.unsqueeze(-2)
        decoder_cross_mask = torch.where(decoder_cross_mask == 0, 1, 0)
        return encoder_mask, patches, patches_pos, encoder_cross_mask, decoder_cross_mask, patches_mask

    patch_and_mask = PatchAndMaskImpl(padding_value=0)
    time_constant: int = patch_and_mask.time_constant
    timestamps, _ = get_timestamps_and_masks(seed, batch_size, max_seq_len)

    dates = timestamps // time_constant
    patches, _ = patch_and_mask.get_patches(dates)

    result = patch_and_mask(timestamps)

    encoder_mask, patches, patches_pos, encoder_cross_mask, decoder_cross_mask, patches_mask = patch_and_mask_creation(
        timestamps
    )

    assert torch.equal(result["encoder_mask"], encoder_mask)
    assert torch.equal(result["patches"], patches)
    assert torch.equal(result["patches_pos"], patches_pos)
    assert torch.equal(result["encoder_cross_mask"], encoder_cross_mask)
    assert torch.equal(result["decoder_cross_mask"], decoder_cross_mask)
    assert torch.equal(result["patches_mask"], patches_mask)
