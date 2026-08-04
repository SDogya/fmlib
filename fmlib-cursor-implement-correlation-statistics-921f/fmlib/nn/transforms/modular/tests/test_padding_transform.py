from typing import Dict, List, Tuple

import pytest
import torch

from fmlib.constants.metadata import DEFAULT_PADDING
from fmlib.nn.transforms.modular.padding_transform import ChangePaddingTransform


def generate_data(seed: int, max_length: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    lengths: List[int] = torch.randint(
        low=0,
        high=max_length + 1,
        size=(17,),
        generator=generator,
    ).tolist()

    def make_data(length: int) -> List[int]:
        data: List[int] = torch.randint(
            low=1,
            high=888,
            size=(length,),
            generator=generator,
        ).tolist()
        return data

    def make_padding(length: int) -> List[int]:
        padding: List[int] = [DEFAULT_PADDING] * (max_length - length)
        return padding

    list_data: List[List[int]] = [make_data(length) for length in lengths]
    list_padding: List[List[int]] = [make_padding(length) for length in lengths]

    input_data: torch.LongTensor = torch.asarray(
        [(padding + data) for padding, data in zip(list_padding, list_data, strict=False)],
        dtype=torch.int64,
    )

    groundtruth_data: torch.LongTensor = torch.asarray(
        [(data + padding) for padding, data in zip(list_padding, list_data, strict=False)],
        dtype=torch.int64,
    )

    return (input_data, groundtruth_data)


@pytest.mark.parametrize("seed", [1, 2, 7, 14, 77])
@pytest.mark.parametrize("max_length", [2, 5, 12, 33])
@pytest.mark.parametrize("name", ["one", "two", "three", "attention_mask"])
def test_padding_post_transform_defined(seed: int, max_length: int, name: str) -> None:
    input_data, groundtruth_data = generate_data(seed, max_length)
    input_mask: torch.BoolTensor = (input_data != DEFAULT_PADDING).bool()
    groundtruth_mask: torch.BoolTensor = (groundtruth_data != DEFAULT_PADDING).bool()

    transform: torch.nn.Module = ChangePaddingTransform(mask_column=name)

    input_batch: Dict[str, torch.Tensor] = {
        name: input_mask,
        "test": input_data,
    }

    with pytest.warns(UserWarning):
        result_batch: Dict[str, torch.Tensor] = transform(input_batch)

    assert torch.allclose(result_batch["test"], groundtruth_data)
    assert torch.allclose(result_batch[name], groundtruth_mask)
