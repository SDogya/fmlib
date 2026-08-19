import warnings
from typing import Callable, Dict, List, Self, Tuple

import torch

from fmlib.constants.transforms import DEFAULT_GET_MASK_NAME


def is_sequence(shape: Tuple[int, ...]) -> bool:
    return len(shape) > 1


def compute_lengths(mask: torch.BoolTensor) -> torch.LongTensor:
    return torch.sum(mask, dim=-1, keepdim=True, dtype=torch.int64)


def change_padding(sequence: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
    raw_length: int = sequence.shape[-1]
    arange: torch.LongTensor = torch.arange(raw_length, device=sequence.device, dtype=torch.int64)
    indices: torch.LongTensor = (raw_length - lengths + arange[None, ...]) % raw_length
    result: torch.Tensor = torch.gather(sequence, dim=-1, index=indices)
    return result


class ChangePaddingTransform(torch.nn.Module):
    """
    Изменяет расположение паддинга в батче.

    Аргументы:
        mask_column (str | Callable[[str], str]): Имя колонки с маской паддинга
            или `Callable` для получения такого имени.
        columns_to_transform (List[str] | None): Список колонок, в которых
            необходимо изменить расположение паддинга.
    """

    def __init__(
        self: Self,
        mask_column: str | Callable[[str], str] = DEFAULT_GET_MASK_NAME,
        columns_to_transform: List[str] | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(mask_column, str):
            warnings.warn("Callable `mask_column`. Performance may be suboptimal.", stacklevel=2)

        self.mask_column: str | Callable[[str], str] = mask_column
        self.columns_to_transform: List[str] | None = columns_to_transform

    def get_columns_to_transform(self: Self, base: Dict[str, torch.Tensor]) -> List[str]:
        if self.columns_to_transform is None:
            columns_to_transform: List[str] = []

            for name, body in base.items():
                if is_sequence(body.shape):
                    columns_to_transform.append(name)

            self.columns_to_transform = sorted(columns_to_transform)
            msg: str = f"Columns to transform: {self.columns_to_transform=}"
            warnings.warn(msg, stacklevel=2)
        return self.columns_to_transform

    def transform_base(self: Self, base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        columns_to_transform: List[str] = self.get_columns_to_transform(base)

        lengths: torch.LongTensor | None = None
        if isinstance(self.mask_column, str):
            lengths = compute_lengths(base[self.mask_column])

        result: Dict[str, torch.Tensor] = {**base}

        for column in columns_to_transform:
            curr_lengths: torch.LongTensor | None = lengths
            if curr_lengths is None:
                mask_name: str = self.mask_column(column)
                curr_lengths = compute_lengths(base[mask_name])

            sequence: torch.Tensor = base[column]
            result[column] = change_padding(sequence, curr_lengths)

        return result

    def forward(self: Self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.transform_base(batch)
