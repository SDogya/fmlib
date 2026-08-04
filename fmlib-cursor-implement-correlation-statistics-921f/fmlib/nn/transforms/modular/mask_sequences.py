import warnings
from typing import Any, Callable, Self

import torch

from fmlib.constants.batches import Batch


class UniformMaskSequences(torch.nn.Module):
    """
    Функция маскирования последовательностей.
    Предназначена для обучения моделей и, в частности, DLT.

    Аргументы:
        proba (float): Вероятность с которой будут маскироваться токены.
        sequences (list[str]): Список последовательностей для маскирования.
        dim (int): Размерность маскирования. По умолчанию - `-1`.
        keep_last (int): Количество последних токенов, которые будут обязательно сохранены.
            По умолчанию - `0`.
        min_seq_len (int): Минимальная длина последовательности. По умолчанию - `0`.
        ref_sequence (str | None): Основная (модельная) последовательность.
            По умолчанию - `None`, т.е. будет использоваться первая из сортированных `sequences`.
        make_mask_name (Callable[[str], str] | None): Функция для создания имени маски.
            По умолчанию - `None`, что превращается в `DEFAULT_MAKE_MASK_NAME`.
        padding_value (Any): Значение заполнения.
            По умолчанию - `None`, что превращается `DEFAULT_PADDING`.
    """

    def __init__(
        self: Self,
        proba: float,
        sequences: list[str],
        dim: int = -1,
        keep_last: int = 0,
        min_seq_len: int = 0,
        ref_sequence: str | None = None,
        make_mask_name: Callable[[str], str] | None = None,
        padding_value: Any | None = None,
    ) -> None:
        super().__init__()
        if make_mask_name is None:
            from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME

            make_mask_name = DEFAULT_MAKE_MASK_NAME

        if padding_value is None:
            from fmlib.constants.metadata import DEFAULT_PADDING

            padding_value = DEFAULT_PADDING

        if len(sequences) == 0:
            msg: str = f"The list of sequences cannot be empty. Got {sequences=}."
            raise ValueError(msg)

        if (proba < 0) or (proba >= 1):
            msg: str = f"The probability must be in [0, 1). Got {proba=}."
            raise ValueError(msg)
        elif proba == 0.0:
            msg: str = f"Sequential masking will be skiped. Got {proba=}."
            warnings.warn(msg, stacklevel=2)

        if min_seq_len < 0:
            msg: str = f"The minimum length must be non-negative. Got {min_seq_len=}."
            raise ValueError(msg)

        if keep_last < 0:
            msg: str = f"Keep last must be non-negative. Got {keep_last=}."
            raise ValueError(msg)

        self.make_mask_name: Callable[[str], str] = make_mask_name
        self.sequences: list[str] = sorted(sequences)
        self.ref_sequence: str | None = ref_sequence
        self.padding_value: Any = padding_value
        self.min_seq_len: int = min_seq_len
        self.keep_last: int = keep_last
        self.proba: float = proba
        self.dim: int = dim

    def get_ref_mask(self: Self, batch: Batch) -> torch.BoolTensor:
        # Получаем последовательность
        ref_sequence: str = self.ref_sequence or self.sequences[0]
        ref_values: torch.Tensor = batch[ref_sequence]
        device: torch.device = ref_values.device

        # Получаем изначальную маску последовательности
        ref_mask: torch.Tensor = ref_values != self.padding_value
        ref_mask_name: str = self.make_mask_name(ref_sequence)

        # Применяем базовую маску (если есть)
        if ref_mask_name in batch:
            ref_mask = ref_mask & batch[ref_mask_name]

        # Делаем случайную маску с вероятностью
        mask: torch.Tensor = self.proba < torch.rand(ref_values.shape, device=device)

        # Сохраняем последние значения в последовательности
        if self.keep_last > 0:
            reverse_ids: torch.Tensor = torch.cumsum(ref_mask.flip(self.dim), dim=-1)
            mask = mask | (reverse_ids.flip(self.dim) <= self.keep_last)

        # Считаем значения в потенциальной общей маске
        counts: torch.Tensor = torch.sum(mask & ref_mask, dim=self.dim)
        needs_masking: torch.Tensor = self.min_seq_len < counts

        # Маскируем только последовательности, которые нуждаются в маскированиий
        mask = mask | needs_masking.unsqueeze(self.dim)
        return mask

    def forward_train(self: Self, batch: Batch) -> Batch:
        ref_mask: torch.Tensor = self.get_ref_mask(batch)

        result: Batch = {**batch}
        for sequence in self.sequences:
            values: torch.Tensor = result[sequence]
            mask: torch.Tensor = values != self.padding_value
            mask_name: str = self.make_mask_name(sequence)
            if mask_name in result:
                mask = mask & batch[mask_name]
            mask = mask & ref_mask

            result[sequence] = torch.where(mask, values, self.padding_value)
            result[mask_name] = mask

        return result

    def forward(self: Self, batch: Batch) -> Batch:
        if self.training and self.proba > 0.0:
            return self.forward_train(batch)
        return batch
