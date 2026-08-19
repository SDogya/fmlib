from typing import Self

import torch

from fmlib.constants.batches import GeneralBatch


class JoinSequences(torch.nn.Module):
    """
    Собирает последовательности и объединяет их под именем `sequences_name`.

    Аргументы:
        sequences (list[str]): Список последовательностей, которые необходимо объединить.
        sequences_name (str): Имя объединенной последовательности. По умолчанию - "sequences".
    """

    def __init__(self: Self, sequences: list[str], sequences_name: str = "sequences") -> None:
        super().__init__()

        self.sequences_name: str = sequences_name
        self.sequences: list[str] = list(sequences)

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        sequences: list[torch.Tensor] = [batch[seq] for seq in self.sequences]
        return {self.sequences_name: torch.stack(sequences, dim=-1)}
