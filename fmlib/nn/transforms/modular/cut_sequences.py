from typing import Any, Self

import torch

from fmlib.constants.batches import GeneralBatch


class CutSequences(torch.nn.Module):
    """
    Собирает последовательности и подрезает их:
        - по длине `max_seq_len`
        - по разнице во времени с последним событием `max_time_delta`

    Аргументы:
        sequences (list[str]): Список последовательностей, которые необходимо подрезать
        max_seq_len (int): Максимальная длина последовательности. По умолчанию - 2048.
        max_time_delta (int): Максимальное время между последним и текущим событием.
            По умолчанию - 500 дней в секундах.
        timestamp_name (str): Имя последовательности с временем.
            По умолчанию - "date_stamp".
        padding_value (Any): Значение для заполнения. По умолчанию - 0.
    """

    def __init__(
        self: Self,
        sequences: list[str],
        max_seq_len: int = 2048,
        max_time_delta: int = 500 * 24 * 60 * 60,
        timestamp_name: str = "date_stamp",
        padding_value: Any = 0,
    ) -> None:
        super().__init__()

        self.max_seq_len: int = max_seq_len
        self.sequences: list[str] = sequences
        self.timestamp_name: str = timestamp_name
        all_sequences: list[str] = [timestamp_name, *sequences]
        self.all_sequences: list[str] = sorted(set(all_sequences))
        self.max_time_delta: int = max_time_delta
        self.padding_value: int = padding_value

    @torch.jit.export
    def cut_by_mask(self: Self, batch: GeneralBatch, mask: torch.BoolTensor) -> GeneralBatch:
        counts: torch.LongTensor = torch.sum(mask, dim=-1, dtype=torch.int64)
        max_count: int = torch.max(counts).detach().cpu().item()

        timestamps: torch.LongTensor = batch[self.timestamp_name]
        max_int: int = torch.iinfo(timestamps.dtype).max

        timestamps_masked: torch.LongTensor = torch.where(mask, timestamps, max_int)
        masked_values, masked_indices = torch.sort(timestamps_masked, dim=-1, stable=True)
        masked_indices = masked_indices[..., :max_count]
        masked_values = masked_values[..., :max_count]

        secondary_mask: torch.BoolTensor = masked_values != max_int
        masked_indices: torch.LongTensor = torch.where(secondary_mask, masked_indices, 0)

        padding_value: Any = self.padding_value

        def apply_padding(tensor: torch.Tensor) -> torch.Tensor:
            """
            Получем последние события и подрезаем конец.
            """
            values: torch.Tensor = torch.gather(tensor, -1, masked_indices)
            result = torch.where(secondary_mask, values, padding_value)
            return result

        results: GeneralBatch = {seq: apply_padding(value) for seq, value in batch.items()}

        return results

    @torch.jit.export
    def cut_by_timestamp(self: Self, batch: GeneralBatch) -> GeneralBatch:
        timestamps: torch.LongTensor = batch[self.timestamp_name]

        # Хотим получить маску для событий в пределах `max_time_delta` от последнего
        timestamp_max: torch.LongTensor = torch.max(timestamps, dim=-1).values
        cutoff: torch.LongTensor = timestamp_max[:, None] - self.max_time_delta
        recent: torch.BoolTensor = (cutoff <= timestamps) & (timestamps != self.padding_value)

        return self.cut_by_mask(batch, recent)

    @torch.jit.export
    def cut_by_index(self: Self, batch: GeneralBatch) -> GeneralBatch:
        timestamps: torch.LongTensor = batch[self.timestamp_name]

        max_count: int = timestamps.size(-1)

        if max_count <= self.max_seq_len:
            return batch

        mask: torch.BoolTensor = timestamps != self.padding_value
        counts: torch.LongTensor = torch.sum(mask, dim=-1, dtype=torch.int64)
        arange: torch.LongTensor = torch.arange(max_count, device=timestamps.device)
        indices_mask: torch.BoolTensor = (counts[:, None] - self.max_seq_len) <= arange[None, :]

        return self.cut_by_mask(batch, (mask & indices_mask))

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        cut_sequences: GeneralBatch = {seq: batch[seq] for seq in self.all_sequences}
        cut_sequences = self.cut_by_timestamp(cut_sequences)
        cut_sequences = self.cut_by_index(cut_sequences)
        return cut_sequences
