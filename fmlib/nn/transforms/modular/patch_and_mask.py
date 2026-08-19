import warnings
from typing import Self, Tuple, TypedDict, cast

import torch
import torch.nn.functional as func

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.modular.timestamp_encoder import TimestampEncoder, TimestampEncoderOutput
from fmlib.nn.transforms.utils.join_batches import join_batches


class PatchAndMaskOutput(TypedDict):
    encoder_mask: torch.BoolTensor
    patches: torch.LongTensor
    patches_pos: torch.LongTensor
    encoder_cross_mask: torch.BoolTensor
    decoder_cross_mask: torch.BoolTensor
    patches_mask: torch.BoolTensor


TRIVAN_DEFAULT_PADDING: int = 0
TRIVAN_TIME_CONSTANT: int = 24 * 60 * 60
TRIVAN_DAY_OF_THE_WEEK_OFFSET: int = 370


class PatchAndMaskImpl(torch.nn.Module):
    def __init__(
        self: Self,
        time_constant: int = TRIVAN_TIME_CONSTANT,
        padding_value: int = TRIVAN_DEFAULT_PADDING,
        day_of_the_week_offset: int = TRIVAN_DAY_OF_THE_WEEK_OFFSET,
        timestamp_encoder: TimestampEncoder | None = None,
    ) -> None:
        super().__init__()

        if timestamp_encoder is None:
            timestamp_encoder = TimestampEncoder()

        self.padding_value: int = padding_value
        self.time_constant: int = time_constant
        self.day_of_the_week_offset: int = day_of_the_week_offset
        self.timestamp_encoder: TimestampEncoder = cast(TimestampEncoder, timestamp_encoder)

    def get_encoder_mask(self: Self, dates: torch.LongTensor) -> torch.BoolTensor:
        delta: torch.Tensor = dates.unsqueeze(-1) - dates.unsqueeze(-2)
        return torch.tril(delta == 0)

    def get_patches(self: Self, dates: torch.LongTensor) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        # Получаем маску для уникальныхп последовательных значений:
        # - Первое значение (first) - всегда первое видимое
        # - Остальные значения (comparison) - если не совпадают с предыдущим
        comparison: torch.BoolTensor = dates[..., 1:] != dates[..., :-1]
        first: torch.BoolTensor = torch.ones_like(dates[..., :1]).bool()
        unique_mask: torch.BoolTensor = torch.cat([first, comparison], dim=-1)

        # Получаем маску для уникальных значений и их количество
        mask: torch.BoolTensor = unique_mask & (dates != self.padding_value)
        counts: torch.LongTensor = torch.sum(mask, dtype=torch.int64, dim=-1)

        max_count: int = torch.max(counts).cpu().item()
        max_length: int = mask.shape[-1]

        arange: torch.LongTensor = torch.arange(max_length, dtype=torch.int64, device=mask.device)

        # - Получаем индексы уникальных значений
        # - Сдвигаем влево, только значения
        # - Обрезаем, заполняем нулями
        indices: torch.LongTensor = torch.where(mask, arange, max_length)
        indices, _ = torch.sort(indices, dim=-1)
        indices = indices[..., :max_count]
        indices_mask: torch.BoolTensor = indices < max_length
        indices = torch.where(indices_mask, indices, 0)

        # Получаем значения и заполняем
        values: torch.Tensor = torch.gather(dates, -1, indices)
        values = torch.where(indices_mask, values, self.padding_value)

        return (values, indices_mask)

    def get_patches_pos(self: Self, patches: torch.LongTensor) -> torch.LongTensor:
        patches_dates: torch.LongTensor = patches * self.time_constant
        dates_encoded: TimestampEncoderOutput = self.timestamp_encoder(patches_dates)

        offset: int = self.day_of_the_week_offset
        patches_tensor_0: torch.LongTensor = dates_encoded["day_of_the_year"] + 1
        patches_tensor_1: torch.LongTesnor = dates_encoded["day_of_the_week"] + offset
        return torch.stack([patches_tensor_0, patches_tensor_1], dim=-1).long()

    def forward(self: Self, timestamps: torch.LongTensor) -> PatchAndMaskOutput:
        dates: torch.LongTensor = timestamps // self.time_constant

        encoder_mask = dates.unsqueeze(-1) - dates.unsqueeze(-2)
        encoder_mask = torch.where(encoder_mask == 0, 1, 0)
        encoder_mask = torch.tril(encoder_mask)

        patches, patches_mask = self.get_patches(dates)
        pos_tensor = self.get_patches_pos(patches)

        encoder_cross_mask = patches.unsqueeze(-1) - dates.unsqueeze(-2)
        encoder_cross_mask = torch.where(encoder_cross_mask == 0, 1, 0)

        shift = func.pad(patches, (0, 1), "constant", self.padding_value)
        decoder_cross_mask = dates.unsqueeze(-1) - shift.unsqueeze(-2)
        decoder_cross_mask = torch.where(decoder_cross_mask == 0, 1, 0)

        result: PatchAndMaskOutput = PatchAndMaskOutput(
            encoder_mask=encoder_mask,
            patches=patches,
            patches_pos=pos_tensor,
            encoder_cross_mask=encoder_cross_mask,
            decoder_cross_mask=decoder_cross_mask,
            patches_mask=patches_mask,
        )

        return result


class PatchAndMask(torch.nn.Module):
    """
    Создаёт тензор из уникальных значений и считает маски
    для attention'а между ними самими, и между ними и входным тензором.

    Аргументы:
        timestamp_name (str): Имя ключа тензора дат в батче.
            По умолчанию - "date_stamp".
        time_constant (int): Константа для перевода даты в секунды.
            По умолчанию - `TRIVAN_TIME_CONSTANT`.
        padding_value (int): Значение для заполнения пустых значений.
            По умолчанию - `TRIVAN_DEFAULT_PADDING`.
        day_of_the_week_offset (int): Смещение для дней недели.
            По умолчанию - `TRIVAN_DAY_OF_THE_WEEK_OFFSET`.
        timestamp_encoder (TimestampEncoder | None): Кодировщик даты.
            По умолчанию - `None`, т.е. будет создан стандартный `TimestampEncoder()`.
        bypass (bool): Пропускать ли батч после обработки. По умолчанию - `True`.
        strict (bool): Вызывать исключения при ошибках. По умолчанию - `True`.
        verbose (bool): Выводить сообщения. По умолчанию - `False`.
    """

    def __init__(
        self: Self,
        timestamp_name: str = "date_stamp",
        time_constant: int = TRIVAN_TIME_CONSTANT,
        padding_value: int = TRIVAN_DEFAULT_PADDING,
        day_of_the_week_offset: int = TRIVAN_DAY_OF_THE_WEEK_OFFSET,
        timestamp_encoder: TimestampEncoder | None = None,
        bypass: bool = True,
        strict: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.bypass: bool = bypass
        self.strict: bool = strict
        self.verbose: bool = verbose
        self.timestamp_name: str = timestamp_name
        self.impl: PatchAndMaskImpl = PatchAndMaskImpl(
            time_constant=time_constant,
            padding_value=padding_value,
            timestamp_encoder=timestamp_encoder,
            day_of_the_week_offset=day_of_the_week_offset,
        )

    def join_batches(self: Self, left: GeneralBatch, right: GeneralBatch) -> GeneralBatch:
        if not self.bypass:
            msg: str = "No bypass expected yet `join_batches` was called."
            if self.strict:
                raise RuntimeError(msg)
            elif self.verbose:
                warnings.warn(msg, stacklevel=2)
        return join_batches(left, right, verbose=self.verbose, strict=self.strict)

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        input_tensor: torch.LongTensor = batch[self.timestamp_name]
        result: PatchAndMaskOutput = self.impl(input_tensor)

        if self.bypass:
            result = self.join_batches(batch, result)

        return result
