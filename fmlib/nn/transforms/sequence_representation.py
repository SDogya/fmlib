from typing import Dict, List, Literal, Self

import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.modular.filtering import FilteringIn
from fmlib.nn.transforms.modular.map import Map
from fmlib.nn.transforms.modular.padding_transform import ChangePaddingTransform

TimeEncoding = Literal["absolute", "delta"]

MetaKey = Literal["event_id", "event_name"]
MetaValue = int | list[int] | str | None
EventMeta = Dict[MetaKey, MetaValue]
EventsMeta = Dict[str, EventMeta]


class WrapToDict(torch.nn.Module):
    """
    Заворачивает входной батч с именем `dict_name` в словарь.
    """

    def __init__(self: Self, dict_name: str) -> None:
        super().__init__()

        self.dict_name: str = dict_name

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        return {self.dict_name: batch}


class MakeSequentialMasks(torch.nn.Module):
    """
    Создаёт маски для последовательности.

    Аргументы:
        events_meta (EventsMeta): Словарь с метаданными событий.
            Ключи - имена событий, значения - словари с метаданными событий.
        events_ids_inp_name (str): Имя входного тензора с идентификаторами событий.
            По умолчанию - `event_ids`.
        padding_mask_inp_name (str): Имя входного тензора с маской заполненности.
            По умолчанию - `evt_attr_9_mask`.
    """

    def __init__(
        self: Self,
        events_meta: EventsMeta,
        events_ids_inp_name: str = "event_ids",
        padding_mask_inp_name: str = "evt_attr_9_mask",
    ) -> None:
        super().__init__()

        self.events_meta: EventsMeta = events_meta
        self.events_ids_inp_name: str = events_ids_inp_name
        self.padding_mask_inp_name: str = padding_mask_inp_name

        self.event_names: list[str] = list(self.events_meta.keys())

        for name in self.event_names:
            meta: EventMeta = self.events_meta[name]
            event_id: int | list[int] | None = meta.get("event_id")
            if event_id is None:
                continue
            elif isinstance(event_id, int):
                buffer: torch.Tensor = torch.tensor([event_id])
            else:
                buffer: torch.Tensor = torch.tensor(list(event_id))
            buffer_name: str = self.get_buffer_name(name)
            self.register_buffer(buffer_name, buffer)

        self.diag_matrix: torch.nn.Buffer = torch.nn.Buffer(
            torch.eye(len(self.events_meta) + 1, dtype=torch.bool)[None, None, ...]
        )

    @staticmethod
    def get_buffer_name(name: str) -> str:
        return f"{name}_event_ids"

    def get_event_ids(self: Self, name: str) -> torch.Tensor | None:
        buffer_name: str = self.get_buffer_name(name)
        return getattr(self, buffer_name, None)

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        padding_mask: torch.BoolTensor = batch[self.padding_mask_inp_name]
        device: torch.device = padding_mask.device

        event_attn_mask: List[torch.Tensor] = []
        for name in self.event_names:
            event_ids: torch.Tensor | None = self.get_event_ids(name)
            if event_ids is None:
                mask: torch.Tensor = padding_mask
            else:
                base: torch.Tensor = batch[self.events_ids_inp_name]
                mask: torch.Tensor = torch.isin(base, event_ids.to(device=device))
            event_attn_mask.append(mask[..., None])
        event_attn_mask.append(torch.ones_like(event_attn_mask[-1]))  # mask for time embedding
        event_attn_mask = torch.cat(event_attn_mask, dim=-1)  # (batch_size, seq_len, n_features)
        expanded_attn_mask = torch.einsum("bsi,bsj->bsij", [event_attn_mask, event_attn_mask])
        event_encoder_attn_mask = expanded_attn_mask | self.diag_matrix.to(device=device)
        return {"event_encoder_attention_mask": event_encoder_attn_mask}


class MakeSequentialEvents(torch.nn.Module):
    """
    Конвертирует в корректные типы тензора событий и применяет паддинг.

    Аргументы:
        events_meta (EventsMeta): Словарь с метаданными событий.
            Ключи - имена событий, значения - словари с метаданными событий.
        events_ids_inp_name (str): Имя входного тензора с идентификаторами событий.
            По умолчанию - `event_ids`.
        padding_mask_inp_name (str): Имя входного тензора с маской заполненности.
            По умолчанию - `evt_attr_9_mask`.
    """

    def __init__(
        self: Self,
        events_meta: EventsMeta,
        events_ids_inp_name: str = "event_ids",
        padding_mask_inp_name: str = "evt_attr_9_mask",
    ) -> None:
        super().__init__()

        self.events_ids_inp_name: str = events_ids_inp_name
        self.padding_mask_inp_name: str = padding_mask_inp_name
        self.events_meta: EventsMeta = events_meta

    def make_padded(self: Self, tensor: torch.Tensor, mask: torch.BoolTensor, padding_value: int) -> torch.Tensor:
        return torch.where(
            other=padding_value,
            condition=mask,
            input=tensor,
        )

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        padding_mask: torch.BoolTensor = batch[self.padding_mask_inp_name]

        result = {name: self.make_padded(batch[name], padding_mask, 0) for name in self.events_meta}
        for name in self.events_meta:
            if result[name].dtype.is_floating_point:
                result[name] = result[name].float()

        result["padding_mask"] = padding_mask
        result["event_ids"] = batch[self.events_ids_inp_name]

        return result


class MakeSequentialTimestamps(torch.nn.Module):
    """
    Изменение временных меток событий.

    Аргументы:
        time_encoding (TimeEncoding | None): Тип кодирования временных меток.
            `delta` - учет относительных временных отклонений, `absolute` - абсолютные метки,
            `None` - без кодирования. По умолчанию - `None`.
        timestamp_inp_name (str): Имя входного тензора с временными метками.
            По умолчанию - `evt_dttm`.
        padding_mask_inp_name (str): Имя входного тензора с маской заполненности.
            По умолчанию - `evt_attr_9_mask`.
    """

    def __init__(
        self: Self,
        time_encoding: TimeEncoding | None = None,
        timestamp_inp_name: str = "evt_dttm",
        padding_mask_inp_name: str = "evt_attr_9_mask",
    ) -> None:
        super().__init__()

        self.timestamp_inp_name: str = timestamp_inp_name
        self.time_encoding: TimeEncoding | None = time_encoding
        self.padding_mask_inp_name: str = padding_mask_inp_name

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        padding_mask: torch.BoolTensor = batch[self.padding_mask_inp_name]

        result: GeneralBatch = {}

        if self.time_encoding == "absolute":
            result["encoding_timestamps"] = batch[self.timestamp_inp_name]
        elif self.time_encoding == "delta":
            timestamps: torch.Tensor = batch[self.timestamp_inp_name]
            time_deltas = timestamps[:, 1:] - timestamps[:, :-1]
            time_deltas = torch.nn.functional.pad(time_deltas, (1, 0), value=0.0)
            time_deltas.masked_fill_(padding_mask.logical_not(), 0)
            result["encoding_timestamps"] = time_deltas
        else:
            if self.time_encoding is not None:
                msg: str = f"Unknown time encoding: {self.time_encoding=}."
                raise ValueError(msg)

        result["positional_timestamps"] = batch[self.timestamp_inp_name]
        return result


def make_for_sequence_representation(
    events_meta: EventsMeta,
    timestamp_inp_name: str = "evt_dttm",
    time_encoding: TimeEncoding | None = None,
    events_ids_inp_name: str = "event_ids",
    padding_mask_inp_name: str = "evt_attr_9_mask",
    bypass: List[str] | None = None,
    change_padding: bool = False,
) -> torch.nn.Module:
    """
    Создает модуль преобразования последовательности событий за один вызов функции.

    Аргументы:
        events_meta (EventsMeta): Словарь с метаданными событий.
            Ключи - имена событий, значения - словари с метаданными событий.
        timestamp_inp_name (str): Имя входного тензора с временными метками.
            По умолчанию - `evt_dttm`.
        time_encoding (TimeEncoding | None): Тип кодирования временных меток.
            `delta` - учет относительных временных отклонений, `absolute` - абсолютные метки,
            `None` - без кодирования. По умолчанию - `None`.
        events_ids_inp_name (str): Имя входного тензора с идентификаторами событий.
            По умолчанию - `event_ids`.
        padding_mask_inp_name (str): Имя входного тензора с маской заполненности.
            По умолчанию - `evt_attr_9_mask`.
        bypass (List[str] | None): Список имен тензоров, которые не будут преобразованы.
            `None` - преобразуются все тензоры, кроме `epk_id`. По умолчанию - `None`.
        change_padding (bool): Изменять маску заполненности c левой на правую.
            По умолчанию - `False`.
    """
    if bypass is None:
        bypass = ["epk_id"]
    bypass = sorted(set(bypass))

    filter_in_list: List[str] = []
    filter_in_list.extend([events_ids_inp_name, padding_mask_inp_name, timestamp_inp_name])
    filter_in_list.extend(list(events_meta.keys()))
    filter_in_list = sorted(set(filter_in_list))

    filtering_in_input = FilteringIn(patterns=filter_in_list)

    if change_padding:
        change_padding_transforms = [
            ChangePaddingTransform(
                mask_column=padding_mask_inp_name,
                columns_to_transform=filter_in_list,
            )
        ]
    else:
        change_padding_transforms = []

    mapping_branch = Map(
        [
            MakeSequentialMasks(
                events_meta=events_meta,
                events_ids_inp_name=events_ids_inp_name,
                padding_mask_inp_name=padding_mask_inp_name,
            ),
            MakeSequentialTimestamps(
                time_encoding=time_encoding,
                timestamp_inp_name=timestamp_inp_name,
                padding_mask_inp_name=padding_mask_inp_name,
            ),
            MakeSequentialEvents(
                events_meta=events_meta,
                events_ids_inp_name=events_ids_inp_name,
                padding_mask_inp_name=padding_mask_inp_name,
            ),
        ]
    )

    events = torch.nn.Sequential(*change_padding_transforms, filtering_in_input, mapping_branch, WrapToDict("events"))

    mapping_full = Map(
        [
            events,
            FilteringIn(bypass),
        ]
    )

    return mapping_full
