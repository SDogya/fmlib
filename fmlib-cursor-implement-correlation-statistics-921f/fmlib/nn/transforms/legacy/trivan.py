import math
from typing import Dict, Self, Tuple, Union

import torch

from fmlib.nn.transforms.modular.timestamp_encoder import TimestampEncoder, TimestampEncoderOutput


class TrivanTransform(torch.nn.Module):
    def __init__(
        self: Self,
        report_date_inp_name: str = "report_dt",
        report_date_padding_mask_inp_name: str = "report_dt_mask",
        timestamp_inp_name: str = "evt_dttm",
        mcc_inp_name: str = "mcc_token",
        brand_inp_name: str = "brand_token",
        place_inp_name: str = "place_token",
        terminal_inp_name: str = "terminal_token",
        amount_inp_name: str = "amt_token",
        padding_mask_inp_name: str = "evt_dttm_mask",
        padding_value: int = 0,
        adapter_inp_names: Tuple[str] = ("sm_model", "okko_model"),
        embedding_dim: int = 512,
        event_max_distance_in_hours: int = 90 * 24,
        positional_encoding_num_buckets: int = 512,
    ) -> None:
        super().__init__()
        self.report_date_inp_name: str = report_date_inp_name
        self.report_date_padding_mask_inp_name: str = report_date_padding_mask_inp_name
        self.timestamp_inp_name: str = timestamp_inp_name
        self.mcc_inp_name: str = mcc_inp_name
        self.brand_inp_name: str = brand_inp_name
        self.place_inp_name: str = place_inp_name
        self.terminal_inp_name: str = terminal_inp_name
        self.amount_inp_name: str = amount_inp_name
        self.padding_mask_inp_name: str = padding_mask_inp_name
        self.adapter_inp_names: Tuple[str] = adapter_inp_names
        self.padding_value: int = padding_value
        self.embedding_dim: int = embedding_dim
        self.event_max_distance_in_hours: int = event_max_distance_in_hours
        self.positional_encoding_num_buckets: int = positional_encoding_num_buckets
        self.timestamp_encoder = TimestampEncoder()

    def _get_timeinfo(
        self: Self,
        batch: Dict[str, torch.Tensor],
        timestamp_col: str,
        padding_mask_col: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_info: TimestampEncoderOutput = self.timestamp_encoder(batch[timestamp_col])
        padding_mask = batch[padding_mask_col]
        weekday = (time_info["day_of_the_week"] + 5050).masked_fill(~padding_mask, self.padding_value)
        day_of_year = (time_info["day_of_the_year"] + 5061).masked_fill(~padding_mask, self.padding_value)
        return (weekday, day_of_year)

    def _get_positional_encoding(
        self: Self,
        start_points: torch.LongTensor,
        timesteps: torch.LongTensor,
        max_distance_in_hours: int,
        num_buckets: int,
    ) -> torch.LongTensor:
        """
        Функция генерирует позиционный энкодинг
        на основе времени события (`timestamp_inp_name`) и времени отсечки (`report_date_inp_name`).

        Если до времени отсечки осталось меньше `num_buckets // 2` часов,
        то значение энкодинга равно количеству часов до отсечки.
        Если до времени отсечки осталось больше `num_buckets // 2` часов,
        то значение энкодинга вычисляется с помощью логарифмической шкалы на основе
        максимальной глубины событий (`max_distance_in_hours`).
        """
        # для каждого события получаем количество часов оставшихся до отсечки
        relative_position = (start_points[:, None] - timesteps) // (60 * 60)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact
        relative_position_if_large += (
            torch.log(relative_position.float() / max_exact) / math.log(max_distance_in_hours / max_exact)
        ) * (num_buckets - max_exact)
        relative_position_if_large = relative_position_if_large.long().clamp(max=num_buckets - 1)
        return torch.where(is_small, relative_position, relative_position_if_large)

    def forward(self: Self, batch: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        batch_size = batch[self.timestamp_inp_name].shape[0]
        device = batch[self.timestamp_inp_name].device
        result: Dict[str, torch.Tensor] = {}

        encoder_weekday, encoder_day_of_year = self._get_timeinfo(batch, self.timestamp_inp_name, self.padding_mask_inp_name)
        result["encoder_input"] = torch.stack(
            (
                encoder_weekday,
                encoder_day_of_year,
                batch[self.mcc_inp_name],
                batch[self.brand_inp_name],
                batch[self.place_inp_name],
                batch[self.terminal_inp_name],
                batch[self.amount_inp_name],
            ),
            dim=-1,
        )

        decoder_weekday, decoder_day_of_year = self._get_timeinfo(
            batch,
            self.report_date_inp_name,
            self.report_date_padding_mask_inp_name,
        )
        ones = torch.ones_like(batch[self.report_date_inp_name], device=device)
        result["decoder_input"] = torch.stack(
            (
                decoder_weekday,
                decoder_day_of_year,
                ones * 3,
                ones * 10,
                ones * 11,
                ones * 12,
                ones * 13,
            ),
            dim=-1,
        ).unsqueeze(1)
        result["positional_encoding"] = self._get_positional_encoding(
            batch[self.report_date_inp_name],
            batch[self.timestamp_inp_name],
            max_distance_in_hours=self.event_max_distance_in_hours,
            num_buckets=self.positional_encoding_num_buckets,
        )
        result["encoder_padding_mask"] = batch[self.padding_mask_inp_name]

        result["adapter_input"] = {}
        for name in self.adapter_inp_names:
            result["adapter_input"][name] = batch.get(
                name,
                torch.zeros(
                    size=(batch_size, self.embedding_dim),
                    device=device,
                ),
            )
        return result
