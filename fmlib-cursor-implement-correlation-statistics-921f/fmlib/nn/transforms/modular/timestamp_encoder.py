from datetime import datetime, timezone
from typing import List, Self, TypedDict

import torch

START_YEAR: int = 1970
END_YEAR: int = 2077

NON_LEAP_YEAR_DAYS: List[int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
LEAP_YEAR_DAYS: List[int] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SECONDS_PER_DAY: List[int] = 60 * 60 * 24


def is_leap_year(year: torch.LongTensor) -> torch.BoolTensor:
    return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


class TimestampEncoderOutput(TypedDict):
    """Выход энкодера таймстемпов."""

    year: torch.LongTensor
    is_year_leap: torch.LongTensor
    day_of_the_year: torch.LongTensor
    month_of_the_year: torch.LongTensor
    day_of_the_month: torch.LongTensor
    day_of_the_week: torch.LongTensor
    hour_of_the_day: torch.LongTensor
    minute_of_the_hour: torch.LongTensor


class TimestampEncoder(torch.nn.Module):
    """
    Энкодер таймстемпов. Получает информацию о времени и даты из
    Unix timestamp'ов.

    Аргументы:
        start_year: Начальный год, с которого начинается отсчёт.
            По умолчанию - 1970.
        end_year: Конечный год, до которого идёт отсчёт.
            По умолчанию - 2077.
        tz: Часовой пояс. По умолчанию - UTC.
    """

    def __init__(
        self: Self,
        start_year: int = START_YEAR,
        end_year: int = END_YEAR,
        tz: timezone = timezone.utc,
    ) -> None:
        super().__init__()

        self.tz: timezone = tz
        self.end_year: int = end_year
        self.start_year: int = start_year

        year_starts: List[int] = [datetime(y, 1, 1).timestamp() for y in range(self.start_year, self.end_year)]

        self.year_starts: torch.nn.Buffer = torch.nn.Buffer(
            torch.asarray(year_starts).to(dtype=torch.int64),
        )

        self.month_days: torch.nn.Buffer = torch.nn.Buffer(
            torch.asarray([LEAP_YEAR_DAYS, NON_LEAP_YEAR_DAYS]),
        )

        self.month_offsets: torch.nn.Buffer = torch.nn.Buffer(
            torch.cumsum(self.month_days, dim=-1, dtype=torch.int64),
        )

        self.month_starts: torch.nn.Buffer = torch.nn.Buffer(
            torch.hstack(
                [
                    torch.zeros((2, 1), dtype=torch.int64),
                    self.month_offsets,
                ],
            ),
        )

    def get_rel_year(self: Self, ts: torch.LongTensor) -> torch.LongTensor:
        return torch.searchsorted(self.year_starts, ts, side="right") - 1

    def get_month(
        self: Self,
        is_year_leap: torch.LongTensor,
        day_of_the_year: torch.LongTensor,
    ) -> torch.LongTensor:
        assert is_year_leap.size() == day_of_the_year.size()
        leap: torch.LongTensor = torch.searchsorted(
            sorted_sequence=self.month_offsets[0, ...],
            input=day_of_the_year,
            side="right",
        )
        non_leap: torch.LongTensor = torch.searchsorted(
            sorted_sequence=self.month_offsets[1, ...],
            input=day_of_the_year,
            side="right",
        )
        return torch.where(is_year_leap, leap, non_leap)

    def forward(self: Self, ts: torch.LongTensor) -> TimestampEncoderOutput:
        # Year Section
        rel_year: torch.LongTensor = self.get_rel_year(ts)
        year: torch.LongTensor = rel_year + self.start_year
        ts_since_year: torch.LongTensor = ts - self.year_starts[rel_year]

        is_year_leap: torch.BoolTensor = is_leap_year(year)
        day_of_the_year: torch.LongTensor = ts_since_year // SECONDS_PER_DAY

        # Month Section
        month: torch.LongTensor = self.get_month(is_year_leap, day_of_the_year)
        day_of_the_month: torch.LongTensor = day_of_the_year - torch.where(
            is_year_leap,
            self.month_starts[0, month],
            self.month_starts[1, month],
        )

        # Week part
        days_since_epoch: torch.LongTensor = ts // SECONDS_PER_DAY
        day_of_the_week: torch.LongTensor = (days_since_epoch + 3) % 7

        # Time part
        seconds_in_day: torch.LongTensor = ts_since_year % SECONDS_PER_DAY
        hour_of_the_day: torch.LongTensor = seconds_in_day // 3600
        seconds_in_hour: torch.LongTensor = seconds_in_day % 3600
        minute_of_the_hour: torch.LongTensor = seconds_in_hour // 60

        return TimestampEncoderOutput(
            year=year,
            is_year_leap=is_year_leap,
            day_of_the_year=day_of_the_year,
            month_of_the_year=month,
            day_of_the_month=day_of_the_month,
            day_of_the_week=day_of_the_week,
            hour_of_the_day=hour_of_the_day,
            minute_of_the_hour=minute_of_the_hour,
        )
