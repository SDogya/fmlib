from typing import Any, List, Self

import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.modular.cast import Cast
from fmlib.nn.transforms.modular.filtering import FilteringIn
from fmlib.nn.transforms.modular.rename import Rename


class DummyColumn(torch.nn.Module):
    """Создаёт колонкуЮ заполненную одним значением."""

    def __init__(
        self: Self,
        column: str,
        reference_column: str = "epk_id",
        dtype: torch.dtype | None = None,
        bypass: bool = True,
        value: Any = 1.0,
    ) -> None:
        super().__init__()

        self.column: str = column
        self.reference_column: str = reference_column
        self.dtype: torch.dtype | None = dtype
        self.bypass: bool = bypass
        self.value: Any = value

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        reference: torch.Tensor = batch[self.reference_column]
        dtype: torch.dtype = self.dtype or reference.dtype
        batch_size: int = reference.size(0)
        column: torch.Tensor = torch.full(
            device=reference.device,
            fill_value=self.value,
            size=(batch_size,),
            dtype=dtype,
        )

        result: GeneralBatch = {self.column: column}
        if self.bypass:
            result.update(batch)
        return result


def make_for_uplift_feature_transformer(
    channel_type: int = 0,
    cat_features_name: str = "cat_features",
    num_features_name: str = "num_features",
    seq_hidden_state_name: str = "seq_hidden_state",
    bypass: List[str] | None = None,
) -> torch.nn.Module:
    """
    Создаёт модуль для трансформации табличных данных за один вызов функции.

    Аргументы:
        cat_features_name (str): Имя тензора с категориальными признаками.
        num_features_name (str): Имя тензора с числовыми признаками.
        seq_hidden_state_name (str): Имя тензора с скрытыми состояниями.
        bypass (List[str] | None): Список имен тензоров, которые не нужно преобразовывать.
            `None` - преобразуются все тензоры, кроме `epk_id`. По умолчанию - `None`.
    """
    if bypass is None:
        bypass = ["epk_id"]

    in_filtering = FilteringIn(
        patterns=[
            *bypass,
            cat_features_name,
            num_features_name,
            seq_hidden_state_name,
        ]
    )

    rename = Rename(
        mapping={
            cat_features_name: "cat_features",
            num_features_name: "num_features",
            seq_hidden_state_name: "hidden_states",
        },
        bypass=bypass,
    )

    dummy = DummyColumn(
        reference_column="cat_features",
        value=channel_type,
        dtype=torch.int64,
        column="group",
    )

    cast = Cast(
        casting={
            "hidden_states": torch.float32,
        },
    )

    out_filtering = FilteringIn(
        patterns=[
            *bypass,
            "group",
            "cat_features",
            "num_features",
            "hidden_states",
        ]
    )

    result = torch.nn.Sequential(
        in_filtering,
        rename,
        dummy,
        cast,
        out_filtering,
    )

    return result
