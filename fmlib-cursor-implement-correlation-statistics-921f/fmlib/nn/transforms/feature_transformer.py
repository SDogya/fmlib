from typing import List

import torch

from fmlib.nn.transforms.modular.cast import Cast
from fmlib.nn.transforms.modular.filtering import FilteringIn
from fmlib.nn.transforms.modular.rename import Rename


def make_for_feature_transformer(
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

    cast = Cast(
        casting={
            "hidden_states": torch.float32,
        },
    )

    out_filtering = FilteringIn(
        patterns=[
            *bypass,
            "cat_features",
            "num_features",
            "hidden_states",
        ]
    )

    result = torch.nn.Sequential(
        in_filtering,
        rename,
        cast,
        out_filtering,
    )

    return result
