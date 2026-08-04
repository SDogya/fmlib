import torch

from fmlib.nn.transforms.modular import (
    ChangePaddingTransform,
    CutSequences,
    FilteringIn,
    JoinSequences,
    Map,
    PatchAndMask,
    Rename,
    UniformMaskSequences,
)


def make_for_dlt(
    sequences: list[str] | None = None,
    timestamp_name: str = "date_stamp",
    bypass: list[str] | str | None = None,
    max_time_delta: int = 500 * 24 * 60 * 60,
    mask_column: str = "date_stamp_mask",
    max_seq_len: int = 2048,
    padding_value: int = 0,
    sequences_name="sequences",
    mask_proba: float = 0.0,
) -> torch.nn.Module:
    """
    Создаёт пайплайн для трансформа DLT модели.

    Аргументы:
        sequences (list[str] | None): Список последовательностей, которые необходимо объединить.
            По умолчанию - None, что превращается в ["dir_token", "ecom_token",
                "mcc_token", "brand_token", "city_token", "amt_token"].
        timestamp_name (str): Имя последовательности с временем. По умолчанию - "date_stamp".
        bypass (list[str] | str | None): Список фичей, которые не будут объединены.
            По умолчанию - None, что превращается в ["epk_id", "report_dt"].
        max_seq_len (int): Максимальная длина последовательности. По умолчанию - 2048.
        max_time_delta (int): Максимальное время между последним и текущим событием.
            По умолчанию - 500 дней в секундах.
        sequences_name (str): Имя объединенной последовательности. По умолчанию - "sequences".
        mask_proba (float): Вероятность маскирования последовательности (используется для обучения).
            По умолчанию - 0.0, т.е. дополнительного маскирования не происходит.
    """
    if sequences is None:
        sequences = ["dir_token", "ecom_token", "mcc_token", "brand_token", "city_token", "amt_token"]

    if bypass is None:
        bypass = ["epk_id", "report_dt"]
    elif isinstance(bypass, str):
        bypass = [bypass]
    else:
        bypass = list(bypass)
    assert isinstance(bypass, list)

    all_sequences: list[str] = [timestamp_name, mask_column, *sequences]
    all_sequences = sorted(set(all_sequences))

    if mask_proba > 0.0:
        pre_masking = torch.nn.Sequential(
            FilteringIn(all_sequences),
            UniformMaskSequences(
                keep_last=1,
                proba=mask_proba,
                sequences=all_sequences,
                padding_value=padding_value,
                ref_sequence=timestamp_name,
                min_seq_len=max_seq_len // 4,
            ),
        )
    else:
        assert mask_proba == 0.0
        pre_masking = FilteringIn(all_sequences)

    result = Map(
        [
            FilteringIn(bypass),
            torch.nn.Sequential(
                pre_masking,
                ChangePaddingTransform(mask_column=mask_column),
                CutSequences(
                    max_time_delta=max_time_delta,
                    timestamp_name=timestamp_name,
                    padding_value=padding_value,
                    max_seq_len=max_seq_len,
                    sequences=sequences,
                ),
                Map(
                    [
                        FilteringIn([timestamp_name]),
                        JoinSequences(
                            sequences=sequences,
                            sequences_name=sequences_name,
                        ),
                        PatchAndMask(
                            timestamp_name=timestamp_name,
                            padding_value=padding_value,
                        ),
                    ]
                ),
            ),
        ]
    )

    return torch.nn.Sequential(
        result,
        FilteringIn(
            [
                *bypass,
                sequences_name,
                timestamp_name,
                "encoder_mask",
                "patches",
                "patches_pos_tensor",
                "encoder_cross_mask",
                "decoder_cross_mask",
                "patches_mask",
            ]
        ),
        Rename(
            {
                sequences_name: "data",
                timestamp_name: "timestamps",
            }
        ),
    )
