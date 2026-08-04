import string
from datetime import datetime

import pytest
import torch

from fmlib.data.io.implementation.indexing import get_mask, get_offsets
from fmlib.nn.blocks import DecoderBlock, EncoderBlock, PositionWiseFFN
from fmlib.nn.models import Trivan
from fmlib.nn.models.heads import TrivanClassificationHead
from fmlib.nn.transforms import TrivanTransform


def _generate_trivan_input_batch(
    generator: torch.Generator,
    batch_size: int,
    seq_len: int,
    padding_value: int,
    epk_id_col_name: str,
    report_dt_col_name: str,
    evt_dttm_col_name: str,
    mcc_token_col_name: str,
    brand_token_col_name: str,
    place_token_col_name: str,
    terminal_token_col_name: str,
    amt_token_col_name: str,
):
    flatten_mask = torch.ones(batch_size, dtype=torch.bool)
    lengths = torch.randint(
        low=1,
        high=seq_len,
        size=(batch_size,),
        dtype=torch.int64,
        generator=generator,
    )
    offsets = get_offsets(lengths)
    indices = torch.arange(batch_size, dtype=torch.int64)
    padding_mask, padded_indices = get_mask(indices, offsets, seq_len)

    def _generate_sequences(lb: int, rb: int):
        nonlocal padding_mask, padded_indices, padding_value, generator
        data = torch.randint(
            low=lb,
            high=rb,
            size=(offsets[-1].item(),),
            dtype=torch.int64,
            generator=generator,
        )
        unmasked_values = torch.take(data, padded_indices)
        return torch.where(padding_mask, unmasked_values, padding_value)

    batch = {
        epk_id_col_name: torch.randint(
            low=1_000_000_000,
            high=2_000_000_000,
            size=(batch_size,),
            dtype=torch.int64,
            generator=generator,
        ),
        f"{epk_id_col_name}_mask": flatten_mask,
        report_dt_col_name: torch.randint(
            low=int(datetime(2026, 1, 1).timestamp()),
            high=int(datetime(2030, 1, 1).timestamp()),
            size=(batch_size,),
            dtype=torch.int64,
            generator=generator,
        ),
        f"{report_dt_col_name}_mask": flatten_mask,
        evt_dttm_col_name: torch.sort(
            _generate_sequences(
                int(datetime(2022, 1, 1).timestamp()),
                int(datetime(2025, 8, 1).timestamp()),
            ),
            dim=-1,
        ).values,
        f"{evt_dttm_col_name}_mask": padding_mask,
        mcc_token_col_name: _generate_sequences(100, 367),
        f"{mcc_token_col_name}_mask": padding_mask,
        brand_token_col_name: _generate_sequences(379, 1632),
        f"{brand_token_col_name}_mask": padding_mask,
        place_token_col_name: _generate_sequences(1649, 5019),
        f"{place_token_col_name}_mask": padding_mask,
        terminal_token_col_name: _generate_sequences(5020, 5021),
        f"{terminal_token_col_name}_mask": padding_mask,
        amt_token_col_name: _generate_sequences(5030, 5049),
        f"{amt_token_col_name}_mask": padding_mask,
    }
    return batch


def _generate_random_string(length: int = 10, generator: torch.Generator | None = None):
    letters = string.ascii_lowercase
    choices = torch.randint(low=0, high=len(letters), size=(length,), generator=generator)
    return "".join([letters[choice] for choice in choices.cpu().tolist()])


@pytest.mark.parametrize("batch_size", [4, 8, 32, 64])
@pytest.mark.parametrize("seq_len", [4, 8, 32, 64])
@pytest.mark.parametrize("vocab_size", [5504])
@pytest.mark.parametrize("padding_value", [0, 1])
@pytest.mark.parametrize("output_layers_count", [4, 8])
@pytest.mark.parametrize("embedding_dim", [64])
@pytest.mark.parametrize("positional_encoding_num_buckets", [256])
@pytest.mark.parametrize("seed", [228, 1337])
def test_trasform_and_model_compatibility(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    padding_value: int,
    output_layers_count: int,
    embedding_dim: int,
    positional_encoding_num_buckets: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)

    def generate_random_string(min_len=5, max_len=10) -> str:
        length: int = torch.randint(low=min_len, high=max_len, size=(1,), generator=generator).cpu().item()
        return _generate_random_string(length, generator)

    epk_id_col_name = generate_random_string()
    report_dt_col_name = generate_random_string()
    evt_dttm_col_name = generate_random_string()
    mcc_token_col_name = generate_random_string()
    brand_token_col_name = generate_random_string()
    place_token_col_name = generate_random_string()
    terminal_token_col_name = generate_random_string()
    amt_token_col_name = generate_random_string()
    batch = _generate_trivan_input_batch(
        generator=generator,
        batch_size=batch_size,
        seq_len=seq_len,
        padding_value=padding_value,
        epk_id_col_name=epk_id_col_name,
        report_dt_col_name=report_dt_col_name,
        evt_dttm_col_name=evt_dttm_col_name,
        mcc_token_col_name=mcc_token_col_name,
        brand_token_col_name=brand_token_col_name,
        place_token_col_name=place_token_col_name,
        terminal_token_col_name=terminal_token_col_name,
        amt_token_col_name=amt_token_col_name,
    )
    transform = TrivanTransform(
        report_date_inp_name=report_dt_col_name,
        report_date_padding_mask_inp_name=f"{report_dt_col_name}_mask",
        timestamp_inp_name=evt_dttm_col_name,
        mcc_inp_name=mcc_token_col_name,
        brand_inp_name=brand_token_col_name,
        place_inp_name=place_token_col_name,
        terminal_inp_name=terminal_token_col_name,
        amount_inp_name=amt_token_col_name,
        padding_mask_inp_name=f"{amt_token_col_name}_mask",
        padding_value=padding_value,
        adapter_inp_names=("sm_model", "okko_model"),
        embedding_dim=embedding_dim,
        event_max_distance_in_hours=90 * 24,
        positional_encoding_num_buckets=positional_encoding_num_buckets,
    )
    transformed_batch = transform(batch)
    assert set(transformed_batch.keys()) == {
        "encoder_input",
        "decoder_input",
        "positional_encoding",
        "encoder_padding_mask",
        "adapter_input",
    }

    output_layers = [_generate_random_string() for _ in range(output_layers_count)]
    model = Trivan(
        vocab_size=vocab_size,
        hidden_dim=embedding_dim,
        positional_encoding_num_buckets=positional_encoding_num_buckets,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        adapter_layer_names=["sm_model", "okko_model"],
        output_layer_names=output_layers,
        adapter_ffn=PositionWiseFFN(embedding_dim, embedding_dim * 4, 0.1),
        input_encoder_block=EncoderBlock(
            embedding_dim,
            8,
            0.1,
            0.1,
            ffn=PositionWiseFFN(embedding_dim, embedding_dim * 4, 0.1),
        ),
        encoder_block=EncoderBlock(
            embedding_dim,
            8,
            0.1,
            0.1,
            ffn=PositionWiseFFN(embedding_dim, embedding_dim * 4, 0.1),
        ),
        decoder_block=DecoderBlock(
            embedding_dim,
            8,
            0.1,
            0.1,
            ffn=PositionWiseFFN(embedding_dim, embedding_dim * 4, 0.1),
        ),
        classification_head=TrivanClassificationHead(embedding_dim, 2),
    )

    model_output = model(**transformed_batch)

    assert set(model_output.keys()) == set(output_layers)
    for output_layer in output_layers:
        assert model_output[output_layer].shape == torch.Size([batch_size, 2])
        assert model_output[output_layer].shape == model_output[output_layers[0]].shape
