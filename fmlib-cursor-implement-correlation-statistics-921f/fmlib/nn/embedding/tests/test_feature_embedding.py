import pytest
import torch

from fmlib.nn.embedding.feature import FeatureEmbedding
from fmlib.nn.embedding.hidden_state_agg import LayerNormConcatenate, LayerNormSum

VOCAB_SIZE = 100
EMB_DIM = 32
HIDDEN_DIM = 64


def make_model(n_num: int, hidden_state_aggregator=None) -> FeatureEmbedding:
    return FeatureEmbedding(
        numerical_feature_count=n_num,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMB_DIM,
        hidden_state_aggregator=hidden_state_aggregator,
    )


def make_inputs(seed: int, batch_size: int, n_cat: int, n_num: int, hidden_dim: int):
    gen = torch.Generator().manual_seed(seed)
    cat = torch.randint(0, VOCAB_SIZE, (batch_size, n_cat), generator=gen)
    num = torch.rand(batch_size, n_num, generator=gen)
    hs = torch.rand(batch_size, hidden_dim, generator=gen)
    return cat, num, hs


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
def test_no_aggregator_output_shape(seed: int, batch_size: int, n_cat: int, n_num: int) -> None:
    model = make_model(n_num)
    cat, num, _ = make_inputs(seed, batch_size, n_cat, n_num, HIDDEN_DIM)
    out = model(cat, num)
    assert out.shape == (batch_size, n_cat + n_num, EMB_DIM)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
def test_no_aggregator_hidden_states_ignored(seed: int, batch_size: int, n_cat: int, n_num: int) -> None:
    model = make_model(n_num)
    cat, num, hs = make_inputs(seed, batch_size, n_cat, n_num, HIDDEN_DIM)
    out_without = model(cat, num)
    out_with = model(cat, num, hidden_states=hs)
    assert torch.equal(out_without, out_with)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
def test_single_concat_aggregator_adds_one_token(seed: int, batch_size: int, n_cat: int, n_num: int) -> None:
    agg = LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM)
    model = make_model(n_num, hidden_state_aggregator=agg)
    cat, num, hs = make_inputs(seed, batch_size, n_cat, n_num, HIDDEN_DIM)
    out = model(cat, num, hidden_states=hs)
    assert out.shape == (batch_size, n_cat + n_num + 1, EMB_DIM)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
def test_single_sum_aggregator_preserves_token_count(seed: int, batch_size: int, n_cat: int, n_num: int) -> None:
    agg = LayerNormSum(hidden_state_dim=EMB_DIM, embedding_dim=EMB_DIM)
    model = make_model(n_num, hidden_state_aggregator=agg)
    cat, num, hs = make_inputs(seed, batch_size, n_cat, n_num, EMB_DIM)
    out = model(cat, num, hidden_states=hs)
    assert out.shape == (batch_size, n_cat + n_num, EMB_DIM)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
def test_single_tensor_and_list_of_one_are_equal(seed: int, batch_size: int, n_cat: int, n_num: int) -> None:
    agg = LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM)
    model = make_model(n_num, hidden_state_aggregator=agg)
    cat, num, hs = make_inputs(seed, batch_size, n_cat, n_num, HIDDEN_DIM)
    out_tensor = model(cat, num, hidden_states=hs)
    out_list = model(cat, num, hidden_states=[hs])
    assert torch.allclose(out_tensor, out_list)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
@pytest.mark.parametrize("n_aggs", [1, 2, 3])
def test_multiple_concat_aggregators_add_n_tokens(
    seed: int, batch_size: int, n_cat: int, n_num: int, n_aggs: int
) -> None:
    aggs = [LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM) for _ in range(n_aggs)]
    model = make_model(n_num, hidden_state_aggregator=aggs)
    gen = torch.Generator().manual_seed(seed)
    cat = torch.randint(0, VOCAB_SIZE, (batch_size, n_cat), generator=gen)
    num = torch.rand(batch_size, n_num, generator=gen)
    hs_list = [torch.rand(batch_size, HIDDEN_DIM, generator=gen) for _ in range(n_aggs)]
    out = model(cat, num, hidden_states=hs_list)
    assert out.shape == (batch_size, n_cat + n_num + n_aggs, EMB_DIM)


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("n_cat,n_num", [(5, 3), (10, 5)])
def test_mixed_aggregator_types_concat_then_sum(seed: int, batch_size: int, n_cat: int, n_num: int) -> None:
    aggs = [
        LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM),
        LayerNormSum(hidden_state_dim=EMB_DIM, embedding_dim=EMB_DIM),
    ]
    model = make_model(n_num, hidden_state_aggregator=aggs)
    gen = torch.Generator().manual_seed(seed)
    cat = torch.randint(0, VOCAB_SIZE, (batch_size, n_cat), generator=gen)
    num = torch.rand(batch_size, n_num, generator=gen)
    hs_list = [torch.rand(batch_size, HIDDEN_DIM, generator=gen), torch.rand(batch_size, EMB_DIM, generator=gen)]
    out = model(cat, num, hidden_states=hs_list)
    # LayerNormConcatenate adds 1 token, LayerNormSum keeps count — net +1
    assert out.shape == (batch_size, n_cat + n_num + 1, EMB_DIM)


@pytest.mark.parametrize("n_aggs", [1, 2, 3])
def test_aggregators_stored_as_module_list(n_aggs: int) -> None:
    aggs = [LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM) for _ in range(n_aggs)]
    model = make_model(3, hidden_state_aggregator=aggs)
    assert isinstance(model.hidden_state_aggregators, torch.nn.ModuleList)
    assert len(model.hidden_state_aggregators) == n_aggs


def test_no_aggregator_stores_none() -> None:
    model = make_model(3, hidden_state_aggregator=None)
    assert model.hidden_state_aggregators is None


def test_raises_when_hidden_states_missing_with_single_aggregator() -> None:
    agg = LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM)
    model = make_model(3, hidden_state_aggregator=agg)
    cat = torch.randint(0, VOCAB_SIZE, (8, 5))
    num = torch.rand(8, 3)
    with pytest.raises(ValueError, match="hidden_states must be provided"):
        model(cat, num)


def test_raises_when_hidden_states_missing_with_multiple_aggregators() -> None:
    aggs = [LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM) for _ in range(2)]
    model = make_model(3, hidden_state_aggregator=aggs)
    cat = torch.randint(0, VOCAB_SIZE, (8, 5))
    num = torch.rand(8, 3)
    with pytest.raises(ValueError, match="hidden_states must be provided"):
        model(cat, num)


@pytest.mark.parametrize("n_aggs,n_hs", [(2, 1), (2, 3), (3, 1), (3, 2)])
def test_raises_when_hidden_states_count_mismatch(n_aggs: int, n_hs: int) -> None:
    aggs = [LayerNormConcatenate(hidden_state_dim=HIDDEN_DIM, embedding_dim=EMB_DIM) for _ in range(n_aggs)]
    model = make_model(3, hidden_state_aggregator=aggs)
    cat = torch.randint(0, VOCAB_SIZE, (8, 5))
    num = torch.rand(8, 3)
    hs_list = [torch.rand(8, HIDDEN_DIM) for _ in range(n_hs)]
    with pytest.raises(ValueError, match="Number of hidden_states"):
        model(cat, num, hidden_states=hs_list)
