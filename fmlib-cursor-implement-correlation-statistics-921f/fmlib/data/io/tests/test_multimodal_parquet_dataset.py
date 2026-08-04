"""
Тесты для ModalityConfig, _merge_multimodal_batch и MultimodalParquetDataset.

Структура:
  - Вспомогательные типы / функции / генераторы эталонных данных на уровне модуля.
  - Параметризованные юнит-тесты для функции слияния.
  - Параметризованные интеграционные тесты для датасета через временные Parquet-файлы.
"""

import re
import warnings
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fmlib.constants.multimodal import EVENT_IDS_COLUMN, EVT_DTTM_COLUMN
from fmlib.data.io.multimodal_batches_iterator import (
    ModalityConfig,
    MultimodalBatchesIterator,
    _merge_multimodal_batch,
)
from fmlib.data.io.multimodal_parquet_dataset import MultimodalParquetDataset

# ---------------------------------------------------------------------------
# Константы уровня модуля
# ---------------------------------------------------------------------------

_PADDING = 0

# Простые двух-модальные сценарии: (времена_a, признаки_a, макс_a, времена_b, признаки_b, макс_b)
# Каждый сценарий — это одна строка батча
TwoModRow = Tuple[
    List[float], List[int], int,   # time_a, feat_a, max_len_a
    List[float], List[int], int,   # time_b, feat_b, max_len_b
]

TWO_MOD_ROWS: List[TwoModRow] = [
    # Все события mod_a раньше mod_b
    ([1.0, 2.0], [10, 20], 8, [3.0, 4.0], [30, 40], 8),
    # Чередующийся порядок
    ([1.0, 3.0], [10, 20], 8, [2.0, 4.0], [30, 40], 8),
    # Все события mod_b раньше mod_a
    ([5.0, 6.0], [50, 60], 8, [1.0, 2.0], [10, 20], 8),
    # Разные длины: mod_a длиннее
    ([1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4], 8, [5.0], [5], 8),
    # max_length обрезает mod_a
    ([1.0, 2.0, 3.0, 4.0, 5.0], [1, 2, 3, 4, 5], 3, [6.0, 7.0], [6, 7], 8),
]

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


@dataclass
class FakeReplicasInfo:
    num_replicas: int = 1
    curr_replica: int = 0


def _f32(rows: List) -> pa.Array:
    return pa.array(rows, type=pa.list_(pa.float32()))


def _i64(rows: List) -> pa.Array:
    return pa.array(rows, type=pa.list_(pa.int64()))


def _make_two_mod_batch(
    time_a: List[List[float]],
    feat_a: List[List[int]],
    time_b: List[List[float]],
    feat_b: List[List[int]],
) -> pa.RecordBatch:
    return pa.table({
        "time_a": _f32(time_a),
        "feat_a": _i64(feat_a),
        "time_b": _f32(time_b),
        "feat_b": _i64(feat_b),
    }).to_batches()[0]


def _two_mod_modalities(max_a: int = 8, max_b: int = 8) -> Dict[str, ModalityConfig]:
    return {
        "mod_a": ModalityConfig(event_id=0, time_column="time_a", columns=["feat_a"], max_length=max_a),
        "mod_b": ModalityConfig(event_id=1, time_column="time_b", columns=["feat_b"], max_length=max_b),
    }


def _merge_two_mod(
    time_a: List[List[float]],
    feat_a: List[List[int]],
    time_b: List[List[float]],
    feat_b: List[List[int]],
    max_a: int = 8,
    max_b: int = 8,
    random_slicing: bool = False,
    generator: Optional[torch.Generator] = None,
    flat_ids: Optional[List[int]] = None,
) -> pa.Table:
    """Удобная обёртка для вызова _merge_multimodal_batch в двух-модальном случае."""
    cols: Dict[str, pa.Array] = {
        "time_a": _f32(time_a),
        "feat_a": _i64(feat_a),
        "time_b": _f32(time_b),
        "feat_b": _i64(feat_b),
    }
    flat_columns: List[str] = []
    if flat_ids is not None:
        cols["row_id"] = pa.array(flat_ids, type=pa.int64())
        flat_columns = ["row_id"]
    batch = pa.table(cols).to_batches()[0]
    return _merge_multimodal_batch(
        batch=batch,
        modalities=_two_mod_modalities(max_a, max_b),
        flat_columns=flat_columns,
        random_slicing=random_slicing,
        generator=generator,
    )


def _ground_truth_merge(
    times: Dict[str, List[float]],
    feats: Dict[str, Dict[str, List[int]]],
    modalities: Dict[str, ModalityConfig],
    random_slicing: bool = False,
    start_offsets: Optional[Dict[str, int]] = None,
) -> Dict[str, List]:
    """Эталонная реализация слияния одной строки на чистом Python+NumPy.

    Аргументы:
        times: Словарь имя_модальности → список временных меток.
        feats: Словарь имя_модальности → {имя_колонки → список значений}.
        modalities: Конфигурации модальностей.
        random_slicing: Если True, используются start_offsets вместо взятия хвоста.
        start_offsets: Заданные начальные позиции среза (только при random_slicing=True).

    Возвращает:
        Словарь с ключами EVT_DTTM_COLUMN, EVENT_IDS_COLUMN и именами признаков.
    """
    names = list(modalities.keys())
    sliced_times: Dict[str, np.ndarray] = {}
    sliced_feats: Dict[str, Dict[str, np.ndarray]] = {}

    for name in names:
        cfg = modalities[name]
        t = np.asarray(times[name], dtype=np.float64)
        seq_len = len(t)
        if seq_len > cfg.max_length:
            if random_slicing and start_offsets is not None:
                s = start_offsets[name]
            else:
                s = seq_len - cfg.max_length
            t = t[s : s + cfg.max_length]
            sliced_feats[name] = {
                col: np.asarray(feats[name][col])[s : s + cfg.max_length]
                for col in cfg.columns
            }
        else:
            sliced_feats[name] = {col: np.asarray(feats[name][col]) for col in cfg.columns}
        sliced_times[name] = t

    sizes = [len(sliced_times[n]) for n in names]
    total = sum(sizes)
    pref = [0] + list(np.cumsum(sizes))

    if total == 0:
        all_cols = [col for cfg in modalities.values() for col in cfg.columns]
        return {EVT_DTTM_COLUMN: [], EVENT_IDS_COLUMN: [], **{c: [] for c in all_cols}}

    all_times = np.concatenate([sliced_times[n] for n in names])
    argsort = np.argsort(all_times, kind="stable")

    event_ids_arr = np.empty(total, dtype=np.int64)
    for i, name in enumerate(names):
        event_ids_arr[pref[i] : pref[i + 1]] = modalities[name].event_id

    result: Dict[str, List] = {
        EVT_DTTM_COLUMN: all_times[argsort].tolist(),
        EVENT_IDS_COLUMN: event_ids_arr[argsort].tolist(),
    }

    for i, (name, cfg) in enumerate(modalities.items()):
        for col in cfg.columns:
            feat = sliced_feats[name][col]
            left_pad, right_pad = pref[i], total - pref[i + 1]
            padded = np.pad(feat, (left_pad, right_pad))
            result[col] = padded[argsort].tolist()

    return result


def _write_two_mod_parquet(
    path: str,
    n_rows: int = 4,
    seed: int = 0,
    min_len: int = 2,
    max_len: int = 6,
) -> None:
    """Записывает Parquet с двумя модальностями в указанный путь."""
    rng = np.random.default_rng(seed)
    len_a = rng.integers(min_len, max_len + 1, size=n_rows)
    len_b = rng.integers(min_len, max_len + 1, size=n_rows)
    rows_ta = [sorted(rng.uniform(0, 20, size=int(la)).tolist()) for la in len_a]
    rows_tb = [sorted(rng.uniform(0, 20, size=int(lb)).tolist()) for lb in len_b]
    rows_fa = [rng.integers(1, 100, size=int(la)).tolist() for la in len_a]
    rows_fb = [rng.integers(1, 100, size=int(lb)).tolist() for lb in len_b]
    pq.write_table(
        pa.table({
            "row_id": pa.array(list(range(n_rows)), type=pa.int64()),
            "feat_a": pa.array(rows_fa, type=pa.list_(pa.int64())),
            "feat_b": pa.array(rows_fb, type=pa.list_(pa.int64())),
            "time_a": pa.array([[float(x) for x in r] for r in rows_ta], type=pa.list_(pa.float32())),
            "time_b": pa.array([[float(x) for x in r] for r in rows_tb], type=pa.list_(pa.float32())),
        }),
        path,
    )


def _make_dataset(
    source: str,
    seq_len: int = 16,
    max_a: int = 8,
    max_b: int = 8,
    batch_size: int = 1,
    partition_size: int = 32,
    random_slicing: bool = False,
    generator: Optional[torch.Generator] = None,
    replicas_info: Optional[FakeReplicasInfo] = None,
) -> MultimodalParquetDataset:
    metadata: Dict[str, Any] = {
        "row_id": {},
        "feat_a": {"sequential": True, "sequence_length": seq_len, "padding": _PADDING},
        "feat_b": {"sequential": True, "sequence_length": seq_len, "padding": _PADDING},
        EVT_DTTM_COLUMN: {"sequential": True, "sequence_length": seq_len, "padding": _PADDING},
        EVENT_IDS_COLUMN: {"sequential": True, "sequence_length": seq_len, "padding": _PADDING},
    }
    kwargs: Dict[str, Any] = dict(
        source=source,
        metadata=metadata,
        modalities=_two_mod_modalities(max_a, max_b),
        partition_size=partition_size,
        batch_size=batch_size,
        random_slicing=random_slicing,
    )
    if generator is not None:
        kwargs["generator"] = generator
    if replicas_info is not None:
        kwargs["replicas_info"] = replicas_info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MultimodalParquetDataset(**kwargs)


# ---------------------------------------------------------------------------
# ModalityConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("columns_input", [["a", "b"], ("a", "b"), iter(["a", "b"])])
def test_modality_config_columns_always_list(columns_input: Any) -> None:
    cfg = ModalityConfig(event_id=0, time_column="t", columns=columns_input, max_length=4)
    assert isinstance(cfg.columns, list)
    assert cfg.columns == ["a", "b"]


def test_modality_config_from_dict() -> None:
    cfg = ModalityConfig(**{"event_id": 3, "time_column": "ts", "columns": ["x"], "max_length": 5})
    assert cfg.event_id == 3
    assert cfg.max_length == 5


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – временной порядок
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", TWO_MOD_ROWS)
def test_merge_evt_dttm_is_sorted(row: TwoModRow) -> None:
    ta, fa, max_a, tb, fb, max_b = row
    table = _merge_two_mod([ta], [fa], [tb], [fb], max_a, max_b)
    result = table.column(EVT_DTTM_COLUMN).to_pylist()[0]
    assert result == sorted(result), f"evt_dttm не отсортирован: {result}"


@pytest.mark.parametrize("n_rows", [1, 3, 5])
def test_merge_all_rows_sorted(n_rows: int) -> None:
    rng = np.random.default_rng(42)
    time_a = [sorted(rng.uniform(0, 10, size=3).tolist()) for _ in range(n_rows)]
    time_b = [sorted(rng.uniform(0, 10, size=2).tolist()) for _ in range(n_rows)]
    feat_a = [[1, 2, 3]] * n_rows
    feat_b = [[4, 5]] * n_rows
    table = _merge_two_mod(time_a, feat_a, time_b, feat_b)
    for row_times in table.column(EVT_DTTM_COLUMN).to_pylist():
        assert row_times == sorted(row_times)


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – корректность event_ids и нулевой паддинг
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", TWO_MOD_ROWS)
def test_merge_values_match_ground_truth(row: TwoModRow) -> None:
    """Результат сравниваем с эталонной реализацией."""
    ta, fa, max_a, tb, fb, max_b = row
    table = _merge_two_mod([ta], [fa], [tb], [fb], max_a, max_b)
    gt = _ground_truth_merge(
        times={"mod_a": ta, "mod_b": tb},
        feats={"mod_a": {"feat_a": fa}, "mod_b": {"feat_b": fb}},
        modalities=_two_mod_modalities(max_a, max_b),
    )
    assert table.column(EVENT_IDS_COLUMN).to_pylist()[0] == gt[EVENT_IDS_COLUMN]
    assert table.column("feat_a").to_pylist()[0] == gt["feat_a"]
    assert table.column("feat_b").to_pylist()[0] == gt["feat_b"]
    assert table.column(EVT_DTTM_COLUMN).to_pylist()[0] == pytest.approx(gt[EVT_DTTM_COLUMN], abs=1e-4)


def test_merge_zero_padding_at_cross_modality_positions() -> None:
    """На позициях одной модальности признаки другой должны быть нулевыми."""
    # time_a < time_b, чередование: a, b, a, b
    table = _merge_two_mod([[1.0, 3.0]], [[11, 22]], [[2.0, 4.0]], [[33, 44]])
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    fa = table.column("feat_a").to_pylist()[0]
    fb = table.column("feat_b").to_pylist()[0]
    for i, eid in enumerate(eids):
        if eid == 0:
            assert fb[i] == 0, f"feat_b должен быть 0 на позиции mod_a (i={i})"
        else:
            assert fa[i] == 0, f"feat_a должен быть 0 на позиции mod_b (i={i})"


def test_merge_feature_values_preserved_at_own_positions() -> None:
    """Значения признаков на позициях своей модальности должны совпадать с исходными."""
    table = _merge_two_mod([[1.0, 3.0]], [[11, 22]], [[2.0, 4.0]], [[33, 44]])
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    fa = table.column("feat_a").to_pylist()[0]
    fb = table.column("feat_b").to_pylist()[0]
    assert [fa[i] for i, e in enumerate(eids) if e == 0] == [11, 22]
    assert [fb[i] for i, e in enumerate(eids) if e == 1] == [33, 44]


def test_merge_total_length_equals_sum_of_modality_lengths() -> None:
    """Длина объединённой строки = сумма длин срезанных модальностей."""
    table = _merge_two_mod([[1.0, 2.0, 3.0]], [[1, 2, 3]], [[4.0, 5.0]], [[4, 5]])
    assert len(table.column(EVT_DTTM_COLUMN).to_pylist()[0]) == 5


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – срезание последовательностей
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_a,max_b,expected_total", [
    (3, 2, 5),   # mod_a обрезается до 3, mod_b — до 2
    (1, 1, 2),   # оба обрезаются до 1
    (8, 8, 5),   # max_length >= длины → без срезания (3+2=5)
])
def test_merge_tail_slicing_total_length(max_a: int, max_b: int, expected_total: int) -> None:
    table = _merge_two_mod([[1.0, 2.0, 3.0]], [[1, 2, 3]], [[4.0, 5.0]], [[4, 5]], max_a, max_b)
    assert len(table.column(EVT_DTTM_COLUMN).to_pylist()[0]) == expected_total


def test_merge_tail_slicing_takes_last_events() -> None:
    """При tail-срезании берётся хвост последовательности."""
    # time_a = [1,2,3,4,5], max=3 → должны остаться [3,4,5]
    table = _merge_two_mod([[1.0, 2.0, 3.0, 4.0, 5.0]], [[1, 2, 3, 4, 5]], [[10.0]], [[9]], max_a=3)
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    fa = table.column("feat_a").to_pylist()[0]
    actual_a_vals = [fa[i] for i, e in enumerate(eids) if e == 0]
    assert actual_a_vals == [3, 4, 5]


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_merge_random_slicing_same_seed_same_result(seed: int) -> None:
    """Два вызова с одним seed должны давать одинаковый результат."""
    g1 = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed)
    times = [[float(i) for i in range(20)]]
    feats = [list(range(20))]
    t1 = _merge_two_mod(times, feats, times, feats, max_a=5, max_b=5, random_slicing=True, generator=g1)
    t2 = _merge_two_mod(times, feats, times, feats, max_a=5, max_b=5, random_slicing=True, generator=g2)
    assert t1.column(EVT_DTTM_COLUMN).to_pylist() == t2.column(EVT_DTTM_COLUMN).to_pylist()


def test_merge_random_slicing_different_seeds_produce_variety() -> None:
    """Разные seed должны давать разные срезы (с высокой вероятностью)."""
    times = [[float(i) for i in range(50)]]
    feats = [list(range(50))]
    results = set()
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        t = _merge_two_mod(times, feats, times, feats, max_a=5, max_b=5, random_slicing=True, generator=g)
        results.add(tuple(t.column(EVT_DTTM_COLUMN).to_pylist()[0]))
    assert len(results) > 1, "Разные seed не дают разных результатов"


def test_merge_random_slicing_result_has_correct_length() -> None:
    g = torch.Generator().manual_seed(99)
    times = [[float(i) for i in range(10)]]
    feats = [list(range(10))]
    table = _merge_two_mod(times, feats, times, feats, max_a=4, max_b=3, random_slicing=True, generator=g)
    assert len(table.column(EVT_DTTM_COLUMN).to_pylist()[0]) == 7


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – краевые случаи пустых модальностей
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_mod", ["a", "b"])
def test_merge_one_modality_empty(empty_mod: str) -> None:
    """Если одна модальность пустая, результат содержит только события другой."""
    if empty_mod == "a":
        table = _merge_two_mod([[]], [[]], [[1.0, 2.0]], [[10, 20]])
        assert all(e == 1 for e in table.column(EVENT_IDS_COLUMN).to_pylist()[0])
        assert table.column("feat_a").to_pylist()[0] == [0, 0]
    else:
        table = _merge_two_mod([[1.0, 2.0]], [[10, 20]], [[]], [[]])
        assert all(e == 0 for e in table.column(EVENT_IDS_COLUMN).to_pylist()[0])
        assert table.column("feat_b").to_pylist()[0] == [0, 0]


def test_merge_both_modalities_empty() -> None:
    """Если обе модальности пусты, результат содержит пустые списки."""
    table = _merge_two_mod([[]], [[]], [[]], [[]])
    assert table.column(EVT_DTTM_COLUMN).to_pylist()[0] == []
    assert table.column(EVENT_IDS_COLUMN).to_pylist()[0] == []
    assert table.column("feat_a").to_pylist()[0] == []
    assert table.column("feat_b").to_pylist()[0] == []


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – три модальности
# ---------------------------------------------------------------------------


def test_merge_three_modalities_order_and_ids() -> None:
    """Три модальности: проверяем порядок времён и идентификаторы."""
    batch = pa.table({
        "time_a": _f32([[1.0, 4.0]]),
        "time_b": _f32([[2.0, 5.0]]),
        "time_c": _f32([[3.0, 6.0]]),
        "feat_a": _i64([[10, 40]]),
        "feat_b": _i64([[20, 50]]),
        "feat_c": _i64([[30, 60]]),
    }).to_batches()[0]
    modalities = {
        "m0": ModalityConfig(event_id=0, time_column="time_a", columns=["feat_a"], max_length=8),
        "m1": ModalityConfig(event_id=1, time_column="time_b", columns=["feat_b"], max_length=8),
        "m2": ModalityConfig(event_id=2, time_column="time_c", columns=["feat_c"], max_length=8),
    }
    table = _merge_multimodal_batch(batch, modalities, flat_columns=[], random_slicing=False, generator=None)
    times = table.column(EVT_DTTM_COLUMN).to_pylist()[0]
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    assert times == sorted(times)
    assert eids == [0, 1, 2, 0, 1, 2]


def test_merge_three_modalities_zero_padding() -> None:
    """У трёх модальностей каждая признаковая колонка нулевая на позициях остальных."""
    batch = pa.table({
        "time_a": _f32([[1.0, 4.0]]),
        "time_b": _f32([[2.0, 5.0]]),
        "time_c": _f32([[3.0, 6.0]]),
        "feat_a": _i64([[10, 40]]),
        "feat_b": _i64([[20, 50]]),
        "feat_c": _i64([[30, 60]]),
    }).to_batches()[0]
    modalities = {
        "m0": ModalityConfig(event_id=0, time_column="time_a", columns=["feat_a"], max_length=8),
        "m1": ModalityConfig(event_id=1, time_column="time_b", columns=["feat_b"], max_length=8),
        "m2": ModalityConfig(event_id=2, time_column="time_c", columns=["feat_c"], max_length=8),
    }
    table = _merge_multimodal_batch(batch, modalities, flat_columns=[], random_slicing=False, generator=None)
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    for col, own_id in [("feat_a", 0), ("feat_b", 1), ("feat_c", 2)]:
        vals = table.column(col).to_pylist()[0]
        for i, eid in enumerate(eids):
            if eid != own_id:
                assert vals[i] == 0, f"{col}[{i}] должен быть 0, так как event_id={eid}≠{own_id}"


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – стабильная сортировка при совпадении времён
# ---------------------------------------------------------------------------


def test_merge_identical_timestamps_stable_sort() -> None:
    """При одинаковых временных метках порядок модальностей сохраняется (stable argsort)."""
    # Обе метки t=1.0 → mod_a (event_id=0) идёт первой, так как она первая в конкатенации
    table = _merge_two_mod([[1.0]], [[99]], [[1.0]], [[77]])
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    assert eids == [0, 1], f"Ожидался [0, 1], получено {eids}"


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – несколько признаков в одной модальности
# ---------------------------------------------------------------------------


def test_merge_multiple_features_per_modality() -> None:
    """Несколько признаковых колонок одной модальности — все нулевые на чужих позициях."""
    batch = pa.table({
        "time_a": _f32([[1.0, 3.0]]),
        "time_b": _f32([[2.0]]),
        "feat_a1": _i64([[10, 20]]),
        "feat_a2": _i64([[100, 200]]),
        "feat_b":  _i64([[30]]),
    }).to_batches()[0]
    modalities = {
        "ma": ModalityConfig(event_id=0, time_column="time_a", columns=["feat_a1", "feat_a2"], max_length=8),
        "mb": ModalityConfig(event_id=1, time_column="time_b", columns=["feat_b"], max_length=8),
    }
    table = _merge_multimodal_batch(batch, modalities, flat_columns=[], random_slicing=False, generator=None)
    eids = table.column(EVENT_IDS_COLUMN).to_pylist()[0]
    for pos, eid in enumerate(eids):
        if eid == 1:  # позиция mod_b
            assert table.column("feat_a1").to_pylist()[0][pos] == 0
            assert table.column("feat_a2").to_pylist()[0][pos] == 0


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – dtype выходных колонок
# ---------------------------------------------------------------------------


def test_merge_evt_dttm_dtype_is_float32() -> None:
    """evt_dttm должен быть float32, чтобы не было dtype-ошибок в модели."""
    table = _merge_two_mod([[1.0, 2.0]], [[1, 2]], [[3.0]], [[3]])
    assert table.column(EVT_DTTM_COLUMN).type == pa.list_(pa.float32())


def test_merge_event_ids_dtype_is_int64() -> None:
    table = _merge_two_mod([[1.0]], [[1]], [[2.0]], [[2]])
    assert table.column(EVENT_IDS_COLUMN).type == pa.list_(pa.int64())


def test_merge_feature_dtype_preserved() -> None:
    """Тип признаковых колонок сохраняется из исходного Parquet."""
    batch = pa.table({
        "time_a": _f32([[1.0]]),
        "time_b": _f32([[2.0]]),
        "feat_a": pa.array([[10]], type=pa.list_(pa.int32())),
        "feat_b": pa.array([[20.0]], type=pa.list_(pa.float32())),
    }).to_batches()[0]
    modalities = {
        "ma": ModalityConfig(event_id=0, time_column="time_a", columns=["feat_a"], max_length=8),
        "mb": ModalityConfig(event_id=1, time_column="time_b", columns=["feat_b"], max_length=8),
    }
    table = _merge_multimodal_batch(batch, modalities, flat_columns=[], random_slicing=False, generator=None)
    assert table.column("feat_a").type == pa.list_(pa.int32())
    assert table.column("feat_b").type == pa.list_(pa.float32())


# ---------------------------------------------------------------------------
# _merge_multimodal_batch – плоские колонки
# ---------------------------------------------------------------------------


def test_merge_flat_column_passthrough() -> None:
    """Плоские колонки передаются в результат без изменений."""
    table = _merge_two_mod([[1.0]], [[10]], [[2.0]], [[20]], flat_ids=[42])
    assert table.column("row_id").to_pylist() == [42]


def test_merge_multiple_flat_columns_passthrough() -> None:
    batch = pa.table({
        "time_a": _f32([[1.0]]),
        "time_b": _f32([[2.0]]),
        "feat_a": _i64([[10]]),
        "feat_b": _i64([[20]]),
        "user_id": pa.array([111], type=pa.int64()),
        "label":   pa.array([1],   type=pa.int64()),
    }).to_batches()[0]
    modalities = _two_mod_modalities()
    table = _merge_multimodal_batch(batch, modalities, flat_columns=["user_id", "label"], random_slicing=False, generator=None)
    assert table.column("user_id").to_pylist() == [111]
    assert table.column("label").to_pylist() == [1]


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – форма и типы выходных тензоров
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [8, 16, 32])
def test_dataset_output_shapes(seq_len: int) -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet", n_rows=2)
        ds = _make_dataset(tmpdir, seq_len=seq_len)
        batch = next(iter(ds))
        assert batch["feat_a"].shape == (1, seq_len)
        assert batch["feat_b"].shape == (1, seq_len)
        assert batch[EVT_DTTM_COLUMN].shape == (1, seq_len)
        assert batch[EVENT_IDS_COLUMN].shape == (1, seq_len)
        assert batch["row_id"].shape == (1,)


def test_dataset_evt_dttm_is_float32() -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet")
        ds = _make_dataset(tmpdir)
        batch = next(iter(ds))
        assert batch[EVT_DTTM_COLUMN].dtype == torch.float32, (
            "evt_dttm должен быть float32 — иначе модель выдаёт dtype-ошибку"
        )


def test_dataset_event_ids_is_int64() -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet")
        ds = _make_dataset(tmpdir)
        batch = next(iter(ds))
        assert batch[EVENT_IDS_COLUMN].dtype == torch.int64


def test_dataset_masks_have_bool_dtype() -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet")
        ds = _make_dataset(tmpdir)
        batch = next(iter(ds))
        for col in ["feat_a", "feat_b", EVT_DTTM_COLUMN, EVENT_IDS_COLUMN]:
            assert batch[f"{col}_mask"].dtype == torch.bool


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – корректность содержимого батча
# ---------------------------------------------------------------------------


def test_dataset_output_temporal_order() -> None:
    """Все временные метки (не паддинг) в каждом батче должны быть упорядочены."""
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet", n_rows=5)
        ds = _make_dataset(tmpdir, batch_size=1)
        for batch in ds:
            mask = batch[f"{EVT_DTTM_COLUMN}_mask"][0]
            real_times = batch[EVT_DTTM_COLUMN][0][mask]
            if len(real_times) > 1:
                assert (real_times[1:] >= real_times[:-1]).all()


def test_dataset_per_modality_max_length_limits_events() -> None:
    """max_length=1 для mod_a оставляет ровно 1 событие с event_id=0 в каждом батче."""
    with TemporaryDirectory() as tmpdir:
        pq.write_table(
            pa.table({
                "row_id": pa.array([0], type=pa.int64()),
                "feat_a": _i64([list(range(10))]),
                "feat_b": _i64([list(range(5))]),
                "time_a": _f32([[float(i) for i in range(10)]]),
                "time_b": _f32([[float(i) * 0.5 for i in range(5)]]),
            }),
            f"{tmpdir}/data.parquet",
        )
        ds = _make_dataset(tmpdir, seq_len=10, max_a=1, max_b=5)
        batch = next(iter(ds))
        mask = batch[f"{EVENT_IDS_COLUMN}_mask"][0]
        eids = batch[EVENT_IDS_COLUMN][0][mask]
        assert (eids == 0).sum().item() == 1
        assert (eids == 1).sum().item() == 5


def test_dataset_zero_padding_in_output_batch() -> None:
    """В выходном батче признаки нулевые на позициях чужой модальности."""
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet", n_rows=3)
        ds = _make_dataset(tmpdir, batch_size=1)
        for batch in ds:
            mask = batch[f"{EVENT_IDS_COLUMN}_mask"][0]
            eids = batch[EVENT_IDS_COLUMN][0][mask]
            fa = batch["feat_a"][0][mask]
            fb = batch["feat_b"][0][mask]
            assert (fa[eids == 1] == 0).all(), "feat_a должен быть 0 на позициях mod_b"
            assert (fb[eids == 0] == 0).all(), "feat_b должен быть 0 на позициях mod_a"


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – длина датасета
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows,batch_size", [(1, 1), (3, 1), (4, 2), (6, 3)])
def test_dataset_len_matches_iteration_count(n_rows: int, batch_size: int) -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet", n_rows=n_rows)
        ds = _make_dataset(tmpdir, batch_size=batch_size)
        count = sum(1 for _ in ds)
        assert count == len(ds)


def test_dataset_len_raises_when_disabled() -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet")
        ds = _make_dataset(tmpdir)
        ds.do_compute_length = False
        with pytest.raises(TypeError):
            len(ds)


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – конфигурация modalities
# ---------------------------------------------------------------------------


def test_dataset_accepts_dict_modalities() -> None:
    """Датасет должен принимать обычные dict вместо ModalityConfig (совместимость с Hydra)."""
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet")
        metadata: Dict[str, Any] = {
            "row_id": {},
            "feat_a": {"sequential": True, "sequence_length": 16, "padding": 0},
            "feat_b": {"sequential": True, "sequence_length": 16, "padding": 0},
            EVT_DTTM_COLUMN: {"sequential": True, "sequence_length": 16, "padding": 0},
            EVENT_IDS_COLUMN: {"sequential": True, "sequence_length": 16, "padding": 0},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = MultimodalParquetDataset(
                source=tmpdir,
                metadata=metadata,
                modalities={
                    "mod_a": {"event_id": 0, "time_column": "time_a", "columns": ["feat_a"], "max_length": 8},
                    "mod_b": {"event_id": 1, "time_column": "time_b", "columns": ["feat_b"], "max_length": 8},
                },
                partition_size=32,
                batch_size=1,
            )
        batch = next(iter(ds))
        assert "feat_a" in batch


def test_dataset_single_modality() -> None:
    """Датасет с одной модальностью: event_ids полностью заполнен её event_id."""
    with TemporaryDirectory() as tmpdir:
        pq.write_table(
            pa.table({
                "feat_a": _i64([[1, 2, 3]]),
                "time_a": _f32([[1.0, 3.0, 2.0]]),
            }),
            f"{tmpdir}/data.parquet",
        )
        metadata: Dict[str, Any] = {
            "feat_a": {"sequential": True, "sequence_length": 8, "padding": 0},
            EVT_DTTM_COLUMN: {"sequential": True, "sequence_length": 8, "padding": 0},
            EVENT_IDS_COLUMN: {"sequential": True, "sequence_length": 8, "padding": 0},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = MultimodalParquetDataset(
                source=tmpdir,
                metadata=metadata,
                modalities={"only": ModalityConfig(event_id=7, time_column="time_a", columns=["feat_a"], max_length=8)},
                partition_size=32,
                batch_size=1,
            )
        batch = next(iter(ds))
        mask = batch[f"{EVENT_IDS_COLUMN}_mask"][0]
        assert (batch[EVENT_IDS_COLUMN][0][mask] == 7).all()


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – предупреждения о параметрах
# ---------------------------------------------------------------------------


def test_dataset_warns_partition_smaller_than_batch() -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet")
        metadata: Dict[str, Any] = {
            "row_id": {},
            "feat_a": {"sequential": True, "sequence_length": 16, "padding": 0},
            "feat_b": {"sequential": True, "sequence_length": 16, "padding": 0},
            EVT_DTTM_COLUMN: {"sequential": True, "sequence_length": 16, "padding": 0},
            EVENT_IDS_COLUMN: {"sequential": True, "sequence_length": 16, "padding": 0},
        }
        with pytest.warns(UserWarning, match=re.escape("partition size is smaller than batch size")):
            MultimodalParquetDataset(
                source=tmpdir,
                metadata=metadata,
                modalities=_two_mod_modalities(),
                partition_size=1,
                batch_size=4,
            )


def test_dataset_warns_partition_not_multiple_of_batch() -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet", n_rows=10)
        metadata: Dict[str, Any] = {
            "row_id": {},
            "feat_a": {"sequential": True, "sequence_length": 16, "padding": 0},
            "feat_b": {"sequential": True, "sequence_length": 16, "padding": 0},
            EVT_DTTM_COLUMN: {"sequential": True, "sequence_length": 16, "padding": 0},
            EVENT_IDS_COLUMN: {"sequential": True, "sequence_length": 16, "padding": 0},
        }
        with pytest.warns(UserWarning, match=re.escape("not multiple of batch size")):
            MultimodalParquetDataset(
                source=tmpdir,
                metadata=metadata,
                modalities=_two_mod_modalities(),
                partition_size=5,
                batch_size=3,
            )


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – random_slicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_dataset_random_slicing_reproducible(seed: int) -> None:
    """Два датасета с одинаковым seed должны выдавать идентичные батчи."""
    with TemporaryDirectory() as tmpdir:
        pq.write_table(
            pa.table({
                "row_id": pa.array([0], type=pa.int64()),
                "feat_a": _i64([list(range(30))]),
                "feat_b": _i64([list(range(30))]),
                "time_a": _f32([[float(i) for i in range(30)]]),
                "time_b": _f32([[float(i) * 0.7 for i in range(30)]]),
            }),
            f"{tmpdir}/data.parquet",
        )
        ds1 = _make_dataset(tmpdir, seq_len=20, max_a=6, max_b=6, random_slicing=True,
                            generator=torch.Generator().manual_seed(seed))
        ds2 = _make_dataset(tmpdir, seq_len=20, max_a=6, max_b=6, random_slicing=True,
                            generator=torch.Generator().manual_seed(seed))
        b1 = next(iter(ds1))
        b2 = next(iter(ds2))
        assert torch.equal(b1[EVT_DTTM_COLUMN], b2[EVT_DTTM_COLUMN])


def test_dataset_random_slicing_varies_across_seeds() -> None:
    """Разные seed должны давать разные последовательности (с высокой вероятностью)."""
    with TemporaryDirectory() as tmpdir:
        pq.write_table(
            pa.table({
                "row_id": pa.array([0], type=pa.int64()),
                "feat_a": _i64([list(range(50))]),
                "feat_b": _i64([list(range(50))]),
                "time_a": _f32([[float(i) for i in range(50)]]),
                "time_b": _f32([[float(i) * 0.3 for i in range(50)]]),
            }),
            f"{tmpdir}/data.parquet",
        )
        results = set()
        for seed in range(10):
            ds = _make_dataset(tmpdir, seq_len=20, max_a=5, max_b=5, random_slicing=True,
                               generator=torch.Generator().manual_seed(seed))
            b = next(iter(ds))
            results.add(tuple(b[EVT_DTTM_COLUMN][0].tolist()))
        assert len(results) > 1


# ---------------------------------------------------------------------------
# MultimodalParquetDataset – распределённое обучение
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_replicas", [1, 2])
def test_dataset_len_with_replicas_matches_iteration(num_replicas: int) -> None:
    with TemporaryDirectory() as tmpdir:
        _write_two_mod_parquet(f"{tmpdir}/data.parquet", n_rows=4)
        ds = _make_dataset(tmpdir, batch_size=1, replicas_info=FakeReplicasInfo(num_replicas=num_replicas))
        count = sum(1 for _ in ds)
        assert count == len(ds)
