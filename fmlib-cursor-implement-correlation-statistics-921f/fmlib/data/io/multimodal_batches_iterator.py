from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Self, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch

from fmlib.constants.device import DEFAULT_DEVICE
from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME
from fmlib.constants.multimodal import EVT_DTTM_COLUMN, EVENT_IDS_COLUMN
from fmlib.data.io.implementation.flat_column import to_flat_columns
from fmlib.data.io.implementation.named_columns import NamedColumns
from fmlib.data.io.implementation.sequence_column import to_sequence_columns
from fmlib.data.io.metadata import Metadata, list_not_sequential
from fmlib.data.io.pyarrow_partition_generator import shuffle_dataset


@dataclass
class ModalityConfig:
    """Конфигурация одной модальности событий.

    Атрибуты:
        event_id: Целочисленный идентификатор, которым заполняется колонка event_ids
            на позициях, принадлежащих данной модальности.
        time_column: Имя колонки в Parquet-файле с временными метками данной модальности.
        columns: Имена признаковых колонок, принадлежащих данной модальности.
        max_length: Максимальное количество событий, сохраняемых из данной модальности
            перед слиянием.
    """

    event_id: int
    time_column: str
    columns: List[str]
    max_length: int

    def __post_init__(self) -> None:
        self.columns = list(self.columns)


def _merge_multimodal_batch(
    batch: pa.RecordBatch,
    modalities: Dict[str, ModalityConfig],
    flat_columns: List[str],
    random_slicing: bool,
    generator: Optional[torch.Generator],
) -> pa.Table:
    """Объединяет последовательности разных модальностей в одну хронологическую.

    Для каждой строки батча:
    1. Обрезает каждую модальность до max_length (с хвоста или случайно при random_slicing=True).
    2. Конкатенирует временные метки всех модальностей и сортирует их (stable argsort).
    3. Дополняет каждый признаковый массив нулями на позициях других модальностей
       и переупорядочивает по argsort.
    4. Формирует колонки evt_dttm (отсортированные метки времени) и event_ids
       (целочисленные идентификаторы модальности).

    Плоские (не последовательные) колонки передаются без изменений.

    Аргументы:
        batch: Батч PyArrow с сырыми колонками из Parquet-файла.
        modalities: Словарь конфигураций модальностей.
        flat_columns: Имена плоских колонок для сквозной передачи.
        random_slicing: Если True — случайный срез; иначе берётся хвост последовательности.
        generator: Генератор случайных чисел (используется при random_slicing=True).

    Возвращает:
        pa.Table с объединёнными колонками, совместимый с output-метаданными.
    """
    n_rows = len(batch)
    modality_names: List[str] = list(modalities.keys())
    all_feature_cols: List[str] = [col for cfg in modalities.values() for col in cfg.columns]
    col_to_mod: Dict[str, Tuple[int, str]] = {
        col: (i, mod_name)
        for i, (mod_name, cfg) in enumerate(modalities.items())
        for col in cfg.columns
    }

    # ── 1. Извлекаем плоские NumPy-массивы одним PyArrow-вызовом на колонку ──────────────
    # Заменяет n_rows × .as_py() на один pc.list_flatten + .to_numpy() на всю колонку.

    mod_info: Dict[str, dict] = {}
    for mod_name, cfg in modalities.items():
        time_col = batch.column(cfg.time_column)

        # Длина каждой строки (null/пустой список → 0)
        lengths_np: np.ndarray = (
            pc.fill_null(pc.list_value_length(time_col), 0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )

        # Кумулятивные смещения в плоский массив
        offsets = np.empty(n_rows + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths_np, out=offsets[1:])

        # Плоский массив временных меток
        flat_times: np.ndarray = pc.list_flatten(time_col).to_numpy(zero_copy_only=False).astype(np.float64)

        # Вычисляем начало и длину среза для каждой строки (векторно)
        max_len = cfg.max_length
        trunc_lengths = np.minimum(lengths_np, max_len)

        if random_slicing:
            needs_trunc = lengths_np > max_len
            n_trunc = int(needs_trunc.sum())
            shifts = np.zeros(n_rows, dtype=np.int64)
            if n_trunc > 0:
                max_shifts = lengths_np[needs_trunc] - max_len
                rand_vals = (
                    torch.rand(n_trunc, generator=generator).numpy()
                    if generator is not None
                    else np.random.rand(n_trunc)
                )
                raw = (rand_vals * (max_shifts + 1)).astype(np.int64)
                shifts[needs_trunc] = np.minimum(raw, max_shifts)
            trunc_starts = offsets[:-1] + shifts
        else:
            # Tail-срез: берём последние max_len элементов
            trunc_starts = np.maximum(offsets[:-1], offsets[1:] - max_len)

        # Плоские массивы признаков (те же смещения, что у времени)
        flat_feats: Dict[str, np.ndarray] = {}
        elem_types: Dict[str, pa.DataType] = {}
        for col in cfg.columns:
            feat_col = batch.column(col)
            t = feat_col.type
            elem_types[col] = t.value_type if (pa.types.is_list(t) or pa.types.is_large_list(t)) else t
            flat_feats[col] = pc.list_flatten(feat_col).to_numpy(zero_copy_only=False)

        mod_info[mod_name] = {
            "trunc_starts": trunc_starts,
            "trunc_lengths": trunc_lengths,
            "flat_times": flat_times,
            "flat_feats": flat_feats,
            "elem_types": elem_types,
        }

    # ── 2. Предварительно выделяем выходные плоские массивы ─────────────────────────────

    total_lengths = np.zeros(n_rows, dtype=np.int64)
    for mn in modality_names:
        total_lengths += mod_info[mn]["trunc_lengths"]

    total_flat = int(total_lengths.sum())

    out_offsets = np.empty(n_rows + 1, dtype=np.int64)
    out_offsets[0] = 0
    np.cumsum(total_lengths, out=out_offsets[1:])

    out_evt_dttm = np.empty(total_flat, dtype=np.float32)
    out_event_ids = np.empty(total_flat, dtype=np.int64)
    # Нули — паддинг для позиций чужих модальностей (инициализируется один раз)
    out_feats: Dict[str, np.ndarray] = {
        col: np.zeros(total_flat, dtype=mod_info[col_to_mod[col][1]]["flat_feats"][col].dtype)
        for col in all_feature_cols
    }

    mod_event_ids = np.array([modalities[mn].event_id for mn in modality_names], dtype=np.int64)

    # ── 3. Построчный merge-and-sort (NumPy-операции над уже извлечёнными срезами) ───────
    # Цикл сохраняется (переменная длина строк), но внутри — только NumPy, без .as_py().

    for row_i in range(n_rows):
        total_len = int(total_lengths[row_i])
        if total_len == 0:
            continue
        out_s = int(out_offsets[row_i])

        starts = [int(mod_info[mn]["trunc_starts"][row_i]) for mn in modality_names]
        sizes = [int(mod_info[mn]["trunc_lengths"][row_i]) for mn in modality_names]

        # Конкатенируем временные метки и сортируем
        all_times = np.concatenate(
            [mod_info[mn]["flat_times"][s: s + sz] for mn, s, sz in zip(modality_names, starts, sizes)]
        )
        argsort = np.argsort(all_times, kind="stable")

        out_evt_dttm[out_s: out_s + total_len] = all_times[argsort].astype(np.float32)

        # event_ids: повторяем id каждой модальности по её размеру, переупорядочиваем
        sizes_np = np.array(sizes, dtype=np.int64)
        out_event_ids[out_s: out_s + total_len] = np.repeat(mod_event_ids, sizes_np)[argsort]

        # Префиксные суммы размеров модальностей в pre-argsort порядке
        pref = np.empty(len(modality_names) + 1, dtype=np.int64)
        pref[0] = 0
        np.cumsum(sizes_np, out=pref[1:])

        # Признаки: scatter без промежуточного padded-буфера.
        # Для каждой колонки находим выходные позиции, принадлежащие её модальности,
        # и записываем только их — остальные позиции уже равны 0 (pre-zero).
        for col in all_feature_cols:
            mod_idx, mod_name = col_to_mod[col]
            sz = sizes[mod_idx]
            if sz == 0:
                continue
            lp = int(pref[mod_idx])
            feat = mod_info[mod_name]["flat_feats"][col][starts[mod_idx]: starts[mod_idx] + sz]
            # Маска: argsort[k] ∈ [lp, lp+sz) → выходная позиция k принадлежит этой модальности
            mask = (argsort >= lp) & (argsort < lp + sz)
            out_feats[col][out_s: out_s + total_len][mask] = feat[argsort[mask] - lp]

    # ── 4. Собираем выходную PyArrow-таблицу ────────────────────────────────────────────

    # int32 достаточно для типичных partition_size (< 2 млрд элементов)
    offsets_pa = pa.array(out_offsets.astype(np.int32))

    out_arrays: Dict[str, pa.Array] = {}
    for col in flat_columns:
        out_arrays[col] = batch.column(col)
    out_arrays[EVT_DTTM_COLUMN] = pa.ListArray.from_arrays(
        offsets_pa, pa.array(out_evt_dttm, type=pa.float32())
    )
    out_arrays[EVENT_IDS_COLUMN] = pa.ListArray.from_arrays(
        offsets_pa, pa.array(out_event_ids, type=pa.int64())
    )
    for col in all_feature_cols:
        _, mod_name = col_to_mod[col]
        elem_type = mod_info[mod_name]["elem_types"][col]
        out_arrays[col] = pa.ListArray.from_arrays(
            offsets_pa, pa.array(out_feats[col], type=elem_type)
        )

    return pa.table(out_arrays)


class MultimodalBatchesIterator:
    """Итератор, читающий мультимодальные Parquet-батчи и объединяющий последовательности хронологически.

    Каждая модальность имеет собственные признаковые колонки (с потенциально разной длиной
    в каждой строке) и отдельную колонку временных меток. События всех модальностей
    объединяются в порядке возрастания времени; на позициях, принадлежащих другим
    модальностям, признаки заполняются нулями.

    Выходной NamedColumns содержит:
    - Все плоские (не последовательные) колонки из Parquet-файла без изменений.
    - Признаковые колонки (list-массивы) с нулевым паддингом на позициях других модальностей.
    - evt_dttm: объединённые отсортированные временные метки.
    - event_ids: целочисленный идентификатор модальности для каждого временного шага.

    Аргументы:
        dataset: PyArrow-датасет для чтения данных.
        metadata: Метаданные выходных колонок.
        modalities: Конфигурации модальностей.
        batch_size: Количество строк в одной партиции при чтении.
        make_mask_name: Функция для формирования имён маскирующих колонок.
        device: Устройство для хранения тензоров.
        pyarrow_kwargs: Дополнительные аргументы для метода to_batches.
        generator: Генератор случайных чисел (для перемешивания и random_slicing).
        random_slicing: Если True — случайный срез длинных последовательностей.
    """

    def __init__(
        self: Self,
        dataset: ds.Dataset,
        metadata: Metadata,
        modalities: Dict[str, ModalityConfig],
        batch_size: int,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        pyarrow_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
        random_slicing: bool = False,
    ) -> None:
        if pyarrow_kwargs is None:
            pyarrow_kwargs = {}
        self.dataset: ds.Dataset = dataset
        self.metadata: Metadata = metadata
        self.modalities: Dict[str, ModalityConfig] = modalities
        self.batch_size: int = batch_size
        self.make_mask_name: Callable[[str], str] = make_mask_name
        self.device: torch.device = device
        self.pyarrow_kwargs: Dict[str, Any] = pyarrow_kwargs
        self.generator: Optional[torch.Generator] = generator
        self.random_slicing: bool = random_slicing

        # Колонки для чтения из Parquet: все выходные колонки минус производные + колонки времён
        _derived = {EVT_DTTM_COLUMN, EVENT_IDS_COLUMN}
        _time_cols = {cfg.time_column for cfg in modalities.values()}
        self._read_columns: List[str] = sorted((set(metadata.keys()) - _derived) | _time_cols)
        self._flat_columns: List[str] = list_not_sequential(metadata)

    def __iter__(self: Self) -> Iterator[NamedColumns]:
        dataset: ds.Dataset = shuffle_dataset(self)
        for batch in dataset.to_batches(
            batch_size=self.batch_size,
            columns=self._read_columns,
            **self.pyarrow_kwargs,
        ):
            merged = _merge_multimodal_batch(
                batch=batch,
                modalities=self.modalities,
                flat_columns=self._flat_columns,
                random_slicing=self.random_slicing,
                generator=self.generator,
            )
            yield NamedColumns(
                columns={
                    **to_flat_columns(merged, self.metadata, self.device),
                    **to_sequence_columns(merged, self.metadata, self.device, random_slicing=False),
                },
                make_mask_name=self.make_mask_name,
            )
