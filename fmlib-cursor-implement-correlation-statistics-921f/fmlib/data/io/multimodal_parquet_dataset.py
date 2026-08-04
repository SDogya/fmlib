import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Self, Union, cast

import pyarrow.dataset as ds
import pyarrow.fs as fs
import torch
from torch.utils.data import IterableDataset

from fmlib.constants.batches import GeneralBatch, GeneralCollateFn
from fmlib.constants.device import DEFAULT_DEVICE
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.constants.io import DEFAULT_COLLATE_FN, DEFAULT_MAKE_MASK_NAME, DEFAULT_REPLICAS_INFO
from fmlib.data.io.metadata import Metadata
from fmlib.data.io.partitioning.replicas import ReplicasInfoProtocol
from fmlib.data.io.utils.compute_length import compute_fixed_size_length

from .fixed_batch_dataset import FixedBatchSizeDataset
from .multimodal_batches_iterator import ModalityConfig, MultimodalBatchesIterator
from .partitioned_iterable_dataset import PartitionedIterableDataset


class MultimodalParquetDataset(IterableDataset):
    """Датасет для загрузки мультимодальных последовательностей событий из Parquet-файлов.

    Каждая модальность имеет собственные признаковые колонки и колонку временных меток,
    причём длины последовательностей для разных модальностей могут отличаться.
    При загрузке события всех модальностей объединяются в хронологическом порядке:
    - Признаковые колонки дополняются нулями на позициях, принадлежащих другим модальностям.
    - Формируется колонка evt_dttm с объединёнными отсортированными временными метками.
    - Формируется колонка event_ids с целочисленными идентификаторами модальности.

    Аргументы:
        source: Путь или список путей к Parquet-файлам/директориям.
        metadata: Схема выходных колонок. Должна включать evt_dttm и event_ids
            как последовательные колонки.
        modalities: Словарь имя → ModalityConfig. Принимает и обычные dict
            (например, из Hydra DictConfig).
        partition_size: Количество записей в одной партиции при чтении из Parquet.
        batch_size: Количество записей в одном выходном батче.
        filesystem: Файловая система PyArrow. По умолчанию DEFAULT_FILESYSTEM.
        make_mask_name: Функция для формирования имён маскирующих колонок.
            По умолчанию DEFAULT_MAKE_MASK_NAME.
        device: Устройство для выходных тензоров. По умолчанию DEFAULT_DEVICE.
        generator: Генератор случайных чисел для перемешивания и random_slicing.
        replicas_info: Информация о репликах для распределённого обучения.
            По умолчанию DEFAULT_REPLICAS_INFO.
        collate_fn: Функция для соединения батчей. По умолчанию DEFAULT_COLLATE_FN.
        random_slicing: Если True — случайный срез длинных последовательностей
            вместо взятия хвоста.

    Атрибуты:
        filesystem: Файловая система PyArrow.
        modalities: Словарь конфигураций модальностей.
        pyarrow_dataset: Базовый PyArrow-датасет.
        batch_size: Размер выходного батча.
        partition_size: Размер партиции при чтении.
        replicas_info: Информация о репликах.
        metadata: Схема выходных колонок.
        iterator: Итератор MultimodalBatchesIterator.
        raw_dataset: PartitionedIterableDataset поверх итератора.
        dataset: FixedBatchSizeDataset с фиксированным размером батча.
        do_compute_length: Флаг вычисления длины датасета. По умолчанию True.
    """

    def __init__(
        self: Self,
        source: Union[str, List[str]],
        metadata: Metadata,
        modalities: Dict[str, Union[ModalityConfig, Dict[str, Any]]],
        partition_size: int,
        batch_size: int,
        filesystem: Union[str, fs.FileSystem] = DEFAULT_FILESYSTEM,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        generator: Optional[torch.Generator] = None,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
        collate_fn: GeneralCollateFn = DEFAULT_COLLATE_FN,
        random_slicing: bool = False,
        **kwargs,
    ) -> None:
        if partition_size < batch_size:
            warnings.warn(
                f"Suboptimal parameters: partition size is smaller than batch size. Got: {partition_size=}, {batch_size=}.",
                stacklevel=2,
            )
        if (partition_size % batch_size) != 0:
            warnings.warn(
                f"Suboptimal parameters: partition size is not multiple of batch size. Got: {partition_size=}, {batch_size=}.",
                stacklevel=2,
            )

        if isinstance(filesystem, str):
            filesystem = fs.FileSystem.from_uri(filesystem)
        assert isinstance(filesystem, fs.FileSystem)
        self.filesystem: fs.FileSystem = cast(fs.FileSystem, filesystem)

        # Принимаем обычные dict (например, из Hydra DictConfig) и конвертируем в ModalityConfig
        self.modalities: Dict[str, ModalityConfig] = {
            name: ModalityConfig(**dict(cfg)) if not isinstance(cfg, ModalityConfig) else cfg
            for name, cfg in modalities.items()
        }

        self.pyarrow_dataset: ds.Dataset = ds.dataset(
            source,
            filesystem=self.filesystem,
            format="parquet",
            **kwargs.get("pyarrow_dataset_kwargs", {}),
        )

        self.batch_size: int = batch_size
        self.partition_size: int = partition_size
        self.replicas_info: ReplicasInfoProtocol = replicas_info
        self.metadata: Metadata = metadata

        self.iterator: MultimodalBatchesIterator = MultimodalBatchesIterator(
            dataset=self.pyarrow_dataset,
            metadata=metadata,
            modalities=self.modalities,
            batch_size=partition_size,
            make_mask_name=make_mask_name,
            device=device,
            generator=generator,
            pyarrow_kwargs=kwargs.get("pyarrow_to_batches_kwargs", {}),
            random_slicing=random_slicing,
        )

        self.raw_dataset: PartitionedIterableDataset = PartitionedIterableDataset(
            batch_size=batch_size,
            iterable=self.iterator,
            generator=generator,
            replicas_info=replicas_info,
        )

        self.dataset: FixedBatchSizeDataset = FixedBatchSizeDataset(
            dataset=self.raw_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        self.do_compute_length: bool = True
        self.cached_lengths: dict[int, int] = {}

    def compute_length(self: Self) -> int:
        """Возвращает длину датасета в батчах фиксированного размера."""
        num_replicas: int = self.replicas_info.num_replicas
        if num_replicas not in self.cached_lengths:
            if len(self.cached_lengths) > 0:
                warnings.warn("`num_replicas` changed. Unable to reuse cached length.", stacklevel=2)
            curr_length: int = compute_fixed_size_length(
                iterable=self.iterator,
                num_replicas=num_replicas,
                batch_size=self.batch_size,
            )
            self.cached_lengths[num_replicas] = curr_length
        return self.cached_lengths[num_replicas]

    def __len__(self: Self) -> int:
        if self.do_compute_length:
            return self.compute_length()
        raise TypeError(
            "Данный экземпляр не поддерживает `len()`. "
            "Установите `do_compute_length=True` для активации."
        )

    def __iter__(self: Self) -> Iterator[GeneralBatch]:
        return iter(self.dataset)
