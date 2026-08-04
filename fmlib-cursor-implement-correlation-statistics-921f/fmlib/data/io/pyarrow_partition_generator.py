from typing import Any, Callable, Dict, Iterator, Protocol, Self

import pyarrow as pa
import pyarrow.dataset as ds
import torch

from fmlib.constants.device import DEFAULT_DEVICE
from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME

from .implementation.flat_column import to_flat_columns
from .implementation.named_columns import NamedColumns
from .implementation.sequence_column import to_sequence_columns
from .metadata import Metadata


class ShufflableDataset(Protocol):
    """Протокол датасета, подлежащего перемешиванию"""

    generator: torch.Generator | None
    dataset: ds.Dataset


def shuffle_dataset_fragments(dataset: ds.Dataset, generator: torch.Generator | None = None) -> ds.Dataset:
    """Перемешивает датасет по фрагментам, используя генератор."""
    fragments: list[str] = sorted([f.path for f in dataset.get_fragments()])
    perm: torch.Tensor = torch.randperm(len(fragments), generator=generator)
    result_fragments = [fragments[i] for i in perm.cpu().tolist()]
    new_dataset: ds.Dataset = ds.dataset(
        filesystem=dataset.filesystem,
        source=result_fragments,
        format=dataset.format,
    )
    return new_dataset


def shuffle_dataset(shufflable: ShufflableDataset) -> ds.Dataset:
    """Опционально перемешивает датасет, если определен генератор."""
    if shufflable.generator is None:
        dataset: ds.Dataset = shufflable.dataset
    else:
        dataset: ds.Dataset = shuffle_dataset_fragments(
            generator=shufflable.generator,
            dataset=shufflable.dataset,
        )
    return dataset


class FragmentIterator:
    """
    Итератор для извлечения данных из датасета по фрагментам и преобразования их в структурированные колонки.
    Фрагмент - это один Parquet-файл в файловой системе.
    Фрагменты могут содержать разное количество строк, это зависит от настроенного партиционирования при записи датасета.

    Аргументы:
        metadata (Metadata): Метаданные, описывающие структуру данных.
        dataset (ds.Dataset): PyArrow-датасет, поддерживающий метод `get_fragments`.
        make_mask_name (Callable[[str], str], optional): Функция для генерации имени маски на основе имени колонки.
            По умолчанию используется `DEFAULT_MAKE_MASK_NAME`.
        device (torch.device, optional): Устройство, на котором будут создаваться тензоры (CPU или GPU).
            По умолчанию — `DEFAULT_DEVICE`.
        pyarrow_kwargs (Dict[str, Any], optional): Дополнительные аргументы, передаваемые в метод `get_fragments`.

    Атрибуты:
        metadata (Metadata): Метаданные, переданные при инициализации.
        dataset (ds.Dataset): Входной датасет.
        make_mask_name (Callable[[str], str]): Функция формирования имени маски.
        device (torch.device): Устройство, на котором создаются тензоры.
        pyarrow_kwargs (Dict[str, Any]): Параметры для метода `get_fragments`.
    """

    def __init__(
        self: Self,
        metadata: Metadata,
        dataset: ds.Dataset,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        pyarrow_kwargs: Dict[str, Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if pyarrow_kwargs is None:
            pyarrow_kwargs = {}
        self.metadata: Metadata = metadata
        self.dataset: ds.Dataset = dataset
        self.make_mask_name: Callable[[str], str] = make_mask_name
        self.device: torch.device = device
        self.pyarrow_kwargs: Dict[str, Any] = pyarrow_kwargs
        self.generator: torch.Generator | None = generator
        self.columns: list[str] = sorted(metadata.keys())

    def __iter__(self: Self) -> Iterator[NamedColumns]:
        fragment: ds.Fragment
        dataset: ds.Dataset = shuffle_dataset(self)
        for fragment in dataset.get_fragments(
            columns=self.columns,
            **self.pyarrow_kwargs,
        ):
            table: pa.Table = fragment.to_table()
            yield NamedColumns(
                columns={
                    **to_flat_columns(table, self.metadata, self.device),
                    **to_sequence_columns(table, self.metadata, self.device),
                },
                make_mask_name=self.make_mask_name,
            )


class BatchesIterator:
    """
    Итератор для побатчевого извлечения данных из parquet-датасета с преобразованием в структурированные колонки.

    Аргументы:
        metadata (Metadata): Метаданные, описывающие структуру и типы данных.
        dataset (ds.Dataset): Pyarrow-датасет, поддерживающий метод to_batches.
        batch_size (int): Размер батча при обработке одной партиции parquet-датасета.
            Размер получаемого батча не всегда будет равен batch_size.
            По причине того, что в фрагменте parquet-файла может содержаться количество строк некратное batch_size.
            Например, если фрагмент содержит 1000 строк и batch_size равен 64,
            то вы получите 15 батчей размера 64 и последний батч будет размером 40.
        make_mask_name (Callable[[str], str], optional): Функция для генерации имени маски.
            По умолчанию `DEFAULT_MAKE_MASK_NAME`.
        device (torch.device, optional): Устройство для хранения тензоров (CPU/GPU). По умолчанию `DEFAULT_DEVICE`.
        pyarrow_kwargs (Dict[str, Any], optional): Дополнительные аргументы для метода `to_batches`.
            Для большего понимания смотрите документацию метода `to_batches` в PyArrow Dataset.

    Атрибуты:
        dataset (ds.Dataset): Входной датасет.
        metadata (Metadata): Метаданные.
        batch_size (int): Размер батча.
        make_mask_name (Callable[[str], str]): Функция формирования имени маски.
        device (torch.device): Устройство, на котором будут формироваться данные.
        pyarrow_kwargs (Dict[str, Any]): Параметры для PyArrow.
    """

    def __init__(
        self: Self,
        metadata: Metadata,
        dataset: ds.Dataset,
        batch_size: int,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        pyarrow_kwargs: Dict[str, Any] | None = None,
        generator: torch.Generator | None = None,
        random_slicing: bool = False
    ) -> None:
        if pyarrow_kwargs is None:
            pyarrow_kwargs = {}
        self.dataset: ds.Dataset = dataset
        self.metadata: Metadata = metadata
        self.batch_size: int = batch_size
        self.make_mask_name: Callable[[str], str] = make_mask_name
        self.device: torch.device = device
        self.pyarrow_kwargs: Dict[str, Any] = pyarrow_kwargs
        self.generator: torch.Generator | None = generator
        self.columns: list[str] = sorted(metadata.keys())
        self.random_slicing: bool = random_slicing

    def __iter__(self: Self) -> Iterator[NamedColumns]:
        batch: pa.RecordBatch
        dataset: ds.Dataset = shuffle_dataset(self)
        for batch in dataset.to_batches(
            batch_size=self.batch_size,
            columns=self.columns,
            **self.pyarrow_kwargs,
        ):
            yield NamedColumns(
                columns={
                    **to_flat_columns(batch, self.metadata, self.device),
                    **to_sequence_columns(batch, self.metadata, self.device, self.random_slicing),
                },
                make_mask_name=self.make_mask_name,
            )
