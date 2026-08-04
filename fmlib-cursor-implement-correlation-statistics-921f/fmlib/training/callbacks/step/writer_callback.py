import warnings
from typing import Any, Dict, List, Self

import pyarrow.fs as fs
import torch

from fmlib.constants.callbacks import DEFAULT_TEMPLATE_EPOCH_DIR
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.constants.io import (
    DEFAULT_REPLICAS_INFO,
    DEFAULT_WRITE_EVERY_N_BATCH,
)
from fmlib.data.io.partitioning.replicas import ReplicasInfoProtocol
from fmlib.data.io.writer import PartitionedParquetWriter
from fmlib.data.utils.prepare_directory import prepare_directory
from fmlib.training.training_types import MetricsType, TrainingState


class WriterCallback:
    """
    Коллбек, позволяющий сохранять результаты работы модели в формате Parquet.
        Переиспользует сущность `fmlib.data.io.writer.PartitionedParquetWriter`.

    Аргументы:
        base_path (str): Папка, в которой будет сохраняться результат.
        output_to_save (List[str]): Список выходов модели, которые будут сохраняться.
        transformed_to_save (List[str]): Список элементов батча, которые будут сохраняться.
        filesystem (fs.FileSystem): Файловая система для сохранения. По умолчанию `DEFAULT_FILESYSTEM`.
        replicas_info (ReplicasInfoProtocol): Информация о репликах. По умолчанию `DEFAULT_REPLICAS_INFO`.
            Используется для того, чтобы не было конфликтов при сохранении.
        write_every_n_batch (int): Частота записи. По умолчанию `DEFAULT_WRITE_EVERY_N_BATCH`.
        writer_kwargs (Dict[str, Any] | None): Дополнительные аргументы для передачи в `ParquetWriter`.
            По умолчанию `None`.

    Аттрибуты:
        template_epoch_dir (str): Шаблон имени директории для сохранения результатов в каждой эпохе.
            По умолчанию `DEFAULT_TEMPLATE_EPOCH_DIR`.
        writer_epoch (int | None): Номер текущей эпохи, для которой открыт `ParquetWriter`.
        writer (PartitionedParquetWriter | None): Объект отвечающий за запись на текущей эпохе.
    """

    def __init__(
        self: Self,
        base_path: str,
        output_to_save: List[str],
        transformed_to_save: List[str],
        filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
        write_every_n_batch: int = DEFAULT_WRITE_EVERY_N_BATCH,
        writer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self.output_to_save: List[str] = sorted(output_to_save)
        self.transformed_to_save: List[str] = sorted(transformed_to_save)

        self.base_path: str = prepare_directory(
            strict=True,
            exists_ok=True,
            filesystem=filesystem,
            path=base_path,
        )

        self.filesystem: fs.FileSystem = filesystem
        self.write_every_n_batch: int = write_every_n_batch
        self.replicas_info: ReplicasInfoProtocol = replicas_info
        self.writer_kwargs: Dict[str, Any] = writer_kwargs or {}

        self.writer_epoch: int | None = None
        self.writer: PartitionedParquetWriter | None = None

        self.template_epoch_dir: str = DEFAULT_TEMPLATE_EPOCH_DIR

        self.reset()

    def get_epoch_path(self: Self, epoch: int) -> str:
        epoch_dir: str = self.template_epoch_dir.format(epoch=epoch)
        return f"{self.base_path}/{epoch_dir}"

    def get_or_make_writer(self: Self, epoch: int) -> PartitionedParquetWriter:
        if self.writer_epoch is not None and (epoch != self.writer_epoch):
            msg: str = f"Epoch changed: {epoch=} vs. {self.writer_epoch=}. Looks like callback is not resetted."
            warnings.warn(msg, stacklevel=2)
            self.reset()

        if self.writer is None:
            self.writer = PartitionedParquetWriter(
                base_path=self.get_epoch_path(epoch),
                filesystem=self.filesystem,
                replicas_info=self.replicas_info,
                write_every_n_batch=self.write_every_n_batch,
                writer_kwargs=self.writer_kwargs,
            )
        assert self.writer is not None
        return self.writer

    @property
    def output_names(self: Self) -> List[str]:
        """
        Подписка на выходы модели.
        """
        return self.output_to_save

    @property
    def transformed_names(self: Self) -> List[str]:
        """
        Подписка на таргеты.
        """
        return self.transformed_to_save

    def reset(self: Self) -> Self:
        """
        Очистка состояния коллбека.
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        assert self.writer is None
        self.writer_epoch = None
        return self

    def __call__(
        self: Self,
        state: TrainingState,
        transformed: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        epoch: int = 0,
        external: MetricsType | None = None,
    ) -> MetricsType:
        """
        Сохраняет во внутреннее состояние результаты работы модели и таргеты.
        """
        if external is None:
            external = {}

        def filter_batch(batch: Dict[str, torch.Tensor], columns: List[str]) -> Dict[str, torch.Tensor]:
            return {name: batch[name] for name in columns}

        data: Dict[str, torch.Tensor] = {
            **filter_batch(outputs, self.output_to_save),
            **filter_batch(transformed, self.transformed_to_save),
        }

        writer: PartitionedParquetWriter = self.get_or_make_writer(epoch)

        writer.write(data=data)

        return {}

    def finalize(self: Self, state: TrainingState, epoch: int = 0) -> MetricsType:
        self.reset()
        return {}
