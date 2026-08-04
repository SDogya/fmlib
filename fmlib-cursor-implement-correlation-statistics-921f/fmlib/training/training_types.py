from typing import ContextManager, Iterable, Protocol, Self, TypedDict

import accelerate
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset

from fmlib.constants.batches import Batch

MetricsValue = str | int | float
MetricsType = dict[str, MetricsValue]
DataType = DataLoader | IterableDataset


class EpochCallbackProtocol(Protocol):
    """
    Протокол для коллбека по эпохам.
    """

    def __call__(self: Self, state: "TrainingState", metrics: MetricsType, epoch: int = 0) -> MetricsType: ...

    def finalize(self: Self, state: "TrainingState") -> MetricsType: ...

    def reset(self: Self) -> Self: ...


class StepCallbackProtocol(Protocol):
    """
    Протокол для коллбека по шагам.
    """

    def __call__(
        self: Self,
        state: "TrainingState",
        transformed: Batch,
        outputs: Batch,
        epoch: int = 0,
        external: MetricsType | None = None,
    ) -> MetricsType: ...

    def finalize(self: Self, state: "TrainingState", epoch: int = 0) -> MetricsType: ...

    def reset(self: Self) -> Self: ...


class TrainingState(TypedDict):
    """
    Структура для хранения состояния обучения.

    Поля:
        accelerator (accelerate.Accelerator): Акселератор для обучения.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Планировщик Learning Rate.
        optimizer (torch.optim.Optimizer): Оптимизатор для обучения.
        pipeline (torch.nn.Module): Модель для обучения, трансформы и лоссы в одном объекте.
        training_data (IterableDataset): Данные для тренировки. Должен быть итерируемый объект,
            возвращающий батчи, например DataLoader или ParquetDataset.
        validation_data (IterableDataset): Данные для валидации. Должен быть итерируемый объект,
            возвращающий батчи, например DataLoader или ParquetDataset.
        training_callbacks (StepCallbackProtocol): Коллбеки для тренировочного шага.
            Будут вызываться после каждого батча.
        validation_callbacks (StepCallbackProtocol): Коллбеки для валидационного шага.
            Будут вызываться после каждого батча.
        epoch_callbacks (EpochCallbackProtocol): Коллбеки для эпох.
        epoch_iterable (Iterable[int]): Итератор по эпохам. Определяет сколько эпох обучения будет.
        config (DictConfig): Конфигурация обучения.
        autocast (ContextManager): Автокастинг для обучения.
    """

    accelerator: accelerate.Accelerator
    scheduler: torch.optim.lr_scheduler.LRScheduler
    optimizer: torch.optim.Optimizer
    pipeline: torch.nn.Module
    training_data: DataType
    validation_data: DataType
    training_step_callbacks: StepCallbackProtocol
    validation_step_callbacks: StepCallbackProtocol
    epoch_callbacks: EpochCallbackProtocol
    epoch_iterable: Iterable[int]
    config: DictConfig
    autocast: ContextManager


class EvaluationState(TypedDict):
    """
    Структура для хранения состояния обучения.

    Поля:
        accelerator (accelerate.Accelerator): Акселератор для обучения.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Планировщик Learning Rate.
        optimizer (torch.optim.Optimizer): Оптимизатор для обучения.
        pipeline (torch.nn.Module): Модель для обучения, трансформы и лоссы в одном объекте.
        training_data (IterableDataset): Данные для тренировки. Должен быть итерируемый объект,
            возвращающий батчи, например DataLoader или ParquetDataset.
        validation_data (IterableDataset): Данные для валидации. Должен быть итерируемый объект,
            возвращающий батчи, например DataLoader или ParquetDataset.
        training_callbacks (StepCallbackProtocol): Коллбеки для тренировочного шага.
            Будут вызываться после каждого батча.
        validation_callbacks (StepCallbackProtocol): Коллбеки для валидационного шага.
            Будут вызываться после каждого батча.
        epoch_callbacks (EpochCallbackProtocol): Коллбеки для эпох.
        epoch_iterable (Iterable[int]): Итератор по эпохам. Определяет сколько эпох обучения будет.
        config (DictConfig): Конфигурация обучения.
        autocast (ContextManager): Автокастинг для обучения.
    """

    accelerator: accelerate.Accelerator
    pipeline: torch.nn.Module
    evaluation_data: DataType
    evaluation_step_callbacks: StepCallbackProtocol
    autocast: ContextManager
