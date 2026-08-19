from typing import List, Protocol, Self

from fmlib.training.training_types import MetricsType, TrainingState


class EpochCallbackProtocol(Protocol):
    """
    Общий протокол коллбека, вызываемого в конце каждой эпохи и обучения.
    """

    def reset(self: Self) -> Self: ...

    def finalize(self: Self, state: TrainingState) -> MetricsType: ...

    def __call__(self: Self, state: TrainingState, metrics: MetricsType, epoch: int = 0) -> MetricsType: ...


class GroupedEpochCallback:
    """
    Общий коллбек, вызываемый в конце каждой эпохи и обучения.
    Объединяет несколько отдельных коллбеков, объединённых `EpochCallbackProtocol`.
    """

    def __init__(self: Self, callbacks: List[EpochCallbackProtocol]) -> None:
        self.callbacks: List[EpochCallbackProtocol] = callbacks
        self.state: MetricsType = {}

    def __call__(self: Self, state: TrainingState, metrics: MetricsType | None = None, epoch: int = 0) -> MetricsType:
        """
        Основной метод вызова коллбеков. Делегирует другим коллбекам.

        Аргументы:
            state (TrainingState): Состояние обучения.
            metrics (MetricsType | None): Словарь внешних метрик. Опционально.
            epoch (int): Номер эпохи. Опционально.

        Возвращает:
            MetricsType: Словарь с результатами.
        """
        if metrics is None:
            metrics = {}
        results: MetricsType = {"epoch": epoch, **metrics}
        for callback in self.callbacks:
            result = callback(state, metrics, epoch=epoch)
            results.update(result)
        self.state = results
        return self.state

    def reset(self: Self) -> Self:
        """
        Сбрасывает состояние коллбеков.

        Возвращает:
            Self: Объект коллбека.
        """
        for callback in self.callbacks:
            callback.reset()
        self.state = {}
        return self

    def finalize(self: Self, state: TrainingState) -> MetricsType:
        """
        Собирает результаты, сбрасывает состояния коллбеков и возвращает.

        Аргументы:
            state (TrainingState): Состояние обучения.

        Возвращает:
            MetricsType: Словарь с результатами.
        """
        results: MetricsType = {**self.state}
        for callback in self.callbacks:
            result = callback.finalize(state)
            results.update(result)
        self.reset()
        return results
