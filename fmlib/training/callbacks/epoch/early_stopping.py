import warnings
from typing import Any, Dict, Iterator, Literal, Self

import accelerate

from fmlib.training.training_loop import TrainingState


class EarlyStoppingIterable:
    """
    Итерируемый объект, формально `Iterable[int]`.
    Служит для остановки процесса обучения, прекращает итерации,
    если `need_to_stop` равен `True`.

    Аргументы:
        max_iter (int): Максимальное ожидаемое количество итераций.

    Поля:
        max_iter (int): Максимальное ожидаемое количество итераций.
        need_to_stop (bool): Флаг необходимости остановки итераций.
    """

    def __init__(self: Self, max_iter: int) -> None:
        if max_iter < 1:
            msg: str = f"Number of iterations must be greater than 0. Got {max_iter=}."
            raise ValueError(msg)

        self.max_iter: int = max_iter
        self.need_to_stop: bool = False

    def __len__(self: Self) -> int:
        return self.max_iter

    def __iter__(self: Self) -> Iterator[int]:
        for i in range(self.max_iter):
            if self.need_to_stop:
                break
            try:
                yield i
            except KeyboardInterrupt:
                self.need_to_stop = True
                break


class EarlyStoppingCallback:
    """
    Коллбек отвечающий за остановку обучения после `patience` эпох без улучшения метрики.

    Аргументы:
        patience (int): Количество эпох без улучшения метрики, после которых обучение будет остановлено.
            По умолчанию - 1.
        metric_to_track (str): Имя метрики, по которой будет определяться чекпоинт.
            По умолчанию - `roc_auc`.
        direction (Literal["max", "min"]): Направление поиска чекпоинта.
            По умолчанию - сохраняются чекпоинты с наибольшей метрикой, `max`.
    """

    def __init__(
        self: Self, patience: int = 1, direction: Literal["min", "max"] = "max", metric_to_track: str = "roc_auc"
    ) -> None:
        if patience < 1:
            msg: str = f"Patience must be greater than 0. Got {patience=}."
            raise ValueError(msg)
        if direction not in {"min", "max"}:
            msg: str = f'Direction must be either "min" or "max". Got {direction=}.'
            raise ValueError(msg)

        self.direction: str = metric_to_track
        self.default_metric_value: float
        if self.direction == "min":
            self.default_metric_value = float("inf")
        else:
            self.default_metric_value = float("-inf")
        self.best_metric_epoch: int = 0
        self.best_metric_value: float = self.default_metric_value

        self.patience: int = patience
        self.metric_to_track: str = metric_to_track

    def reset(self: Self) -> Self:
        self.best_metric_epoch = 0
        self.best_metric_value = self.default_metric_value
        return self

    def __call__(self: Self, state: TrainingState, metrics: Dict[str, Any], epoch: int = 0) -> Dict[str, Any]:
        if self.metric_to_track not in metrics:
            warnings.warn("Metric to track not found. Defaulting.", stacklevel=2)

        metric_value: float = metrics.get(self.metric_to_track, self.default_metric_value)

        is_new_best: bool
        if self.direction == "min":
            is_new_best = metric_value < self.best_metric_value
        else:
            is_new_best = metric_value > self.best_metric_value

        if is_new_best:
            self.best_metric_epoch = epoch
            self.best_metric_value = metric_value

        local_need_to_stop: bool = self.patience <= (epoch - self.best_metric_epoch)

        need_to_stop: bool = any(accelerate.utils.gather_object([local_need_to_stop]))

        epoch_iterable: Any = state["epoch_iterable"]
        assert isinstance(epoch_iterable, EarlyStoppingIterable)
        epoch_iterable.need_to_stop = need_to_stop

        return {
            "stopped": need_to_stop,
            "best_metric_epoch": self.best_metric_epoch,
            "best_metric_value": self.best_metric_value,
        }

    def finalize(self: Self, state: TrainingState) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "best_metric_epoch": self.best_metric_epoch,
            "best_metric_value": self.best_metric_value,
        }
        self.reset()
        return result
