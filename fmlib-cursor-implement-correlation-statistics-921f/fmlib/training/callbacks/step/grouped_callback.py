import warnings
from typing import Dict, List, Protocol, Self

import accelerate
import torch

from fmlib.constants.batches import Batch
from fmlib.data.io.collate import general_collate
from fmlib.training.training_types import MetricsType, TrainingState
from fmlib.utils.to_device import to_device


class StepCallbackProtocol(Protocol):
    """
    Протокол для коллбеков, вызываемых в конце каждого батча на обучении и валидации.
    """

    @property
    def output_names(self: Self) -> List[str]:
        """
        Возвращает список имен выходов модели, которые требуются в коллбеке.
        """
        ...

    @property
    def transformed_names(self: Self) -> List[str]:
        """
        Возвращает список имен выходов трансформов, которые требуются в коллбеке.
        """
        ...

    def reset(self: Self) -> Self:
        """
        Очищает состояние коллбека.
        """
        ...

    def __call__(
        self: Self,
        state: TrainingState,
        transformed: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        epoch: int = 0,
        external: MetricsType | None = None,
    ) -> MetricsType:
        """
        Осуществляет расчеты коллбека на каждом шаге.

        Примечание: Как правило это сводится к накоплению данных для финализации.

        Аргументы:
            state(TrainingState): Состояние обучения.
            transformed (Dict[str, torch.Tensor]): Преобразованные данные.
            outputs (Dict[str, torch.Tensor]): Выходы модели.
            epoch (int): Номер эпохи.
                По умолчанию - `0`.
            external (MetricsType | None): Внешние метрики.
        """
        ...

    def finalize(self: Self, state: TrainingState, epoch: int = 0) -> MetricsType:
        """
        Осуществляет финальный подсчет и возвращает результаты.

        Аргументы:
        state (TrainingState): Состояние обучения.
        epoch (int): Номер эпохи.
            По умолчанию - `0`.
        """
        ...


def filter_by_keys(keys: List[str], batch: Batch) -> Batch:
    result: Batch = {key: batch[key].detach() for key in keys}
    return result


def get_unique(names: List[List[str]]) -> List[str]:
    results: set = set()
    for name in names:
        results = results | set(name)
    return sorted(results)


def get_transformed_names(callbacks: List[StepCallbackProtocol], forced: List[str] | None = None) -> List[str]:
    if forced is None:
        forced = []

    def _get_transformed_names(callback) -> List[str]:
        if hasattr(callback, "transformed_names"):
            return list(callback.transformed_names)
        warnings.warn('No "transformed_names" found.', stacklevel=2)
        return []

    result: List[List[str]] = [_get_transformed_names(cb) for cb in callbacks]
    return get_unique([*result, forced])


def get_output_names(callbacks: List[StepCallbackProtocol], forced: List[str] | None = None) -> List[str]:
    if forced is None:
        forced = []

    def _get_output_names(callback) -> List[str]:
        if hasattr(callback, "output_names"):
            return list(callback.output_names)
        warnings.warn('No "output_names" found.', stacklevel=2)
        return []

    result: List[List[str]] = [_get_output_names(cb) for cb in callbacks]
    return get_unique([*result, forced])


def adapt_externals(accelerator: accelerate.Accelerator, externals: Batch, template: str = "{name}_rank_{rank}") -> Batch:
    rank: int = accelerator.process_index

    def _make_name(name: str) -> str:
        return template.format(name=name, rank=rank)

    adapted = [(_make_name(name), value) for name, value in externals.items()]
    all_adapted = accelerator.gather_for_metrics(adapted, use_gather_object=True)
    return dict(all_adapted)


class GroupedStepCallback:
    """
    Общий коллбек, вызываемый в конце каждого батча шага обучения.
    Объединяет несколько отдельных коллбеков, объединённых `StepCallbackProtocol`.

    Аргументы:
        callbacks (List[StepCallbackProtocol]): Список коллбеков.
        forced_outputs (List[str] | None): Список подписок на выходные данные.
            По умолчанию - `None`.
        forced_transforms (List[str] | None): Список подписок на преобразованные данные.
            По умолчанию - `None`.
    """

    def __init__(
        self: Self,
        callbacks: List[StepCallbackProtocol],
        forced_outputs: List[str] | None = None,
        forced_transforms: List[str] | None = None,
    ) -> None:
        if forced_transforms is None:
            forced_transforms = []
        if forced_outputs is None:
            forced_outputs = []
        self.stats: MetricsType = {}

        self.callbacks: List[StepCallbackProtocol] = callbacks
        self.output_names: List[str] = get_output_names(callbacks, forced_outputs)
        self.transformed_names: List[str] = get_transformed_names(callbacks, forced_transforms)

    def reset(self: Self) -> Self:
        """
        Сбрасывает состояние коллбеков.

        Возвращает:
            Self: Объект коллбека.
        """
        self.stats = {}
        for callback in self.callbacks:
            callback.reset()
        return self

    def __call__(
        self: Self,
        state: TrainingState,
        transformed: Batch,
        outputs: Batch,
        epoch: int = 0,
        external: MetricsType | None = None,
    ) -> MetricsType:
        """
        Вызывает все внутренние коллбеки.

        Внимание: Все коллбеки вызываются только на основном процессе и получают
        все данные по соответствующим именам.

        Аргументы:
            state (TrainingState): Состояние обучения.
            transformed (Batch): Преобразованные данные.
            outputs (Batch): Выходные данные.
            epoch (int): Номер эпохи.
                По умолчанию - `0`.
            external (MetricsType | None): Внешние метрики.
                По умолчанию - `None`.

        Возвращает:
            MetricsType: Словарь с результатами.
        """
        if external is None:
            external = {}
        accelerator: accelerate.Accelerator = state["accelerator"]
        accelerator.wait_for_everyone()

        results: MetricsType = adapt_externals(accelerator, external)

        if len(self.callbacks) > 0:
            outputs_filtered: Batch = filter_by_keys(self.output_names, outputs)
            transformed_filtered: Batch = filter_by_keys(self.transformed_names, transformed)
            all_objects: Batch = accelerator.gather_for_metrics((transformed_filtered, outputs_filtered))

            if accelerator.is_main_process:
                assert len(all_objects) % 2 == 0
                all_outputs: Batch = general_collate([t for i, t in enumerate(all_objects) if i % 2 == 1])
                all_transformed: Batch = general_collate([t for i, t in enumerate(all_objects) if i % 2 == 0])

                all_outputs, all_transformed = to_device(all_outputs), to_device(all_transformed)
            else:

                def simulate_data(batch: Batch) -> Batch:
                    result: Batch = {}
                    for key in sorted(batch.keys()):
                        sample: torch.Tensor = batch[key]
                        result[key] = torch.tensor([], dtype=sample.dtype)
                    return result

                all_outputs, all_transformed = simulate_data(outputs_filtered), simulate_data(transformed_filtered)

            for callback in self.callbacks:
                result: MetricsType = callback(state=state, epoch=epoch, transformed=all_transformed, outputs=all_outputs)
                results.update(result)

        results: List[MetricsType] = accelerate.utils.broadcast_object_list([results])
        assert len(results) == 1
        self.stats = results[-1]

        return self.stats

    def finalize(self: Self, state: TrainingState, epoch: int = 0) -> MetricsType:
        """
        Собирает результаты, сбрасывает состояния коллбеков и возвращает.

        Аргументы:
            state (TrainingState): Состояние обучения.
            epoch (int): Номер эпохи.
                По умолчанию - `0`.

        Возвращает:
            MetricsType: Словарь с результатами.
        """
        results: MetricsType = {**self.stats}
        for callback in self.callbacks:
            result: MetricsType = callback.finalize(state=state, epoch=epoch)
            results.update(result)
            callback.reset()
        self.reset()
        return results
