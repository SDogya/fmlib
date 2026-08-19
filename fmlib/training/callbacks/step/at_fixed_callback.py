import warnings
from typing import Any, Self, cast

import accelerate
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from fmlib.constants.callbacks import DEFAULT_FIXED_PRECISIONS, DEFAULT_FIXED_RECALLS
from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME
from fmlib.training.training_types import MetricsType, TrainingState


class AtFixedCallback:
    """
    Коллбек, подсчитывающий Precision@Recall & Recall@Precision на задачах классификации.

    Аргументы:
        precision_at_recall_template (str): Шаблон для имени метрики Precision@Recall.
            По умолчанию - `precision@{recall}`, строка должна содержать `{recall}`.
        recall_at_precision_template (str): Шаблон для имени метрики Recall@Precision.
            По умолчанию - `recall@{precision}`, строка должна содержать `{precision}`.
        fixed_recalls (list[float] | None): Список фиксированных значений Recall.
            По умолчанию - DEFAULT_FIXED_RECALLS.
        fixed_precisions (list[float] | None): Список фиксированных значений Precision.
            По умолчанию - DEFAULT_FIXED_PRECISIONS.
        target_name (str): Имя таргета.
            По умолчанию - `target`.
        output_name (str): Имя выхода модели.
            По умолчанию - `logits`.
        target_slice (tuple): Срез таргета. Применяется для получения колонки из таргета.
            По умолчанию - `(...,)`.
        output_slice (tuple): Срез выхода модели. Применяется для получения колонки из выходов.
            По умолчанию - `(...,)`.
        precision_recall_kwargs (Dict[str, Any] | None): Аргументы для `precision_recall_curve` из scikit-learn.
            По умолчанию - `None`.
        verbose_one_class (bool | None): Deprecated. То же, что `verbose`.
            По умолчанию - None.
        verbose (bool): Если `True`, выводит предупреждение, если в выборке только один класс.
            По умолчанию - `True`.
        target_mask_name (str | None): Название тензора масок таргетов.
            По умолчанию - `None`, т.е. будет получена применением
            к `target_name` функции `DEFAULT_MAKE_MASK_NAME`.

    Поля:
        targets (List[torch.Tensor]): Список таргетов для данного шага обучения или валидации.
        outputs (List[torch.Tensor]): Список выходов модели для данного шага обучения или валидации.
        target_masks (List[torch.Tensor]): Список тензоров маскирования таргетов.
    """

    def __init__(
        self: Self,
        precision_at_recall_template: str = "precision@{recall}",
        recall_at_precision_template: str = "recall@{precision}",
        fixed_recalls: list[float] | None = DEFAULT_FIXED_RECALLS,
        fixed_precisions: list[float] | None = DEFAULT_FIXED_PRECISIONS,
        target_name: str = "target",
        output_name: str = "logits",
        target_slice: tuple = (...,),
        output_slice: tuple = (...,),
        precision_recall_kwargs: dict[str, Any] | None = None,
        verbose_one_class: bool | None = None,
        verbose: bool = True,
        target_mask_name: str | None = None,
    ) -> None:
        if precision_recall_kwargs is None:
            precision_recall_kwargs = {}

        if target_mask_name is None:
            make_mask_name = DEFAULT_MAKE_MASK_NAME
            target_mask_name = make_mask_name(target_name)

        if (fixed_recalls is None) and (fixed_precisions is None):
            msg: str = "Please provide either `fixed_recalls` or `fixed_precisions`."
            raise ValueError(msg)
        if fixed_recalls is None:
            fixed_recalls = []
        if fixed_precisions is None:
            fixed_precisions = []

        if verbose_one_class is not None:
            msg: str = "`verbose_one_class` parameter is deprecated. It will override `verbose`."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            verbose = cast(bool, verbose_one_class)

        self.target_name: str = target_name
        self.output_name: str = output_name
        self.target_mask_name: str = target_mask_name
        self.target_slice: slice = target_slice
        self.output_slice: slice = output_slice
        self.outputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.target_masks: list[torch.Tensor] = []

        self.verbose: bool = verbose

        self.fixed_recalls: list[float] = cast(list[float], sorted(fixed_recalls))
        self.fixed_precisions: list[float] = cast(list[float], sorted(fixed_precisions))

        self.precision_at_recall_template: str = precision_at_recall_template
        self.recall_at_precision_template: str = recall_at_precision_template

        self.precision_recall_kwargs: dict[str, Any] = cast(dict[str, Any], precision_recall_kwargs)

    @property
    def output_names(self: Self) -> list[str]:
        """
        Подписка на выходы модели.
        """
        return [self.output_name]

    @property
    def transformed_names(self: Self) -> list[str]:
        """
        Подписка на таргеты.
        """
        return [self.target_name]

    def reset(self: Self) -> Self:
        """
        Очистка состояния коллбека.
        """
        self.target_masks = []
        self.outputs = []
        self.targets = []
        return self

    def __call__(
        self: Self,
        state: TrainingState,
        transformed: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        epoch: int = 0,
        external: MetricsType | None = None,
    ) -> MetricsType:
        """
        Сохраняет во внутреннее состояние результаты работы модели и таргеты.
        """
        if external is None:
            external = {}

        accelerator = state["accelerator"]
        if accelerator.is_main_process:
            output: torch.Tensor = outputs[self.output_name]
            output = output[*self.output_slice].detach().cpu()

            target: torch.Tensor = transformed[self.target_name]
            target = target[*self.target_slice].detach().cpu()

            if self.target_mask_name in transformed:
                target_mask: torch.Tensor = transformed[self.target_mask_name]
                target_mask = target_mask[*self.target_slice].detach().cpu()
            else:
                target_mask = torch.ones_like(target, dtype=torch.bool)

            self.outputs.append(output)
            self.targets.append(target)
            self.target_masks.append(target_mask)

        return {}

    def compute_results(self: Self, targets: np.ndarray, outputs: np.ndarray) -> dict[str, float]:
        precisions, recalls, _ = precision_recall_curve(targets, outputs, **self.precision_recall_kwargs)

        result: dict[str, float] = {}

        if len(self.fixed_precisions) > 0:
            fixed_precisions: np.ndarray = np.asarray(self.fixed_precisions, dtype=np.float32).ravel()
            precision_indices: np.ndarray = np.searchsorted(precisions, fixed_precisions, side="left")

            precision_indices = np.clip(precision_indices, 0, len(precisions) - 1)
            recall_at_precision: np.ndarray = recalls[precision_indices]

            for recall, precision in zip(recall_at_precision, self.fixed_precisions, strict=True):
                name: str = self.recall_at_precision_template.format(precision=precision)
                result[name] = float(recall)

            del precision_indices, recall_at_precision

        if len(self.fixed_recalls) > 0:
            # Reordering `...[::-1]`` is needed because `recalls` are descending.
            fixed_recalls: np.ndarray = np.asarray(self.fixed_recalls, dtype=np.float32).ravel()
            recall_indices: np.ndarray = np.searchsorted(recalls[::-1], fixed_recalls, side="left")

            recall_indices = np.clip(recall_indices, 0, len(recalls) - 1)
            precision_at_recall: np.ndarray = precisions[::-1][recall_indices]

            for recall, precision in zip(self.fixed_recalls, precision_at_recall, strict=True):
                name: str = self.precision_at_recall_template.format(recall=recall)
                result[name] = float(precision)

            del recall_indices, precision_at_recall

        return result

    def finalize(self: Self, state: TrainingState, epoch: int = 0) -> MetricsType:
        """
        Подсчитывает и возвращает результаты.
        """
        accelerator: accelerate.Accelerator = state["accelerator"]

        result: float = float("nan")
        if accelerator.is_main_process:
            masks: torch.Tensor = torch.cat(self.target_masks, dim=0).bool()
            if torch.any(masks).item():
                outputs: torch.Tensor = torch.cat(self.outputs, dim=0).float()
                targets: torch.Tensor = torch.cat(self.targets, dim=0).bool()

                outputs, targets = outputs[masks], targets[masks]

                has_positives: bool = torch.any(targets).item()
                has_negatives: bool = torch.any(~targets).item()

                if has_positives and has_negatives:
                    result = self.compute_results(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())
                elif self.verbose:
                    warnings.warn(f"Has one class only. {has_positives=}, {has_negatives=}", stacklevel=2)
            elif self.verbose:
                warnings.warn("All targets are masked out.", stacklevel=2)

        full_result: list[MetricsType] = accelerate.utils.broadcast_object_list([result])
        assert len(full_result) == 1
        return full_result[-1]
