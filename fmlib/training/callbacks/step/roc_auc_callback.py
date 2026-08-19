import warnings
from typing import Any, Dict, List, Self, cast

import accelerate
import torch
from sklearn.metrics import roc_auc_score

from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME
from fmlib.training.training_types import MetricsType, TrainingState


class RocAucCallback:
    """
    Коллбек, подсчитывающий ROC-AUC на задачах классификации.

    Аргументы:
        metric_name (str): Имя метрики, которую нужно сохранить в state.
            По умолчанию - `roc_auc`.
        target_name (str): Имя таргета.
            По умолчанию - `target`.
        output_name (str): Имя выхода модели.
            По умолчанию - `logits`.
        target_slice (tuple): Срез таргета. Применяется для получения колонки из таргета.
            По умолчанию - `(...,)`.
        output_slice (tuple): Срез выхода модели. Применяется для получения колонки из выходов.
            По умолчанию - `(...,)`.
        roc_auc_kwargs (Dict[str, Any] | None): Аргументы для `roc_auc_score` из scikit-learn.
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
        metric_name: str = "roc_auc",
        target_name: str = "target",
        output_name: str = "logits",
        target_slice: tuple = (...,),
        output_slice: tuple = (...,),
        roc_auc_kwargs: Dict[str, Any] | None = None,
        verbose_one_class: bool | None = None,
        verbose: bool = True,
        target_mask_name: str | None = None,
    ) -> None:
        if roc_auc_kwargs is None:
            roc_auc_kwargs = {}
        if target_mask_name is None:
            make_mask_name = DEFAULT_MAKE_MASK_NAME
            target_mask_name = make_mask_name(target_name)

        if verbose_one_class is not None:
            msg: str = "`verbose_one_class` parameter is deprecated. It will override `verbose`."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            verbose = cast(bool, verbose_one_class)

        self.target_mask_name: str = target_mask_name
        self.metric_name: str = metric_name
        self.target_name: str = target_name
        self.output_name: str = output_name
        self.target_slice: slice = target_slice
        self.output_slice: slice = output_slice

        self.target_masks: List[torch.Tensor] = []
        self.outputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

        self.verbose: bool = verbose
        self.roc_auc_kwargs: Dict[str, Any] = roc_auc_kwargs

    @property
    def output_names(self: Self) -> List[str]:
        """
        Подписка на выходы модели.
        """
        return [self.output_name]

    @property
    def transformed_names(self: Self) -> List[str]:
        """
        Подписка на таргеты.
        """
        return [self.target_name]

    def reset(self: Self) -> Self:
        """
        Очистка состояния коллбека.
        """
        self.outputs = []
        self.targets = []
        self.target_masks = []
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
                    result = roc_auc_score(targets.numpy(), outputs.numpy(), **self.roc_auc_kwargs)
                elif self.verbose:
                    warnings.warn(f"Has one class only. {has_positives=}, {has_negatives=}", stacklevel=2)
            elif self.verbose:
                warnings.warn("All targets are masked out.", stacklevel=2)

        full_result: List[float] = accelerate.utils.broadcast_object_list([result])

        assert len(full_result) == 1

        return {self.metric_name: full_result[-1]}
