import warnings
from typing import Any, Dict, List, Self

import accelerate
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from fmlib.training.callbacks.step.external import qini_auc_score, uplift_at_k, uplift_auc_score
from fmlib.training.training_types import MetricsType, TrainingState


def calculate_uplift_metrics(
    treatment_scores: np.ndarray,
    control_scores: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    ks: List[float] | None = None,
    prefix: str = "",
) -> MetricsType:
    """
    Подсчитывает Uplift метрики с использованием библиотеки sklift.

    Аргументы:
        treatment_scores (np.ndarray): Массив оценок для таргетной группы.
        control_scores (np.ndarray): Массив оценок для контрольной группы.
        targets (np.ndarray): Массив целевых значений (взят ли продукт).
        groups (np.ndarray): Массив групп (таргетная или контрольная).
        ks (List[float] | None): Список значений k для uplift_at_k.
            По умолчанию - `None`, без подсчёта.
        prefix (str): Префикс для имен метрик. Будет добавлен к названию метрик.
    """

    if ks is None:
        ks = []
    uplifts: np.ndarray = treatment_scores - control_scores
    metrics_at_k: MetricsType = {
        f"{prefix}uplift_at_{k}": uplift_at_k(treatment=groups, uplift=uplifts, y_true=targets, strategy="overall", k=k)
        for k in sorted(ks)
    }
    metrics_all: MetricsType = {
        f"{prefix}treatment_roc_auc_score": roc_auc_score(
            y_score=treatment_scores[groups == 1.0],
            y_true=targets[groups == 1.0],
        ),
        f"{prefix}control_roc_auc_score": roc_auc_score(
            y_score=control_scores[groups == 0.0],
            y_true=targets[groups == 0.0],
        ),
        f"{prefix}uplift_auc_score": uplift_auc_score(
            treatment=groups,
            y_true=targets,
            uplift=uplifts,
        ),
        f"{prefix}qini_auc_score": qini_auc_score(
            treatment=groups,
            y_true=targets,
            uplift=uplifts,
        ),
    }
    metrics: MetricsType = {**metrics_at_k, **metrics_all}
    return metrics


class UpliftCallback:
    """
    Коллбек, подсчитывающий Uplift метрики:
        - uplift_at_k
        - uplift_auc_score
        - qini_auc_score
        - treatment_roc_auc_score
        - control_roc_auc_score

    Аргументы:
        group_name (str): Имя флага группы, будет взят из `transformed`.
            По умолчанию - `group`.
        target_name (str): Имя таргета, будет взят из `transformed`.
            По умолчанию - `target`.
        control_output_name (str): Имя выхода модели для контрольной группы.
            По умолчанию - `control_logits`.
        treatment_output_name (str): Имя выхода модели для таргетной группы.
            По умолчанию - `treatment_logits`.
        group_slice (tuple): Срез флагов группы. Применяется для получения колонки из флагов группы.
            По умолчанию - `(...,)`.
        target_slice (tuple): Срез таргета. Применяется для получения колонки из таргета.
            По умолчанию - `(...,)`.
        control_output_slice (tuple): Срез выхода модели для контрольной группы.
        Применяется для получения колонки из выходов.
            По умолчанию - `(...,)`.
        treatment_output_slice (tuple): Срез выхода модели для таргетной группы.
        Применяется для получения колонки из выходов.
            По умолчанию - `(...,)`.
        invert_groups (bool): Если `True`, то в качестве таргета будет взят противоположный флаг.
            По умолчанию - `True`.
        common_prefix (str): Префикс для имен метрик. Будет добавлен к названию метрик.
            По умолчанию - "", пустая строка.
        ks (List[float] | None): Список значений k для uplift_at_k.
            По умолчанию - `None`, без подсчёта.
        classifier_template (DictConfig | None): Шаблон для создания классификатора калибровки.
            По умолчанию - `None`, без создания калибровки.
            Обязан быть создан с флагом `_partial_: True`.

    Поля:
        groups (List[torch.Tensor]): Список флагов группы для данного шага обучения или валидации.
        targets (List[torch.Tensor]): Список таргетов для данного шага обучения или валидации.
        control_outputs (List[torch.Tensor]): Список контрольных выходов модели для данного шага обучения или валидации.
        treatment_outputs (List[torch.Tensor]): Список таргетных выходов модели для данного шага обучения или валидации.
    """

    def __init__(
        self: Self,
        group_name: str = "group",
        target_name: str = "target",
        control_output_name: str = "control_logits",
        treatment_output_name: str = "treatment_logits",
        group_slice: tuple = (...,),
        target_slice: tuple = (...,),
        control_output_slice: tuple = (...,),
        treatment_output_slice: tuple = (...,),
        invert_groups: bool = True,
        common_prefix: str = "",
        ks: List[float] | None = None,
        classifier_template: DictConfig | None = None,
    ) -> None:
        if ks is None:
            ks = []

        for k in sorted(ks):
            if k <= 0.0:
                msg: str = f"Quantile {k=} must be positive."
                raise ValueError(msg)
            elif k >= 1.0:
                msg: str = f"Quantile {k=} must be less than 1.0."
                raise ValueError(msg)

        self.group_name: str = group_name
        self.target_name: str = target_name
        self.control_output_name: str = control_output_name
        self.treatment_output_name: str = treatment_output_name

        self.group_slice: slice = group_slice
        self.target_slice: slice = target_slice
        self.control_output_slice: slice = control_output_slice
        self.treatment_output_slice: slice = treatment_output_slice

        self.groups: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.control_outputs: List[torch.Tensor] = []
        self.treatment_outputs: List[torch.Tensor] = []

        self.ks: List[float] = sorted(ks)
        self.invert_groups: bool = invert_groups
        self.common_prefix: str = common_prefix
        self.classifier_template: DictConfig | None = classifier_template

    @property
    def output_names(self: Self) -> List[str]:
        """
        Подписка на выходы модели.
        """
        return [self.control_output_name, self.treatment_output_name]

    @property
    def transformed_names(self: Self) -> List[str]:
        """
        Подписка на таргеты.
        """
        return [self.target_name, self.group_name]

    def reset(self: Self) -> Self:
        """
        Очистка состояния коллбека.
        """
        self.treatment_outputs = []
        self.control_outputs = []
        self.targets = []
        self.groups = []
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
            treatment_output: torch.Tensor = outputs[self.treatment_output_name]
            treatment_output = treatment_output[*self.treatment_output_slice].detach().cpu()

            control_output: torch.Tensor = outputs[self.control_output_name]
            control_output = control_output[*self.control_output_slice].detach().cpu()

            target: torch.Tensor = transformed[self.target_name]
            target = target[*self.target_slice].detach().cpu()

            group: torch.Tensor = transformed[self.group_name]
            group = group[*self.group_slice].detach().cpu()

            self.treatment_outputs.append(treatment_output)
            self.control_outputs.append(control_output)

            self.targets.append(target)
            self.groups.append(group)

        return {}

    def classifier_metrics(self: Self, classifier: Any, prefix: str) -> MetricsType:
        """
        Возвращает коеффициенты для классификатора, если есть.
        """

        def get_attr_value(attr: str) -> MetricsType:
            result: MetricsType = {}
            try:
                if hasattr(classifier, attr):
                    raw_coef: Any = getattr(classifier, attr)
                    coef: np.ndarray = np.asarray(raw_coef)
                    if coef.size == 1:
                        name: str = f"{prefix}_{attr}"
                        result = {name: float(coef.ravel().item())}
                    else:
                        msg: str = f"Too many elements in {attr=} for metric."
                        warnings.warn(msg, stacklevel=2)
            except ValueError as error:
                msg: str = f"Unable to convert calibrator attribute {attr=}: {error=}"
                warnings.warn(msg, stacklevel=2)
            return result

        c: MetricsType = get_attr_value("C_")
        coefs: MetricsType = get_attr_value("coef_")
        intercepts: MetricsType = get_attr_value("intercept_")
        result: MetricsType = {**c, **coefs, **intercepts}

        return result

    def compute_not_calibrated_metrics(
        self: Self, treatment_scores: np.ndarray, control_scores: np.ndarray, targets: np.ndarray, groups: np.ndarray
    ) -> MetricsType:
        """
        Подсчитывает метрики на не калиброванных данных.
        """
        uncalibrated_metrics: MetricsType = calculate_uplift_metrics(
            prefix=f"{self.common_prefix}not_calibrated_",
            treatment_scores=treatment_scores,
            control_scores=control_scores,
            targets=targets,
            groups=groups,
            ks=self.ks,
        )
        return uncalibrated_metrics

    def compute_calibrated_metrics(
        self: Self, treatment_scores: np.ndarray, control_scores: np.ndarray, targets: np.ndarray, groups: np.ndarray
    ) -> MetricsType:
        """
        Подсчитывает метрики с калибровкой:
        - Создает калибраторные классификаторы для каждой группы
        - Обучает на срезах по группам
        - Калибрует предсказания
        - Подсчитывает метрики на калиброванных данных
        - (Опционально) Получает коэффициенты из классификаторов
        - Возвращает всю информацию
        """
        t_calibrator: Any = self.classifier_template()
        c_calibrator: Any = self.classifier_template()

        treatment_x: np.ndarray = treatment_scores[groups == 1.0].reshape(-1, 1)
        control_x: np.ndarray = control_scores[groups == 0.0].reshape(-1, 1)

        t_calibrator.fit(treatment_x, targets[groups == 1.0].reshape(-1, 1))
        c_calibrator.fit(control_x, targets[groups == 0.0].reshape(-1, 1))

        calibrated_treatment_scores: np.ndarray = t_calibrator.predict_proba(treatment_scores.reshape(-1, 1))[:, 1]
        calibrated_control_scores: np.ndarray = c_calibrator.predict_proba(control_scores.reshape(-1, 1))[:, 1]

        calibrated_metrics: MetricsType = calculate_uplift_metrics(
            prefix=f"{self.common_prefix}calibrated_",
            treatment_scores=calibrated_treatment_scores,
            control_scores=calibrated_control_scores,
            targets=targets,
            groups=groups,
            ks=self.ks,
        )

        t_calibrator_prefix: str = f"{self.common_prefix}calibrated_t"
        t_calibrator_metrics: MetricsType = self.classifier_metrics(
            prefix=t_calibrator_prefix,
            classifier=t_calibrator,
        )
        calibrated_metrics.update(t_calibrator_metrics)

        c_calibrator_prefix: str = f"{self.common_prefix}calibrated_c"
        c_calibrator_metrics: MetricsType = self.classifier_metrics(
            prefix=c_calibrator_prefix,
            classifier=c_calibrator,
        )
        calibrated_metrics.update(c_calibrator_metrics)

        return calibrated_metrics

    def finalize(self: Self, state: TrainingState, epoch: int = 0) -> MetricsType:
        """
        Подсчитывает и возвращает результаты.
        """
        accelerator: accelerate.Accelerator = state["accelerator"]

        result: MetricsType = {}
        if accelerator.is_main_process:
            treatment_outputs: torch.Tensor = torch.cat(self.treatment_outputs, dim=0).float()
            control_outputs: torch.Tensor = torch.cat(self.control_outputs, dim=0).float()
            targets: torch.Tensor = torch.cat(self.targets, dim=0).bool()
            groups: torch.Tensor = torch.cat(self.groups, dim=0).bool()

            if self.invert_groups:
                groups = ~groups

            def to_numpy(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
                return tensor.to(dtype=dtype).numpy().ravel()

            np_treatment_scores: np.ndarray = to_numpy(treatment_outputs)
            np_control_scores: np.ndarray = to_numpy(control_outputs)
            np_targets: np.ndarray = to_numpy(targets)
            np_groups: np.ndarray = to_numpy(groups)

            del treatment_outputs, control_outputs, targets, groups

            not_calibrated_metrics: MetricsType = self.compute_not_calibrated_metrics(
                treatment_scores=np_treatment_scores,
                control_scores=np_control_scores,
                targets=np_targets,
                groups=np_groups,
            )

            calibrated_metrics: MetricsType = {}
            if self.classifier_template is not None:
                calibrated_metrics: MetricsType = self.compute_calibrated_metrics(
                    treatment_scores=np_treatment_scores,
                    control_scores=np_control_scores,
                    targets=np_targets,
                    groups=np_groups,
                )

            del np_groups, np_targets, np_control_scores, np_treatment_scores

            result = {**calibrated_metrics, **not_calibrated_metrics}

        full_result: List[MetricsType] = accelerate.utils.broadcast_object_list([result])
        assert len(full_result) == 1
        return full_result[-1]
