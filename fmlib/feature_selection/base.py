"""Общие контракты и вспомогательные заглушки для этапов отбора признаков."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from fmlib.feature_selection.utils.verbose import VerboseRecorder, default_verbose_recorder


@dataclass(frozen=True)
class FeatureDecision:
    """Решение по отдельному признаку, принятое на этапе отбора.

    Args:
        feature: Имя признака.
        stage: Имя этапа пайплайна (``preprocessing``, ``statistics`` или
            ``model``).
        method: Имя метода отбора внутри этапа.
        reason: Машиночитаемый код причины.
        value: Измеренное значение метрики, если есть.
        threshold: Порог, использованный для принятия решения, если есть.
        keep: Остаётся ли признак кандидатом после этого решения.
    """

    feature: str
    stage: str
    method: str
    reason: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    keep: bool = False

    def to_dict(self: "FeatureDecision") -> dict[str, Any]:
        """Сериализует в обычный словарь."""
        return {
            "feature": self.feature,
            "stage": self.stage,
            "method": self.method,
            "reason": self.reason,
            "value": self.value,
            "threshold": self.threshold,
            "keep": self.keep,
        }

    @classmethod
    def from_dict(cls: type["FeatureDecision"], payload: dict[str, Any]) -> "FeatureDecision":
        """Создаёт объект из словаря."""
        return cls(
            feature=payload["feature"],
            stage=payload["stage"],
            method=payload["method"],
            reason=payload["reason"],
            value=payload.get("value"),
            threshold=payload.get("threshold"),
            keep=payload.get("keep", False),
        )


@dataclass
class StageContext:
    """Контекст выполнения, передаваемый методам отбора.

    Args:
        spark: Активная сессия Spark (в каркасе используется утиная типизация).
        datasets: Словарь имён выборок и соответствующих объектов с интерфейсом DataFrame.
        schema: Проверенная схема признаков.
        config: Проверенная конфигурация пайплайна.
        seed: Базовый seed для воспроизводимости.
        candidates: Текущие имена признаков-кандидатов.
        scores: Словарь имён методов отбора и рассчитанных ими оценок важности.
        datasets_mode: Передан ли на вход один DataFrame или словарь выборок.
        decisions: Решения, накопленные на завершённых этапах пайплайна.
        verbose_log: Средство подробного журналирования по методам. Активно только при ``execution.verbose``.
        step_index: Позиция текущего шага пайплайна, начиная с 0.
        run_seed: Seed для этого шага. При ``None`` используется ``seed``.
        output_dir: Необязательный каталог промежуточных артефактов для каждого метода.
        local_numeric_sample: Кэшированный числовой DataFrame в памяти драйвера, совместно используемый
            LightGBM и BorutaSHAP при совпадении параметров выборки.
        statistics_row_transforms: Выполненные преобразования строк для ключа кэша статистик.
    """

    spark: Any
    datasets: dict[str, Any]
    schema: Any
    config: Any
    seed: int
    candidates: list[str]
    scores: dict[str, Any] = field(default_factory=dict)
    datasets_mode: str = "mapping"
    decisions: list[FeatureDecision] = field(default_factory=list)
    verbose_log: VerboseRecorder = field(default_factory=default_verbose_recorder)
    step_index: int = 0
    run_seed: Optional[int] = None
    output_dir: Optional[Path] = None
    local_numeric_sample: Optional[Any] = None
    statistics_row_transforms: list[dict[str, Any]] = field(default_factory=list)


def step_seed(context: StageContext) -> int:
    """Seed для текущего шага пайплайна."""
    if context.run_seed is None:
        return context.seed
    return context.run_seed


def resolve_step_seed(
    params: Mapping[str, Any] | None,
    context: StageContext,
) -> int:
    """Возвращает ``params.seed``, если задано, иначе ``execution.seed``.

    Сначала проверяет словарь шага, затем вложенный блок ``params``, чтобы
    ``- lightgbm: ${model}`` (где seed задан в ``model.params``) работал
    так же, как плоская запись ``- lightgbm: {seed: 17}``.

    Не использует ``context.run_seed`` как запасное значение: переопределение на предыдущем шаге
    не должно влиять на последующий шаг, в котором ``seed`` не указан.
    """
    if params is None:
        return context.seed
    if params.get("seed") is not None:
        return int(params["seed"])
    nested = params.get("params")
    if isinstance(nested, Mapping) and nested.get("seed") is not None:
        return int(nested["seed"])
    return context.seed


def bind_process_rng(seed: int) -> None:
    """Инициализирует общие для процесса генераторы ``random`` и NumPy значением ``seed``.

    ``PYTHONHASHSEED`` остаётся без изменений. Побитовая идентичность результатов не
    гарантируется, если модель использует ``n_jobs != 1`` или CatBoost на GPU.
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))


def persist_step_artifact(
    context: StageContext,
    remaining: Sequence[str],
    *,
    stage_name: str,
    method_name: str,
) -> None:
    """Записывает артефакт метода, если задан ``context.output_dir``.

    Имя файла — ``{step_index:02d}_{stage}_{method}_results.json``.
    """
    output_dir = context.output_dir
    if output_dir is None:
        return
    from fmlib.feature_selection.result import _save_intermediate_result

    _save_intermediate_result(
        remaining=list(remaining),
        decisions=context.decisions,
        stage_name=stage_name,
        method_name=method_name,
        output_dir=output_dir,
        context=context,
        step_index=context.step_index,
    )


class Selector(Protocol):
    """Протокол отдельного метода отбора признаков."""

    method_name: str

    def select(
        self: Selector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Оценивает кандидатов и возвращает решения о сохранении или исключении.

        Args:
            context: Общий контекст этапа.
            candidates: Признаки, которые ещё рассматриваются для отбора.

        Returns:
            Решения по исключённым и, при необходимости, сохранённым признакам. Исключённые признаки
            должны иметь ``keep=False`` и удаляются перед следующим методом отбора.
        """
        ...


def derive_seed(root_seed: int, *parts: str) -> int:
    """Вычисляет стабильный целочисленный seed из базового seed и идентификаторов этапа.

    Args:
        root_seed: Базовый seed из конфигурации выполнения.
        *parts: Идентификаторы этапа/метода, учитываемые при вычислении seed.

    Returns:
        Детерминированный 32-битный целочисленный seed.
    """
    payload = f"{root_seed}:" + ":".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def apply_drop_decisions(
    candidates: Sequence[str],
    decisions: Sequence[FeatureDecision],
) -> list[str]:
    """Удаляет исключённые признаки из кандидатов, сохраняя порядок.

    Args:
        candidates: Имена текущих кандидатов.
        decisions: Решения метода отбора; исключаются только записи с ``keep=False``.

    Returns:
        Оставшиеся кандидаты в исходном порядке.
    """
    dropped = {decision.feature for decision in decisions if not decision.keep}
    return [name for name in candidates if name not in dropped]
