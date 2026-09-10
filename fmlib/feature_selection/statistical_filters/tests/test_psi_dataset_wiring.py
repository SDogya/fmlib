"""Какие совокупности сравнивает PSI и что происходит при их отсутствии.

``PsiSelector`` выбирает пару baseline/actual по ``config.mode``:

* ``train_valid`` сравнивает ``datasets['train']`` с ``datasets['valid']``
  или со строками ``valid`` из ``FeatureSchema.split``;
* ``month_over_month`` разделяет ``train`` по ``month_column``.

``datasets['test']`` не читается ни в одном режиме: отложенная выборка должна
оставаться вне отбора, чтобы итоговый набор признаков можно было оценить на данных,
не участвовавших в его выборе. При отсутствии совокупности возникает ошибка вместо сохранения
всех признаков, поскольку незаметный пропуск в отчёте выглядит точно так же, как
«измерено и стабильно».
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, PsiConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters.psi import PsiSelector
from fmlib.feature_selection.utils.conftest import require_spark_session

_ROWS = 2000
_FEATURE = "f"
_TARGET = "target"
_MONTH = "month_part"
_DRIFTED_PSI = 1.0


def _frame(loc: float, months: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Создаёт DataFrame с единственным признаком, распределённым вокруг ``loc``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            _FEATURE: rng.normal(loc, 1.0, _ROWS),
            _TARGET: rng.integers(0, 2, _ROWS),
            _MONTH: rng.choice(months, _ROWS),
        },
    )


def _context(
    datasets: dict[str, pd.DataFrame],
    *,
    split: str | None = None,
) -> StageContext:
    """Создаёт контекст для выборок pandas с общей схемой."""
    schema = FeatureSchema(
        categorical=(),
        continuous=(_FEATURE,),
        target=_TARGET,
        task_type="binary_classification",
        time=_MONTH,
        split=split,
    )
    return StageContext(
        spark=require_spark_session(),
        datasets=dict(datasets),
        schema=schema,
        config=FeatureSelectionConfig(),
        seed=42,
        candidates=[_FEATURE],
        run_seed=42,
    )


def _decide(
    datasets: dict[str, pd.DataFrame],
    mode: str,
    *,
    split: str | None = None,
) -> Any:
    """Выполняет PSI для ``datasets`` и возвращает решение по единственному признаку."""
    selector = PsiSelector(PsiConfig(mode=mode, threshold=0.1))
    return selector.select(_context(datasets, split=split), [_FEATURE])[0]


@pytest.mark.parametrize("mode", ["train_valid", "month_over_month"])
def test_held_out_test_split_never_reaches_selection(mode: str) -> None:
    """Сильный сдвиг в выборке ``test`` не должен влиять на решения."""
    datasets = {
        "train": _frame(0.0, (1, 2, 3), seed=1),
        "valid": _frame(0.0, (1, 2, 3), seed=2),
    }
    without_test = _decide(datasets, mode)
    with_test = _decide({**datasets, "test": _frame(5.0, (4,), seed=3)}, mode)

    assert with_test.value == pytest.approx(without_test.value)
    assert with_test.keep


def test_train_valid_mode_detects_drift_between_train_and_valid() -> None:
    """Режим ``train_valid`` должен обнаруживать сдвиг в выборке ``valid``."""
    decision = _decide(
        {
            "train": _frame(0.0, (1, 2, 3), seed=1),
            "valid": _frame(5.0, (1, 2, 3), seed=2),
        },
        "train_valid",
    )

    assert decision.value > _DRIFTED_PSI
    assert not decision.keep


def test_train_valid_mode_keeps_a_stable_feature() -> None:
    """Совпадающие совокупности должны давать оценку около нуля, а не просто проходить отбор."""
    decision = _decide(
        {
            "train": _frame(0.0, (1, 2, 3), seed=1),
            "valid": _frame(0.0, (1, 2, 3), seed=2),
        },
        "train_valid",
    )

    assert decision.value < 0.1
    assert decision.keep


def test_train_valid_mode_reads_valid_rows_from_the_split_column() -> None:
    """При отсутствии набора ``valid`` актуальная выборка определяется по столбцу разбиения."""
    frame = pd.concat(
        [
            _frame(0.0, (1, 2), seed=1).assign(part="train"),
            _frame(5.0, (1, 2), seed=2).assign(part="valid"),
        ],
        ignore_index=True,
    )
    decision = _decide({"train": frame}, "train_valid", split="part")

    assert decision.value > _DRIFTED_PSI
    assert not decision.keep


def test_month_over_month_splits_a_pandas_frame_by_time() -> None:
    """Разбиение по месяцам не должно зависеть от API Spark DataFrame."""
    frame = pd.concat(
        [_frame(0.0, (1, 2), seed=1), _frame(5.0, (3,), seed=2)],
        ignore_index=True,
    )
    decision = _decide({"train": frame}, "month_over_month")

    assert decision.value > _DRIFTED_PSI
    assert not decision.keep


def test_missing_valid_population_is_reported_not_skipped() -> None:
    """При отсутствии ``valid`` и столбца разбиения запуск должен завершаться явной ошибкой."""
    with pytest.raises(ExecutionError, match="requires datasets\\['valid'\\]"):
        _decide({"train": _frame(0.0, (1, 2, 3), seed=1)}, "train_valid")


def test_too_few_periods_for_month_over_month_is_reported() -> None:
    """Один период нельзя разделить на более раннюю и последнюю части."""
    with pytest.raises(ExecutionError, match="distinct periods"):
        _decide({"train": _frame(0.0, (1,), seed=1)}, "month_over_month")


def test_missing_month_column_is_reported() -> None:
    """Ошибка должна содержать имя отсутствующего в DataFrame столбца месяца."""
    frame = _frame(0.0, (1, 2, 3), seed=1).drop(columns=[_MONTH])
    selector = PsiSelector(PsiConfig(mode="month_over_month", threshold=0.1))
    schema = FeatureSchema(
        categorical=(),
        continuous=(_FEATURE,),
        target=_TARGET,
        task_type="binary_classification",
    )
    context = StageContext(
        spark=require_spark_session(),
        datasets={"train": frame},
        schema=schema,
        config=FeatureSelectionConfig(),
        seed=42,
        candidates=[_FEATURE],
        run_seed=42,
    )

    with pytest.raises(ExecutionError, match="month_column"):
        selector.select(context, [_FEATURE])
