"""Which populations PSI compares, and what happens when they are missing.

``PsiSelector`` picks its baseline/actual pair from ``config.mode``:

* ``train_valid`` compares ``datasets['train']`` against ``datasets['valid']``,
  or against the ``valid`` rows of ``FeatureSchema.split``;
* ``month_over_month`` splits ``train`` on ``month_column``.

``datasets['test']`` is never read in either mode: the held-out split has to
stay out of selection so the resulting feature set can be judged on data that
took no part in choosing it. A missing population raises instead of passing
every feature, because a silent skip reads in the report exactly like
"measured and stable".
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, PsiConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.psi import PsiSelector
from fmlib.feature_selection.utils.conftest import require_spark_session

_ROWS = 2000
_FEATURE = "f"
_TARGET = "target"
_MONTH = "month_part"
_DRIFTED_PSI = 1.0


def _frame(loc: float, months: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Build a frame whose single feature is centred on ``loc``."""
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
    """Build a context over pandas splits sharing one schema."""
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
    """Run PSI over ``datasets`` and report the single feature's outcome.

    ``select`` returns drops only, so the measured value comes from the scores
    the selector records for every candidate; ``keep`` is the absence of a drop.
    """
    selector = PsiSelector(PsiConfig(mode=mode, threshold=0.1))
    context = _context(datasets, split=split)
    decisions = selector.select(context, [_FEATURE])
    dropped = {decision.feature for decision in decisions}
    return SimpleNamespace(
        value=context.scores["psi"]["values"][_FEATURE],
        keep=_FEATURE not in dropped,
    )


@pytest.mark.parametrize("mode", ["train_valid", "month_over_month"])
def test_held_out_test_split_never_reaches_selection(mode: str) -> None:
    """A far-shifted ``test`` split must not change any decision."""
    datasets = {
        "train": _frame(0.0, (1, 2, 3), seed=1),
        "valid": _frame(0.0, (1, 2, 3), seed=2),
    }
    without_test = _decide(datasets, mode)
    with_test = _decide({**datasets, "test": _frame(5.0, (4,), seed=3)}, mode)

    assert with_test.value == pytest.approx(without_test.value)
    assert with_test.keep


def test_train_valid_mode_detects_drift_between_train_and_valid() -> None:
    """A drifted ``valid`` split is what ``train_valid`` is meant to catch."""
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
    """Matching populations must score near zero rather than merely pass."""
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
    """With no ``valid`` dataset the split column supplies the actual set."""
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
    """The month split must not depend on the Spark DataFrame API."""
    frame = pd.concat(
        [_frame(0.0, (1, 2), seed=1), _frame(5.0, (3,), seed=2)],
        ignore_index=True,
    )
    decision = _decide({"train": frame}, "month_over_month")

    assert decision.value > _DRIFTED_PSI
    assert not decision.keep


def test_missing_valid_population_is_reported_not_skipped() -> None:
    """Without ``valid`` or a split column the run must fail loudly."""
    with pytest.raises(ExecutionError, match="requires datasets\\['valid'\\]"):
        _decide({"train": _frame(0.0, (1, 2, 3), seed=1)}, "train_valid")


def test_too_few_periods_for_month_over_month_is_reported() -> None:
    """One period cannot be split into earlier and latest halves."""
    with pytest.raises(ExecutionError, match="distinct periods"):
        _decide({"train": _frame(0.0, (1,), seed=1)}, "month_over_month")


def test_missing_month_column_is_reported() -> None:
    """A month column absent from the frame must name itself in the error."""
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
