"""Tests for the pieces of the runner a wrong answer would silently corrupt.

The override syntax decides which threshold a sweep actually measured, and the
per-seed baseline decides what ``auc_delta`` means. Both are easy to get wrong
in a way that produces a plausible table rather than an error.
"""

from __future__ import annotations

import pytest

from benchmarks.open_data.run_benchmark import (
    _apply_overrides,
    _flat_table,
    parse_pipeline_spec,
)


def test_parses_a_bare_name() -> None:
    """A spec without overrides is just the config name."""
    assert parse_pipeline_spec("cheap") == ("cheap", {})


def test_parses_several_overrides() -> None:
    """Overrides are comma separated and keep their raw text."""
    name, overrides = parse_pipeline_spec("only_iv@threshold=0.005,num_bins=20")
    assert name == "only_iv"
    assert overrides == {"threshold": "0.005", "num_bins": "20"}


def test_rejects_an_override_without_a_value() -> None:
    """A typo must fail loudly rather than sweep the default threshold."""
    with pytest.raises(ValueError, match="not key=value"):
        parse_pipeline_spec("only_iv@threshold")


def test_bare_key_targets_the_only_step() -> None:
    """Single-method configs take ``threshold=`` without naming the method."""
    payload = {"order": [{"iv": {"threshold": 0.02, "num_bins": 10}}]}
    _apply_overrides(payload, {"threshold": "0.005"})
    assert payload["order"][0]["iv"] == {"threshold": 0.005, "num_bins": 10}


def test_dotted_key_targets_a_named_step() -> None:
    """In a multi-step config the method has to be named."""
    payload = {"order": [{"null_rate": {"threshold": 0.95}}, {"iv": {"threshold": 0.02}}]}
    _apply_overrides(payload, {"iv.threshold": "0.1"})
    assert payload["order"][0]["null_rate"]["threshold"] == 0.95
    assert payload["order"][1]["iv"]["threshold"] == 0.1


def test_unknown_head_falls_through_to_the_payload() -> None:
    """Keys that name no step address the config's own sections."""
    payload = {"order": [{"iv": {"threshold": 0.02}}], "execution": {"seed": 42}}
    _apply_overrides(payload, {"execution.seed": "7"})
    assert payload["execution"]["seed"] == 7


def test_values_keep_their_yaml_type() -> None:
    """``false`` is a boolean and ``0.02`` a float, not strings."""
    payload = {"order": [{"correlation": {"threshold": 0.95, "tie_break": "null_rate"}}]}
    _apply_overrides(payload, {"threshold": "0.8", "tie_break": "iv"})
    step = payload["order"][0]["correlation"]
    assert step == {"threshold": 0.8, "tie_break": "iv"}


def _row(dataset: str, pipeline: str, seed: int, auc: float) -> dict:
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "seed": seed,
        "status": "ok",
        "n_features_in": 10,
        "n_features_out": 5,
        "seconds": 1.0,
        "peak_rss_gb": 0.1,
        "n_jobs": 1,
        "evaluation": {"auc_test": auc, "auc_valid": auc, "fit_seconds": 0.1},
        "ground_truth": {},
        "error": "",
    }


def test_delta_is_taken_against_the_same_seed() -> None:
    """The split moves with the seed, so a cross-seed baseline would mislead."""
    rows = [
        _row("d", "baseline", 1, 0.90),
        _row("d", "cheap", 1, 0.88),
        _row("d", "baseline", 2, 0.80),
        _row("d", "cheap", 2, 0.79),
    ]
    table = _flat_table(rows).set_index(["pipeline", "seed"])["auc_delta"]
    assert table[("cheap", 1)] == pytest.approx(-0.02)
    assert table[("cheap", 2)] == pytest.approx(-0.01)
    assert table[("baseline", 1)] == pytest.approx(0.0)
