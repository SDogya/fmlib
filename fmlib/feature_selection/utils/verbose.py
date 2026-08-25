"""Per-method verbose event log for feature selection.

Gated by ``execution.verbose``. Events are sizes, timings and numeric summaries.
Feature-name lists are stripped: those already live in the JSON artifacts written
after each stage.

Lines go to stdout (so Jupyter cells see them even without logging config) and
to the stdlib logger. JSON is written to ``verbose_log.json`` only when
``fit_select(..., output_dir=...)`` is set; otherwise the payload stays on
``SelectionResult.verbose_log``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from fmlib.feature_selection.backends.spark import get_columns

logger = logging.getLogger(__name__)

VERBOSE_LOG_FILENAME = "verbose_log.json"
LINE_PREFIX = "fs.verbose"

# Keys that would dump keep/drop name lists already stored in stage JSONs.
_FEATURE_LIST_KEYS = frozenset(
    {
        "features",
        "dropped",
        "kept",
        "candidates",
        "selected",
        "selected_features",
        "accepted",
        "rejected",
        "tentative",
        "unknown",
        "requested",
        "lgbm_selected",
        "shap_selected",
        "lgbm_dropped",
        "shap_dropped",
        "fold_sets",
    },
)


def default_verbose_recorder() -> "VerboseRecorder":
    """Build a silent recorder for StageContext defaults and unit tests."""
    from fmlib.feature_selection.config import VerboseConfig

    return VerboseRecorder(verbose=VerboseConfig())


def echo(line: str) -> None:
    """Print a line to stdout so notebooks see it without logging setup."""
    print(line, file=sys.stdout, flush=True)  # noqa: T201 - notebooks have no logging config
    logger.info("%s", line)


def announce_saved(path: Path, n_events: int) -> None:
    """Tell the user where the verbose JSON landed."""
    echo(f"{LINE_PREFIX} wrote {path} ({n_events} events)")


def announce_not_saved(n_events: int) -> None:
    """Tell the user why JSON was not written (typical Jupyter call)."""
    echo(
        f"{LINE_PREFIX} {n_events} events kept on result.verbose_log; "
        f"pass output_dir=... to write {VERBOSE_LOG_FILENAME}"
    )


def announce_save_failed(exc: BaseException) -> None:
    """Surface a dump failure on stdout, not only in logging."""
    echo(f"{LINE_PREFIX} failed to write {VERBOSE_LOG_FILENAME}: {exc}")


@dataclass
class VerboseRecorder:
    """In-memory event log gated by per-method ``execution.verbose`` flags."""

    verbose: Any
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    wall_started_at: float = field(default_factory=time.time)

    def enabled(self: VerboseRecorder, method: str) -> bool:
        """Return whether ``method`` should emit events."""
        return bool(getattr(self.verbose, method, False))

    def any_enabled(self: VerboseRecorder) -> bool:
        """Return whether at least one method is verbose."""
        flags = getattr(self.verbose, "__dataclass_fields__", None)
        if flags is None:
            return False
        return any(bool(getattr(self.verbose, name, False)) for name in flags)

    def emit(self: VerboseRecorder, method: str, stage: str, **payload: Any) -> None:
        """Append one event when ``method`` is verbose and echo it to stdout."""
        if not self.enabled(method):
            return
        clean = _sanitize(payload)
        event = {
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 6),
            "method": method,
            "stage": stage,
            **clean,
        }
        self.events.append(event)
        echo(f"{LINE_PREFIX} {method}/{stage} {_compact(clean)}")

    def snapshot_frame(
        self: VerboseRecorder,
        frame: Any,
        *,
        count_rows: bool = False,
    ) -> dict[str, Any]:
        """Describe a DataFrame-like object without Spark actions.

        ``count_rows`` is accepted for call-site compatibility and ignored:
        verbose logging must never call ``count()`` / ``collect()`` / ``toPandas()``.
        """
        del count_rows
        info: dict[str, Any] = {"type": type(frame).__name__}
        n_cols = _n_cols(frame)
        if n_cols is not None:
            info["n_cols"] = n_cols
        n_rows = _n_rows(frame)
        if n_rows is not None:
            info["n_rows"] = n_rows
        memory_mb = _memory_mb(frame)
        if memory_mb is not None:
            info["memory_mb"] = memory_mb
        return info

    def snapshot_datasets(
        self: VerboseRecorder,
        datasets: Mapping[str, Any],
        *,
        count_rows: bool = False,
    ) -> dict[str, Any]:
        """Describe every split in ``datasets`` without Spark actions."""
        del count_rows
        return {
            name: self.snapshot_frame(frame)
            for name, frame in datasets.items()
        }

    def to_dict(self: VerboseRecorder) -> dict[str, Any]:
        """Serialize the recorder for ``verbose_log.json`` and ``result.verbose_log``."""
        verbose_payload: Any
        try:
            verbose_payload = asdict(self.verbose)
        except TypeError:
            verbose_payload = str(self.verbose)
        return {
            "started_at_unix": self.wall_started_at,
            "duration_seconds": round(time.perf_counter() - self.started_at, 6),
            "verbose": verbose_payload,
            "n_events": len(self.events),
            "events": list(self.events),
        }

    def save(self: VerboseRecorder, path: Path) -> Path:
        """Write the verbose log as JSON to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved verbose log to %s (%s events)", path, len(self.events))
        return path


@contextmanager
def verbose_span(
    recorder: VerboseRecorder,
    method: str,
    **start_payload: Any,
) -> Iterator[dict[str, Any] | None]:
    """Emit start/end (or error) events around a method body.

    Yields a mutable dict for end-event fields when verbose, otherwise ``None``.
    """
    if not recorder.enabled(method):
        yield None
        return
    started = time.perf_counter()
    recorder.emit(method, "start", **start_payload)
    extra: dict[str, Any] = {}
    try:
        yield extra
    except Exception as exc:
        recorder.emit(
            method,
            "error",
            duration_seconds=round(time.perf_counter() - started, 6),
            error_type=type(exc).__name__,
            error=str(exc).splitlines()[0][:500],
        )
        raise
    extra.setdefault(
        "duration_seconds",
        round(time.perf_counter() - started, 6),
    )
    recorder.emit(method, "end", **extra)


def _recorder(context: Any) -> VerboseRecorder | None:
    recorder = getattr(context, "verbose_log", None)
    if recorder is None or not hasattr(recorder, "emit") or not hasattr(recorder, "enabled"):
        return None
    return recorder


def emit(context: Any, method: str, stage: str, **payload: Any) -> None:
    """Emit an event from a selector when that method is verbose."""
    recorder = _recorder(context)
    if recorder is None:
        return
    recorder.emit(method, stage, **payload)


def enabled(context: Any, method: str) -> bool:
    """Return whether ``method`` is verbose on ``context``."""
    recorder = _recorder(context)
    if recorder is None:
        return False
    return bool(recorder.enabled(method))


def run_selector_logged(
    selector: Any,
    context: Any,
    candidates: Sequence[str],
) -> list[Any]:
    """Run ``selector.select`` and record timing / size events when verbose."""
    method = selector.method_name
    recorder = _recorder(context)
    if recorder is None or not recorder.enabled(method):
        return selector.select(context, candidates)

    start_payload: dict[str, Any] = {
        "n_candidates_in": len(candidates),
        "datasets": recorder.snapshot_datasets(context.datasets),
        "step_index": getattr(context, "step_index", 0),
    }
    with verbose_span(recorder, method, **start_payload) as span:
        decisions = selector.select(context, candidates)
        if span is not None:
            span.update(_decision_summary(decisions, len(candidates)))
        return decisions


def _decision_summary(decisions: Sequence[Any], n_candidates_in: int) -> dict[str, Any]:
    """Summarize keep/drop decisions without listing feature names."""
    n_drop = 0
    n_keep = 0
    reasons: dict[str, int] = {}
    values: list[float] = []
    for decision in decisions:
        keep = bool(getattr(decision, "keep", False))
        if keep:
            n_keep += 1
        else:
            n_drop += 1
            reason = str(getattr(decision, "reason", None) or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        value = getattr(decision, "value", None)
        if isinstance(value, (int, float)) and value is not None:
            values.append(float(value))
    summary: dict[str, Any] = {
        "n_candidates_in": n_candidates_in,
        "n_candidates_out": n_candidates_in - n_drop,
        "n_dropped": n_drop,
        "n_keep_decisions": n_keep,
        "n_decisions": len(decisions),
        "drop_reasons": reasons,
    }
    if values:
        summary["value_min"] = min(values)
        summary["value_max"] = max(values)
        summary["value_mean"] = round(sum(values) / len(values), 6)
    return summary


def _sanitize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop feature-name lists and coerce values to JSON-friendly types."""
    return {
        key: _jsonable(value)
        for key, value in payload.items()
        if key not in _FEATURE_LIST_KEYS
    }


def _jsonable(value: Any) -> Any:
    """Convert a value to a JSON-serializable form."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if str(key) not in _FEATURE_LIST_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _compact(payload: Mapping[str, Any]) -> str:
    """Render a short logger-friendly summary of an event payload."""
    parts: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            parts.append(f"{key}={{{len(value)}}}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)}]")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _n_cols(frame: Any) -> int | None:
    """Return the number of columns when cheaply available."""
    try:
        return len(get_columns(frame))
    except Exception:  # noqa: BLE001 - verbose logging must never fail the run
        columns = getattr(frame, "columns", None)
        if columns is None:
            shape = getattr(frame, "shape", None)
            if shape is not None:
                try:
                    return int(shape[1])
                except (TypeError, ValueError, IndexError):
                    return None
            return None
        try:
            return len(list(columns))
        except TypeError:
            return None


def _n_rows(frame: Any) -> int | None:
    """Return a cheap local row count. Never runs Spark ``count()``."""
    if type(frame).__module__.startswith("pyspark"):
        return None
    shape = getattr(frame, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a core test/runtime dep
        pd = None  # type: ignore[assignment]
    if pd is not None and isinstance(frame, pd.DataFrame):
        return len(frame)
    return None


def _memory_mb(frame: Any) -> float | None:
    """Return pandas/numpy memory usage in MiB when available."""
    memory_usage = getattr(frame, "memory_usage", None)
    if callable(memory_usage):
        try:
            used = memory_usage(deep=True)
            total = used.sum() if hasattr(used, "sum") else used
            return round(float(total) / (1024 * 1024), 4)
        except Exception:  # noqa: BLE001 - optional diagnostic
            return None
    nbytes = getattr(frame, "nbytes", None)
    if isinstance(nbytes, (int, float)):
        return round(float(nbytes) / (1024 * 1024), 4)
    return None
