"""JSON cache of statistical metrics computed on the full candidate set.

Thresholds are applied later, so a change of ``order`` or of a drop threshold
reuses the same numbers. Compute-parameter changes (for example
``low_variance.scale_method``) get their own entry beside the old one.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from fmlib.feature_selection.exceptions import ConfigError

FORMAT_VERSION = 1
DEFAULT_CACHE_FILENAME = "statistics_metrics.json"
CACHEABLE_METHODS = frozenset(
    {
        "null_rate",
        "constants",
        "low_variance",
        "correlation",
        "psi",
        "iv",
    },
)


def dataset_fingerprint(
    datasets: Mapping[str, Any],
    schema: Any,
    *,
    dataset_id: Optional[str] = None,
) -> dict[str, Any]:
    """Identify the data a cached metric was measured on.

    Without this the key is made of method parameters only, so ``null_rate``
    -- whose key is otherwise empty -- reuses numbers measured on a different
    table. The parts are cheap and change whenever the numbers would: the
    candidate list, the target definition, and the shape of every split (which
    also catches an upstream ``row_sample``).

    ``dataset_id`` is the caller's own name for the data. Pass it whenever the
    same logical dataset can arrive with a different row count (an incremental
    load, a re-partition) and the metrics should still be reused.

    Args:
        datasets: Split name to DataFrame mapping, as resolved by the pipeline.
        schema: Validated feature schema.
        dataset_id: Optional stable identifier supplied by the caller.

    Returns:
        JSON-friendly mapping mixed into every per-method fingerprint.
    """
    candidates = list(schema.candidate_features())
    payload: dict[str, Any] = {
        "n_candidates": len(candidates),
        "candidates_sha256": hashlib.sha256(
            "\n".join(candidates).encode("utf-8"),
        ).hexdigest()[:16],
        "target": schema.target,
        "task_type": schema.task_type,
        "splits": {
            name: _frame_shape(datasets[name]) for name in sorted(datasets)
        },
    }
    if dataset_id is not None and str(dataset_id).strip():
        payload["dataset_id"] = str(dataset_id)
    return payload


def _frame_shape(frame: Any) -> dict[str, Any]:
    """Return ``{n_rows, n_cols}`` for a pandas or Spark frame."""
    columns = getattr(frame, "columns", None)
    n_cols = len(list(columns)) if columns is not None else None
    try:
        n_rows = int(len(frame))
    except TypeError:
        try:
            n_rows = int(frame.count())
        except Exception:  # noqa: BLE001 - unknown backend; shape stays partial
            n_rows = None
    return {"n_rows": n_rows, "n_cols": n_cols}


def compute_fingerprint(
    method: str,
    settings: Any,
    *,
    max_local_rows: int,
    seed: int | None = None,
    task_type: str | None = None,
    data: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the cache key for one statistics method.

    Thresholds are omitted: they are applied later. Only values that change
    the stored numbers belong here -- the method's compute parameters and,
    via ``data``, the dataset those numbers were measured on.
    """
    method_key = _method_fingerprint(
        method,
        settings,
        max_local_rows=max_local_rows,
        seed=seed,
        task_type=task_type,
    )
    if data is None:
        return method_key
    return {"data": dict(data), **method_key}


def _method_fingerprint(
    method: str,
    settings: Any,
    *,
    max_local_rows: int,
    seed: int | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Compute-parameter part of the cache key for one statistics method."""
    if method == "null_rate":
        return {}
    if method == "constants":
        min_unique = getattr(settings, "min_unique", None)
        if min_unique is None:
            return {}
        return {"min_unique": int(min_unique)}
    if method == "low_variance":
        return {"scale_method": str(getattr(settings, "scale_method", "robust"))}
    if method == "correlation":
        max_rows = min(int(getattr(settings, "max_rows", 100_000)), int(max_local_rows))
        return {
            "method": str(getattr(settings, "method", "pearson")),
            "max_rows": max_rows,
            "seed": seed,
            "stratified": str(task_type or "") != "regression",
        }
    if method == "psi":
        fingerprint: dict[str, Any] = {
            "mode": str(getattr(settings, "mode", "train_valid")),
            "num_bins": int(getattr(settings, "num_bins", 10)),
            "subsample_rows": getattr(settings, "subsample_rows", None),
            "max_levels": getattr(settings, "max_levels", None),
            "relative_error": float(getattr(settings, "relative_error", 0.001)),
            "batch_size": int(getattr(settings, "batch_size", 100)),
            "month_column": str(getattr(settings, "month_column", "month_part")),
            "test_months": int(getattr(settings, "test_months", 1)),
            "n_jobs": int(getattr(settings, "n_jobs", -1)),
        }
        if fingerprint["subsample_rows"] is not None:
            fingerprint["seed"] = seed
        return fingerprint
    if method == "iv":
        return {
            "num_bins": int(getattr(settings, "num_bins", 10)),
            "eps": float(getattr(settings, "eps", 1e-4)),
            "min_bin_share": float(getattr(settings, "min_bin_share", 0.0)),
            "max_levels": getattr(settings, "max_levels", None),
            "relative_error": float(getattr(settings, "relative_error", 0.001)),
            "batch_size": int(getattr(settings, "batch_size", 50)),
        }
    return {}


def resolve_cache_path(path: Optional[Union[str, Path]]) -> Path:
    """Resolve the cache file path. ``None`` means ``statistics_metrics.json`` in cwd."""
    if path is None or not str(path).strip():
        return Path.cwd() / DEFAULT_CACHE_FILENAME
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def fingerprint_json(fingerprint: Mapping[str, Any]) -> str:
    """Canonical JSON for an entry lookup key."""
    return json.dumps(_jsonable(dict(fingerprint)), sort_keys=True, separators=(",", ":"))


class StatisticsMetricsCache:
    """On-disk list of ``{method, fingerprint, metrics}`` entries."""

    def __init__(
        self: StatisticsMetricsCache,
        path: Path,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.data_fingerprint: Optional[dict[str, Any]] = None
        if payload is None:
            self._entries: list[dict[str, Any]] = []
        else:
            self._entries = [dict(entry) for entry in payload.get("entries", [])]

    @classmethod
    def load(
        cls: type[StatisticsMetricsCache],
        path: Union[str, Path],
        *,
        force_recompute: bool = False,
    ) -> StatisticsMetricsCache:
        """Read an existing file, or start empty.

        A corrupt file is an error unless ``force_recompute`` is set, in which
        case the cache starts empty and will be rewritten on the first upsert.
        """
        file_path = Path(path)
        if not file_path.exists():
            return cls(file_path)
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if force_recompute:
                return cls(file_path)
            msg = (
                f"statistics.cache: failed to read {str(file_path)!r}: {exc}. "
                "Fix the file or set statistics.cache.force_recompute: true."
            )
            raise ConfigError(msg) from exc
        if not isinstance(raw, Mapping):
            if force_recompute:
                return cls(file_path)
            msg = f"statistics.cache: {str(file_path)!r} root must be a mapping."
            raise ConfigError(msg)
        version = raw.get("format_version")
        if version != FORMAT_VERSION:
            if force_recompute:
                return cls(file_path)
            msg = (
                f"statistics.cache: {str(file_path)!r} has format_version="
                f"{version!r}, expected {FORMAT_VERSION}."
            )
            raise ConfigError(msg)
        entries = raw.get("entries", [])
        if not isinstance(entries, list):
            if force_recompute:
                return cls(file_path)
            msg = f"statistics.cache: {str(file_path)!r} 'entries' must be a list."
            raise ConfigError(msg)
        return cls(file_path, {"entries": entries})

    def lookup(
        self: StatisticsMetricsCache,
        method: str,
        fingerprint: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return cached metrics for ``method`` + fingerprint, or ``None``."""
        key = fingerprint_json(fingerprint)
        for entry in self._entries:
            if entry.get("method") != method:
                continue
            stored = entry.get("fingerprint", {})
            if not isinstance(stored, Mapping):
                continue
            if fingerprint_json(stored) == key:
                metrics = entry.get("metrics")
                if isinstance(metrics, Mapping):
                    return dict(metrics)
                return None
        return None

    def upsert(
        self: StatisticsMetricsCache,
        method: str,
        fingerprint: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> None:
        """Replace or append one entry and save immediately."""
        key = fingerprint_json(fingerprint)
        record = {
            "method": method,
            "fingerprint": _jsonable(dict(fingerprint)),
            "metrics": _jsonable(dict(metrics)),
        }
        replaced = False
        updated: list[dict[str, Any]] = []
        for entry in self._entries:
            stored = entry.get("fingerprint", {})
            if (
                entry.get("method") == method
                and isinstance(stored, Mapping)
                and fingerprint_json(stored) == key
            ):
                updated.append(record)
                replaced = True
            else:
                updated.append(entry)
        if not replaced:
            updated.append(record)
        self._entries = updated
        self.save()

    def save(self: StatisticsMetricsCache) -> None:
        """Write the cache file, creating parent directories as needed."""
        payload = {
            "format_version": FORMAT_VERSION,
            "entries": self._entries,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _jsonable(value: Any) -> Any:
    """Convert numpy / non-finite numbers into JSON-friendly values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    try:
        import numpy as np
    except ImportError:
        return value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    return value
