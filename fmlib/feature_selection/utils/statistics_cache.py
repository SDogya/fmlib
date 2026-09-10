"""JSON-кэш статистических метрик для кандидатов, оставшихся перед текущим шагом.

Пороги применяются позже и могут меняться без пересчёта, если данные, параметры
вычисления и упорядоченный список кандидатов совпадают. Изменения любого из этих
компонентов получают отдельную запись рядом с прежней.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from fmlib.feature_selection.exceptions import ConfigError

FORMAT_VERSION = 2
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


def compute_fingerprint(
    method: str,
    settings: Any,
    *,
    max_local_rows: int,
    seed: int | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Формирует часть ключа кэша, описывающую параметры статистического метода.

    Пороги не включаются: они применяются позже. Здесь нужны только значения, влияющие на
    сохраняемые результаты вычислений. Исполнитель добавляет fingerprint данных
    из ``compute_data_fingerprint`` перед поиском и сохранением записи.
    """
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
            "min_bin_share": float(getattr(settings, "min_bin_share", 0.0)),
            "max_levels": getattr(settings, "max_levels", 50),
            "mode": str(getattr(settings, "mode", "train_valid")),
            "num_bins": int(getattr(settings, "num_bins", 10)),
            "subsample_rows": getattr(settings, "subsample_rows", None),
            "eps": float(getattr(settings, "eps", 1e-4)),
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


def compute_data_fingerprint(context: Any) -> dict[str, Any]:
    """Описывает snapshot данных, текущую схему и выполненные преобразования строк.

    Для pandas дополнительно хешируется содержимое с сохранением порядка строк.
    Для Spark читается только схема: актуальность snapshot гарантирует пользователь
    через ``dataset_version``. Полный обход Spark ради ключа не выполняется.
    Внешний test не читается и не влияет на статистики.
    """
    import pandas as pd

    version = context.config.statistics.cache.dataset_version
    if not isinstance(version, str) or not version.strip():
        msg = "statistics.cache requires a non-empty dataset_version."
        raise ConfigError(msg)
    splits = {}
    for name in ("train", "valid"):
        frame = context.datasets.get(name)
        if frame is None:
            continue
        if isinstance(frame, pd.DataFrame):
            try:
                rows = pd.util.hash_pandas_object(frame, index=True, categorize=True)
            except (TypeError, ValueError) as exc:
                msg = f"statistics.cache: cannot fingerprint pandas split {name!r}: {exc}. Disable the cache."
                raise ConfigError(msg) from exc
            splits[name] = {
                "backend": "pandas",
                "columns": [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()],
                "rows": len(frame),
                "content": hashlib.sha256(rows.to_numpy().tobytes()).hexdigest(),
            }
        elif type(frame).__module__.startswith("pyspark"):
            splits[name] = {"backend": "spark", "schema": frame.schema.jsonValue()}
        else:
            msg = f"statistics.cache: unsupported split type {type(frame)!r}. Disable the cache."
            raise ConfigError(msg)
    return {
        "dataset_version": version,
        "schema": asdict(context.schema),
        "splits": splits,
        "row_transforms": list(context.statistics_row_transforms),
    }


def resolve_cache_path(
    path: Optional[Union[str, Path]],
    *,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Определяет явный путь или файл внутри каталога результатов; общего пути по умолчанию нет."""
    if path is None or not str(path).strip():
        if output_dir is None:
            msg = "statistics.cache requires an explicit path or pipeline output_dir."
            raise ConfigError(msg)
        return Path(output_dir).expanduser().resolve() / DEFAULT_CACHE_FILENAME
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def fingerprint_json(fingerprint: Mapping[str, Any]) -> str:
    """Возвращает канонический JSON для ключа поиска записи."""
    return json.dumps(_jsonable(dict(fingerprint)), sort_keys=True, separators=(",", ":"))


class StatisticsMetricsCache:
    """Дисковый список записей ``{method, fingerprint, metrics}``."""

    def __init__(
        self: StatisticsMetricsCache,
        path: Path,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
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
        """Читает существующий файл или создаёт пустой кэш.

        Повреждённый файл вызывает ошибку, если не задан ``force_recompute``; при заданном флаге
        кэш изначально пуст и будет перезаписан при первом upsert.
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
                f"{version!r}, expected {FORMAT_VERSION}. "
                "Use a new cache path or set statistics.cache.force_recompute: true."
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
        """Возвращает метрики для ``method`` и полного ключа кеша или ``None``."""
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
        """Заменяет или добавляет одну запись и сразу сохраняет её."""
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
        """Записывает файл кэша, при необходимости создавая родительские каталоги."""
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
    """Преобразует числа numpy и неконечные числа в значения, совместимые с JSON."""
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
