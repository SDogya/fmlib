"""Фасад пайплайна отбора признаков."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from fmlib.feature_selection.backends.spark import (
    ensure_dataframe,
    ensure_spark_session,
    get_columns,
)
from fmlib.feature_selection.base import StageContext, bind_process_rng
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
from fmlib.feature_selection.result import (
    DroppedFeature,
    SelectionResult,
    _next_available_path,
    stage_backends_for_config,
)
from fmlib.feature_selection.runner import run_order, validate_order_prerequisites
from fmlib.feature_selection.schema import FeatureSchema, ensure_no_feature_leak
from fmlib.feature_selection.utils.verbose import (
    VERBOSE_LOG_FILENAME,
    VerboseRecorder,
    announce_not_saved,
    announce_save_failed,
    announce_saved,
)

_DATASET_KEYS = ("train", "valid", "test")
logger = logging.getLogger(__name__)


class FeatureSelectionPipeline:
    """Настраиваемый пайплайн отбора признаков.

    Шаги выполняются в порядке ``config.order``. Каждый шаг может относиться к любому этапу
    (preprocessing, statistics, model); повторения допустимы.

    Args:
        config: Проверенная конфигурация пайплайна.
    """

    def __init__(self: FeatureSelectionPipeline, config: FeatureSelectionConfig) -> None:
        config.validate()
        self.config = config

    def fit_select(
        self: FeatureSelectionPipeline,
        spark: Any,
        *,
        schema: FeatureSchema,
        data: Any = None,
        datasets: Optional[Mapping[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> SelectionResult:
        """Выполняет шаги отбора и возвращает артефакт.

        Принимает либо один DataFrame ``data`` (с ``schema.split``), либо
        словарь ``datasets`` с обязательным ключом ``train``. Входные данные не изменяются.

        Args:
            spark: Активная сессия Spark (в каркасе используется утиная типизация).
            schema: Описание ролей признаков.
            data: Один Spark DataFrame с необязательным столбцом разбиения.
            datasets: Словарь DataFrame для ``train`` / ``valid`` / ``test``.
            output_dir: Необязательный каталог для промежуточных артефактов с защитой от коллизий имён
                и ``final_results.json``.

        Returns:
            ``SelectionResult`` с отобранными и исключёнными признаками и метаданными.

        Raises:
            ConfigError: При взаимоисключающих или неполных вариантах входных данных.
            SchemaError: При некорректной схеме или отсутствии столбцов.
            BackendError: Если ``spark`` или DataFrame непригодны для работы.
        """
        ensure_spark_session(spark)
        resolved, datasets_mode = self._resolve_datasets(
            data=data,
            datasets=datasets,
            schema=schema,
        )
        self._validate_inputs(resolved, schema=schema, datasets_mode=datasets_mode)

        recorder = VerboseRecorder(self.config.execution.verbose)
        pipeline_started = time.perf_counter()
        if recorder.enabled("pipeline"):
            recorder.emit(
                "pipeline",
                "start",
                datasets_mode=datasets_mode,
                seed=self.config.execution.seed,
                n_categorical=len(schema.categorical),
                n_continuous=len(schema.continuous),
                n_candidates=len(schema.candidate_features()),
                order_methods=[step.method for step in self.config.order],
                dataset_splits=list(resolved),
                datasets=recorder.snapshot_datasets(resolved),
                output_dir=str(output_dir) if output_dir is not None else None,
            )

        candidates = schema.candidate_features()
        ensure_no_feature_leak(candidates, schema)
        if not candidates:
            msg = (
                "FeatureSchema has no candidate features. Provide non-empty "
                "categorical and/or continuous lists."
            )
            raise SchemaError(msg)

        seed = self.config.execution.seed
        context = StageContext(
            spark=spark,
            datasets=resolved,
            schema=schema,
            config=self.config,
            seed=seed,
            candidates=list(candidates),
            datasets_mode=datasets_mode,
            verbose_log=recorder,
            run_seed=seed,
            output_dir=Path(output_dir) if output_dir is not None else None,
        )
        bind_process_rng(seed)

        validate_order_prerequisites(context)

        output_path = context.output_dir
        remaining = list(candidates)
        try:
            remaining, _ = run_order(context, remaining)
            schema = context.schema

            dropped = [
                DroppedFeature(
                    feature=item.feature,
                    stage=item.stage,
                    method=item.method,
                    reason=item.reason,
                    value=item.value,
                    threshold=item.threshold,
                )
                for item in context.decisions
                if not item.keep
            ]

            warnings: list[str] = []

            if recorder.enabled("pipeline"):
                recorder.emit(
                    "pipeline",
                    "end",
                    duration_seconds=round(time.perf_counter() - pipeline_started, 6),
                    n_selected=len(remaining),
                    n_dropped=len(dropped),
                    n_warnings=len(warnings),
                    datasets_mode=datasets_mode,
                    n_steps=context.step_index,
                )
            result = SelectionResult(
                selected_features=list(remaining),
                dropped_features=dropped,
                schema=schema,
                config=self.config.to_dict(),
                seed=seed,
                stage_backends=stage_backends_for_config(self.config),
                warnings=warnings,
                datasets_mode=datasets_mode,
                scores=dict(context.scores),
                verbose_log=recorder.to_dict() if recorder.any_enabled() else None,
            )
            if output_path is not None:
                result.save(output_path / "final_results.json")
            return result
        except Exception as exc:
            if recorder.enabled("pipeline"):
                recorder.emit(
                    "pipeline",
                    "error",
                    duration_seconds=round(time.perf_counter() - pipeline_started, 6),
                    error_type=type(exc).__name__,
                    error=str(exc).splitlines()[0][:500],
                )
            raise
        finally:
            if recorder.any_enabled() and output_path is None:
                announce_not_saved(len(recorder.events))
            elif recorder.any_enabled() and output_path is not None:
                try:
                    saved = recorder.save(
                        _next_available_path(output_path / VERBOSE_LOG_FILENAME),
                    )
                    announce_saved(saved, len(recorder.events))
                except Exception as save_exc:
                    logger.exception("Failed to write feature-selection %s", VERBOSE_LOG_FILENAME)
                    announce_save_failed(save_exc)

    def transform(self: FeatureSelectionPipeline, data: Any, result: SelectionResult) -> Any:
        """Применяет полученный при отборе ``SelectionResult`` к объекту с интерфейсом DataFrame.

        Args:
            data: Входной объект с интерфейсом DataFrame.
            result: Ранее полученный артефакт отбора.

        Returns:
            Проекция на отобранные и служебные столбцы.
        """
        return result.apply(data)


    @staticmethod
    def _resolve_datasets(
        *,
        data: Any,
        datasets: Optional[Mapping[str, Any]],
        schema: FeatureSchema,
    ) -> tuple[dict[str, Any], str]:
        if data is not None and datasets is not None:
            msg = (
                "Pass either data=... or datasets=..., not both. "
                "Use a single DataFrame with FeatureSchema.split, or a train/valid/test mapping."
            )
            raise ConfigError(
                msg,
            )
        if data is None and datasets is None:
            msg = "One of data=... or datasets=... is required."
            raise ConfigError(
                msg,
            )

        if data is not None:
            ensure_dataframe(data, name="data")
            if schema.split is None:
                msg = (
                    "FeatureSchema.split is required when passing a single data DataFrame. "
                    "Set split or use datasets={'train': ...}."
                )
                raise SchemaError(
                    msg,
                )
            return {"train": data}, "single"

        assert datasets is not None
        unknown = sorted(set(datasets) - set(_DATASET_KEYS))
        if unknown:
            msg = f"Unsupported datasets keys: {unknown}. Allowed: {list(_DATASET_KEYS)}."
            raise ConfigError(
                msg,
            )
        if "train" not in datasets:
            msg = "datasets must contain a 'train' DataFrame."
            raise ConfigError(msg)
        resolved: dict[str, Any] = {}
        for key in _DATASET_KEYS:
            if key in datasets:
                resolved[key] = ensure_dataframe(datasets[key], name=f"datasets[{key!r}]")
        return resolved, "mapping"

    @staticmethod
    def _validate_inputs(
        datasets: Mapping[str, Any],
        *,
        schema: FeatureSchema,
        datasets_mode: str,
    ) -> None:
        require_split = datasets_mode == "single"
        reference_columns = get_columns(datasets["train"])
        schema.validate_against_columns(reference_columns, require_split=require_split)

        for name, frame in datasets.items():
            columns = set(get_columns(frame))
            missing = [col for col in schema.all_declared_columns() if col not in columns]

            # split column is only required on the single-DF path
            if datasets_mode == "mapping":
                missing = [col for col in missing if col != schema.split]

            if missing:
                msg = f"datasets[{name!r}] is missing declared columns: {missing}. Align all splits to the same feature schema."
                raise SchemaError(
                    msg,
                )
