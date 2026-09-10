"""Общие функции работы с типами задач для LightGBM, CatBoost RFE и BorutaSHAP.

``FeatureSchema.task_type`` / ``execution.task_type`` принимает одно из значений:
``binary_classification``, ``classification`` (многоклассовая классификация) или ``regression``.
Для бинарной задачи оценка и SHAP сохраняют прежний расчёт по положительному классу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.schema import TASK_TYPES


@dataclass(frozen=True)
class TaskRuntime:
    """Определённая задача моделирования и закодированный вектор целевой переменной."""

    task_type: str
    n_classes: int | None
    classes: np.ndarray | None
    class_counts: np.ndarray | None
    encoded_target: np.ndarray

    @property
    def is_regression(self: TaskRuntime) -> bool:
        """Возвращает, является ли задача числовой регрессией."""
        return self.task_type == "regression"

    @property
    def is_binary(self: TaskRuntime) -> bool:
        """Возвращает, является ли задача классификацией на два класса."""
        return self.task_type == "binary_classification"

    @property
    def stratify(self: TaskRuntime) -> bool:
        """Возвращает, нужно ли стратифицировать разбиения по целевой переменной."""
        return not self.is_regression


def binary_task() -> TaskRuntime:
    """Возвращает бинарные настройки по умолчанию для функций, формирующих итоговые параметры без меток."""
    return TaskRuntime(
        task_type="binary_classification",
        n_classes=2,
        classes=None,
        class_counts=None,
        encoded_target=np.asarray([], dtype=np.int64),
    )


def resolve_task(
    task_type: str,
    target: Any,
    *,
    method_name: str,
    encode_labels: bool = True,
) -> TaskRuntime:
    """Проверяет метки и кодирует их для заданного ``task_type``.

    Бинарные метки не меняются, чтобы положительный класс LightGBM оставался под
    индексом 1. Многоклассовые метки отображаются в ``0 .. K-1``, если
    ``encode_labels`` равно true (LightGBM / Boruta). CatBoost сохраняет исходные
    метки. Целевые значения регрессии приводятся к ``float64``.
    """
    if task_type not in TASK_TYPES:
        msg = (
            f"{method_name}: unsupported task_type={task_type!r}. "
            f"Expected one of: {sorted(TASK_TYPES)}."
        )
        raise ExecutionError(msg)

    values = np.asarray(target)
    if task_type == "regression":
        try:
            encoded = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            msg = f"{method_name}: regression target must be numeric."
            raise ExecutionError(msg) from exc
        if encoded.size == 0:
            msg = f"{method_name}: target column is empty after sampling."
            raise ExecutionError(msg)
        return TaskRuntime(
            task_type=task_type,
            n_classes=None,
            classes=None,
            class_counts=None,
            encoded_target=encoded,
        )

    classes, class_counts = np.unique(values, return_counts=True)
    if task_type == "binary_classification":
        if len(classes) != 2:
            msg = (
                f"{method_name}: binary classification requires exactly two "
                f"target classes; got {classes.tolist()}."
            )
            raise ExecutionError(msg)
        return TaskRuntime(
            task_type=task_type,
            n_classes=2,
            classes=classes,
            class_counts=class_counts,
            encoded_target=values,
        )

    if len(classes) < 2:
        msg = (
            f"{method_name}: classification requires at least two target "
            f"classes; got {classes.tolist()}."
        )
        raise ExecutionError(msg)
    if not encode_labels:
        return TaskRuntime(
            task_type=task_type,
            n_classes=int(len(classes)),
            classes=classes,
            class_counts=class_counts,
            encoded_target=values,
        )
    try:
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        msg = (
            f"{method_name}: scikit-learn is required to encode multiclass "
            "labels. Install the sklearn optional dependency."
        )
        raise BackendError(msg) from exc
    encoded = LabelEncoder().fit_transform(values)
    return TaskRuntime(
        task_type=task_type,
        n_classes=int(len(classes)),
        classes=classes,
        class_counts=class_counts,
        encoded_target=np.asarray(encoded),
    )


def lgbm_objective_params(
    task: TaskRuntime | str,
    *,
    n_classes: int | None = None,
) -> dict[str, Any]:
    """Возвращает LightGBM ``objective`` / ``metric`` (и ``num_class``) для задачи."""
    task_type, classes = _task_fields(task, n_classes)
    if task_type == "regression":
        return {"objective": "regression", "metric": "rmse"}
    if task_type == "classification":
        if classes is None or int(classes) < 2:
            msg = "lightgbm: multiclass objective requires n_classes >= 2."
            raise ExecutionError(msg)
        return {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": int(classes),
        }
    return {"objective": "binary", "metric": "auc"}


def catboost_loss_params(task: TaskRuntime | str) -> dict[str, Any]:
    """Возвращает CatBoost ``loss_function``, если она отличается от бинарной по умолчанию.

    Бинарный ``CatBoostClassifier`` уже использует Logloss; добавление ключа
    изменило бы прежний словарь параметров.
    """
    task_type, _n_classes = _task_fields(task, None)
    if task_type == "regression":
        return {"loss_function": "RMSE"}
    if task_type == "classification":
        return {"loss_function": "MultiClass"}
    return {}


def optuna_direction(task: TaskRuntime | str) -> str:
    """Возвращает направление оптимизации исследования Optuna для метрики задачи."""
    task_type, _n_classes = _task_fields(task, None)
    if task_type == "regression":
        return "minimize"
    return "maximize"


def make_folds(task: TaskRuntime, *, n_folds: int, seed: int) -> Any:
    """Возвращает объект разбиения ``KFold`` или ``StratifiedKFold`` с перемешиванием."""
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as exc:
        msg = (
            "scikit-learn is required for cross-validation. "
            "Install the sklearn optional dependency."
        )
        raise BackendError(msg) from exc
    if task.is_regression:
        return KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def lgbm_estimator_class(task: TaskRuntime | str) -> Any:
    """Возвращает ``LGBMClassifier`` или ``LGBMRegressor``."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        msg = "LightGBM is required. Install the lightgbm optional dependency."
        raise BackendError(msg) from exc
    task_type, _n_classes = _task_fields(task, None)
    if task_type == "regression":
        return lgb.LGBMRegressor
    return lgb.LGBMClassifier


def catboost_estimator_class(task: TaskRuntime | str) -> Any:
    """Возвращает ``CatBoostClassifier`` или ``CatBoostRegressor``."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        msg = "CatBoost is required. Install the catboost optional dependency."
        raise BackendError(msg) from exc
    task_type, _n_classes = _task_fields(task, None)
    if task_type == "regression":
        return CatBoostRegressor
    return CatBoostClassifier


def rf_estimator_class(task: TaskRuntime | str) -> Any:
    """Возвращает ``RandomForestClassifier`` или ``RandomForestRegressor``."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:
        msg = (
            "scikit-learn RandomForest is required. "
            "Install the sklearn optional dependency."
        )
        raise BackendError(msg) from exc
    task_type, _n_classes = _task_fields(task, None)
    if task_type == "regression":
        return RandomForestRegressor
    return RandomForestClassifier


def score_model(task: TaskRuntime, model: Any, features: Any, target: Any) -> float:
    """Оценивает обученную модель с помощью метрики задачи."""
    labels = np.asarray(target)
    if task.is_regression:
        predictions = np.asarray(model.predict(features)).reshape(-1)
        return _regression_rmse(labels, predictions)
    probabilities = np.asarray(model.predict_proba(features))
    if task.is_binary:
        return _binary_auc(labels, probabilities[:, 1])
    class_order = getattr(model, "classes_", None)
    return _multiclass_auc(labels, probabilities, class_order)


def normalize_binary_shap_values(shap_values: Any) -> np.ndarray:
    """Нормализует выходы поддерживаемых версий SHAP для положительного класса."""
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values
    if hasattr(values, "values"):
        values = values.values

    normalized = np.asarray(values)
    if normalized.ndim == 3:
        normalized = normalized[:, :, 1]
    if normalized.ndim != 2:
        msg = (
            "lightgbm: unsupported SHAP output shape "
            f"{normalized.shape!r}; expected a 2D feature matrix."
        )
        raise ExecutionError(msg)
    return normalized


def shap_mean_abs(task: TaskRuntime | str, shap_values: Any) -> np.ndarray:
    """Возвращает среднее |SHAP| по каждому признаку для текущей задачи.

    Для бинарной задачи сохраняет срез положительного класса. Для многоклассовой усредняет |SHAP| по
    классам, затем по строкам. Для регрессии форма уже равна ``(n_samples, n_features)``.
    """
    task_type, _n_classes = _task_fields(task, None)
    if task_type == "binary_classification":
        matrix = normalize_binary_shap_values(shap_values)
        return np.abs(matrix).mean(axis=0)

    matrix = _shap_row_matrix(shap_values)
    return np.abs(matrix).mean(axis=0)


def _shap_row_matrix(shap_values: Any) -> np.ndarray:
    """Приводит результат SHAP к форме ``(n_samples, n_features)``."""
    if isinstance(shap_values, list):
        stacked = np.stack(
            [_as_shap_array(item) for item in shap_values],
            axis=-1,
        )
        return np.mean(np.abs(stacked), axis=-1)
    values = _as_shap_array(shap_values)
    if values.ndim == 3:
        return np.mean(np.abs(values), axis=-1)
    if values.ndim == 2:
        return values
    msg = (
        "unsupported SHAP output shape "
        f"{values.shape!r}; expected a 2D feature matrix or a class axis."
    )
    raise ExecutionError(msg)


def _as_shap_array(values: Any) -> np.ndarray:
    """Извлекает ``ndarray`` из SHAP Explanation или объекта с интерфейсом массива."""
    raw = values.values if hasattr(values, "values") else values
    return np.asarray(raw)


def _task_fields(
    task: TaskRuntime | str,
    n_classes: int | None,
) -> tuple[str, int | None]:
    """Принимает ``TaskRuntime`` или строку ``task_type``."""
    if isinstance(task, TaskRuntime):
        return task.task_type, task.n_classes if n_classes is None else n_classes
    return str(task), n_classes


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Вычисляет ROC-AUC по оценкам положительного класса."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        msg = (
            "scikit-learn is required for ROC-AUC. "
            "Install the sklearn optional dependency."
        )
        raise BackendError(msg) from exc
    return float(roc_auc_score(labels, scores))


def _multiclass_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_order: Any | None = None,
) -> float:
    """Вычисляет макроусреднённый ROC-AUC «один против остальных» по полной матрице вероятностей."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        msg = (
            "scikit-learn is required for ROC-AUC. "
            "Install the sklearn optional dependency."
        )
        raise BackendError(msg) from exc
    kwargs: dict[str, Any] = {"multi_class": "ovr", "average": "macro"}
    if class_order is not None:
        kwargs["labels"] = np.asarray(class_order)
    return float(roc_auc_score(labels, probabilities, **kwargs))


def _regression_rmse(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Вычисляет корень из среднеквадратичной ошибки."""
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        try:
            from sklearn.metrics import mean_squared_error
        except ImportError as exc:
            msg = (
                "scikit-learn is required for RMSE. "
                "Install the sklearn optional dependency."
            )
            raise BackendError(msg) from exc
        return float(mean_squared_error(labels, predictions, squared=False))
    return float(root_mean_squared_error(labels, predictions))


def require_min_class_count(
    task: TaskRuntime,
    *,
    min_count: int,
    method_name: str,
) -> None:
    """Вызывает ошибку, если в классе задачи классификации меньше ``min_count`` объектов."""
    if task.is_regression or task.class_counts is None:
        return
    if int(task.class_counts.min()) < min_count:
        msg = (
            f"{method_name}: each target class must contain at least "
            f"{min_count} rows after sampling; class counts are "
            f"{task.class_counts.tolist()}."
        )
        raise ExecutionError(msg)
