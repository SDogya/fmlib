"""Исключения модуля отбора признаков."""


class FeatureSelectionError(Exception):
    """Базовая ошибка пайплайна отбора признаков.

    Args:
        message: Понятное описание, по возможности с указанием способа исправления.
    """

    def __init__(self: "FeatureSelectionError", message: str) -> None:
        super().__init__(message)
        self.message = message


class SchemaError(FeatureSelectionError):
    """Возникает при некорректной FeatureSchema или недопустимых входных столбцах."""


class ConfigError(FeatureSelectionError):
    """Возникает при некорректной или несовместимой FeatureSelectionConfig."""


class CapacityError(FeatureSelectionError):
    """Возникает при превышении лимитов загрузки данных в локальную память."""


class BackendError(FeatureSelectionError):
    """Возникает, если требуемый бэкенд или необязательная зависимость недоступны."""


class ExecutionError(FeatureSelectionError):
    """Возникает при сбое во время выполнения этапа."""
