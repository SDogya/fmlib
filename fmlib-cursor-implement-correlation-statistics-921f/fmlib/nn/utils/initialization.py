import contextlib
from copy import deepcopy
from typing import Callable, Sequence, cast

import torch

InitFn = Callable[[torch.nn.Module], torch.nn.Module]


def xavier_initialization(module: torch.nn.Module) -> None:
    """
    Инициализация параметров всех модулей методом Xavier normal.

    Args:
        module (torch.nn.Module): Модуль для инициализации.

    """
    for _, param in module.named_parameters():
        with contextlib.suppress(ValueError):
            torch.nn.init.xavier_normal_(param.data)
    return module


def nba_init_weights(module: torch.nn.Module) -> None:
    """
    Инициализация весов, консистентно с nba.

    Аргументы:
        module (torch.nn.Module): Модуль для инициализации.

    *Примечание:* Предлагается использовать следующим образом:
    ```
    def __init__(self, ...) -> None:
        ...
        self.apply(nba_init_weights)
    ```
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


def nba_initalization(module: torch.nn.Module) -> torch.nn.Module:
    """
    Инициализация модулей, консистентно с NBA моделями: Ivan, Trivan, DLT.

    Аргументы:
        module (torch.nn.Module): Модуль для инициализации.

    *Примечание:* Предлагается использовать следующим образом:
    ```
    nba_initialization(model)
    ```
    """
    return module.apply(nba_init_weights)


INIT2FN: dict[str, InitFn] = {
    "xavier": xavier_initialization,
    "nba": nba_initalization,
    "none": lambda x: x,
}


def clone_module_to_modulelist(module: torch.nn.Module, copy_count: int, init: str | InitFn = "xavier") -> torch.nn.ModuleList:
    """
    Создаёт и возвращает список (ModuleList) из нескольких копий заданного модуля.

    Каждая копия модуля инициализируется с использованием заданной функции.

    Аргументы:
        module (torch.nn.Module): Базовый модуль, который будет клонироваться.
        copy_count (int): Количество копий модуля, которые нужно создать.
        init (str | InitFn): Инициализация для клонированных модулей. По умолчанию - "xavier".

    Returns:
        torch.nn.ModuleList: Список зарегистрированных модулей, готовых для использования в модели.
    """
    if isinstance(init, str):
        init = INIT2FN[init]
    init_casted: InitFn = cast(InitFn, init)

    copies: list[torch.nn.Module] = []
    for _ in range(copy_count):
        module_copy: torch.nn.Module = deepcopy(module)
        copies.append(init_casted(module_copy))
    return torch.nn.ModuleList(copies)


def clone_module_to_moduledict(
    module: torch.nn.Module, layer_names: Sequence[str], init: str | InitFn = "xavier"
) -> torch.nn.ModuleDict:
    """
    Создаёт и возвращает словарь (ModuleDict) из нескольких копий заданного модуля.

    Каждая копия модуля инициализируется с использованием заданной функции.
    Копии модулей сохраняются под ключами, указанными в `layer_names`.

    Аргументы:
        module (torch.nn.Module): Базовый модуль, который будет клонироваться.
        layer_names (Sequence[str]): Последовательность строк — имена для ключей в ModuleDict.
        init (str | InitFn): Инициализация для клонированных модулей. По умолчанию - "xavier".

    Returns:
        torch.nn.ModuleDict: Словарь зарегистрированных модулей, доступных по ключам.
    """
    if isinstance(init, str):
        init = INIT2FN[init]
    init_casted: InitFn = cast(InitFn, init)

    copies: dict[torch.nn.Module] = {}
    for name in layer_names:
        module_copy: torch.nn.Module = deepcopy(module)
        copies[name] = init_casted(module_copy)
    return torch.nn.ModuleDict(copies)
