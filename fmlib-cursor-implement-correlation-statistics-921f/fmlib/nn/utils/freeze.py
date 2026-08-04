import torch


def set_requires_grad(model: torch.nn.Module, requires_grad: bool = False) -> torch.nn.Module:
    """Функция, для установки флага requires_grad для всех параметров модели."""
    model.requires_grad_(requires_grad)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad != requires_grad:
            msg: str = f'Parameter "{name}" is not freezed/unfreezed: {parameter.requires_grad=} vs. {requires_grad=}.'
            raise ValueError(msg)

    return model


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Замораживает модель, т.е. ставит флаг requires_grad=False для всех параметров модели."""
    return set_requires_grad(model, requires_grad=False)


def unfreeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Размораживает модель, т.е. ставит флаг requires_grad=True для всех параметров модели."""
    return set_requires_grad(model, requires_grad=True)
