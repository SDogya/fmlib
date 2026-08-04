from typing import Dict, List, Self, cast

import torch

from fmlib.constants.batches import Batch
from fmlib.training.training_types import MetricsType, MetricsValue


class CombinedLoss(torch.nn.Module):
    """
    Множество лоссов, сомещенный в один модуль.

    Аргументы:
        losses (torch.nn.ModuleDict): Словарь из имён лоссов и их модулей.
        coeffs (Dict[str, float] | None, optional): Словарь из имён лоссов и их коэффициентов.
            По умолчанию - `None`
        track_window (int): Размер окна подсчетов статистики.
            По умолчанию - 128

    Примечание: если в словаре `coeffs` не указан коэффициент для лосса, то он равен 1.
    """

    def __init__(
        self: Self, 
        losses: Dict[str, torch.nn.Module],
        coeffs: Dict[str, float] | None = None,
        track_window: int = 128,
        epoch_reset: bool = True
    ) -> None:
        super().__init__()

        if coeffs is None:
            coeffs = {}

        if track_window < 1:
            msg: str = f"Track window must be positive, but got {track_window=}."
            raise ValueError(msg)

        self.track_window: int = track_window
        self.stats: List[MetricsType] = []
        self.coeffs: Dict[str, float] = dict(coeffs)
        self.losses: torch.nn.ModuleDict = torch.nn.ModuleDict(losses)
        self.epoch_reset: bool = epoch_reset

    @torch.jit.export
    def reset_stats(self: Self) -> Self:
        self.stats = []
        return self

    @torch.jit.export
    def update_stats(self: Self, stats: MetricsType) -> None:
        if self.track_window <= len(self.stats):
            self.stats.pop(0)
        self.stats.append(stats)
        assert len(self.stats) <= self.track_window

    @torch.jit.export
    def get_stats(self: Self) -> MetricsType:
        result: MetricsType = {"stats_length": len(self.stats)}
        if len(self.stats) > 0:
            keys: List[str] = list(self.stats[-1].keys())
            for key in sorted(keys):
                values: List[MetricsValue] = [s[key] for s in self.stats]
                values_tensor: torch.Tensor = torch.asarray(values).ravel()
                result[f"{key}_median"] = values_tensor.median().item()
                result[f"{key}_mean"] = values_tensor.mean().item()
                result[f"{key}_max"] = values_tensor.max().item()
                result[f"{key}_min"] = values_tensor.min().item()
                result[f"{key}_last"] = values[-1]
        return result

    def get_subloss_stats(self: Self, name: str, loss: torch.nn.Module) -> MetricsType:
        def detach_if_needed(value: torch.Tensor | MetricsValue) -> torch.Tensor:
            if torch.is_tensor(value):
                return value.cpu().detach().item()
            else:
                return value

        result: MetricsType = {}
        if hasattr(loss, "get_stats"):
            raw_stats: MetricsType = loss.get_stats()
            result = {f"{name}_{key}": detach_if_needed(value) for key, value in raw_stats.items()}

        return result

    def forward(self: Self, outputs: Batch, targets: Batch) -> torch.Tensor:
        new_stats = {}

        result: torch.Tensor | None = None

        for name, loss in self.losses.items():
            coeff: float = self.coeffs.get(name, 1.0)
            value: torch.Tensor = loss(outputs, targets)

            new_stats[name] = value.detach().cpu().item()
            subloss_stats: MetricsType = self.get_subloss_stats(name, loss)
            new_stats.update(subloss_stats)

            assert torch.is_tensor(value)
            assert torch.numel(value) == 1

            weighted: torch.Tensor = coeff * value
            result = weighted if result is None else result + weighted

        assert result is not None
        assert torch.is_tensor(result)
        assert torch.numel(result) == 1

        self.update_stats(new_stats)

        return cast(torch.Tensor, result)
