import time
from typing import Any, Dict, List, Self

import torch

from fmlib.training.training_types import MetricsType, TrainingState


def _compute_grad_norm(optimizer: torch.optim.Optimizer) -> float:
    total_norm = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


class TrainingStatsCallback:
    """
    Коллбек для сбора статистик тренировочного процесса за эпоху:
        - mean_epoch_lr: среднее значение learning rate по шагам эпохи
        - grad_norm: средняя норма градиента по шагам эпохи
        - epoch_time: среднее время одного шага за эпоху

    Норма градиента вычисляется через post-step hook на внутреннем оптимизаторе —
    после optimizer.step(), но до optimizer.zero_grad(), поэтому градиенты ещё доступны.
    Требует PyTorch >= 2.0 для register_step_post_hook; при отсутствии — grad_norm не логируется.
    """

    def __init__(self: Self) -> None:
        self._lr_values: List[float] = []
        self._grad_norm_values: List[float] = []
        self._step_times: List[float] = []
        self._step_start_time: float | None = None
        self._last_grad_norm: float | None = None
        self._hook_handle: Any = None

    def _register_grad_norm_hook(self: Self, optimizer: torch.optim.Optimizer) -> None:
        inner = getattr(optimizer, "optimizer", optimizer)
        if not hasattr(inner, "register_step_post_hook"):
            return

        def _hook(opt: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
            self._last_grad_norm = _compute_grad_norm(opt)

        self._hook_handle = inner.register_step_post_hook(_hook)

    def reset(self: Self) -> Self:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._lr_values = []
        self._grad_norm_values = []
        self._step_times = []
        self._last_grad_norm = None
        self._step_start_time = time.perf_counter()
        return self

    def __call__(
        self: Self,
        state: TrainingState,
        transformed: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        epoch: int = 0,
        external: MetricsType | None = None,
    ) -> MetricsType:
        optimizer = state["optimizer"]

        if self._hook_handle is None:
            self._register_grad_norm_hook(optimizer)

        now = time.perf_counter()
        if self._step_start_time is not None:
            self._step_times.append(now - self._step_start_time)
        self._step_start_time = now

        lrs = [group["lr"] for group in optimizer.param_groups]
        self._lr_values.append(sum(lrs) / len(lrs))

        if self._last_grad_norm is not None:
            self._grad_norm_values.append(self._last_grad_norm)
            self._last_grad_norm = None

        return {}

    def finalize(self: Self, state: TrainingState, epoch: int = 0) -> MetricsType:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        result: MetricsType = {}
        if self._lr_values:
            result["mean_epoch_lr"] = sum(self._lr_values) / len(self._lr_values)
        if self._grad_norm_values:
            result["grad_norm"] = sum(self._grad_norm_values) / len(self._grad_norm_values)
        if self._step_times:
            result["epoch_time"] = sum(self._step_times) / len(self._step_times)
        return result
