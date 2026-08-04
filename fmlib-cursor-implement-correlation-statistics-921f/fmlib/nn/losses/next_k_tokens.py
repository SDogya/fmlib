import warnings
from copy import deepcopy
from typing import Dict, Self, Tuple

import torch
import torch.nn.functional as F  # noqa: N812

from fmlib.constants.batches import Batch
from fmlib.constants.losses import (
    DEFAULT_K_TOKENS_CATEGORICAL_LOSS,
    DEFAULT_K_TOKENS_EMPTY_LABEL,
    DEFAULT_K_TOKENS_EVENT_IDS_INP_NAME,
    DEFAULT_K_TOKENS_LAST_HIDDEN_STATE_NAME,
    DEFAULT_K_TOKENS_NUMERIC_LOSS,
    DEFAULT_K_TOKENS_PADDING_MASK_NAME,
    DEFAULT_K_TOKENS_SEQ_FEATURES_NAME,
    DEFAULT_K_TOKENS_TIMESTAMP_NAME,
)
from fmlib.training.training_types import MetricsType


class NextKTokensLoss(torch.nn.Module):
    """
    Лосс для тела модели на предсказания следующих токенов

    Аргументы:
        hidden_size (int): размерность скрытого слоя в трансформере
        columns_meta (Dict[str, Dict[str, str | int | float | None]]): метаданные колонок.
            Имеют вид:
            ```yaml
            channel_group:
                type: categorical
                cardinality: 10
                event_id: 2
            ```
        horizon (int): количество шагов вперед, на которое налагается лосс.
        horizion_loss_weight (float): степень коэффициентов для лоссов вперед.
            `[(coef + 1) ** horizion_loss_weight for coef in range(horizon)]`
        feature_loss_weights (Dict[str, float] | None): веса для лоссов по признакам.
            По умолчанию - `None`, т.е. коэффициент 1.0
        numeric_loss (torch.nn.Module): loss для числовых признаков.
            По умолчанию - DEFAULT_K_TOKENS_NUMERIC_LOSS.
        categorical_loss: loss для категориальных признаков.
            По умолчанию - DEFAULT_K_TOKENS_CATEGORICAL_LOSS.
        seq_features_name (str): название колонки с последовательными признаками.
            По умолчанию - DEFAULT_K_TOKENS_SEQ_FEATURES_NAME.
        timestamp_name (str): название колонки с временными метками.
            По умолчанию - DEFAULT_K_TOKENS_TIMESTAMP_NAME.
        event_ids_inp_name (str): название колонки с event_ids.
            По умолчанию - DEFAULT_K_TOKENS_EVENT_IDS_INP_NAME.
        padding_mask_name (str): название колонки с маской.
            По умолчанию - DEFAULT_K_TOKENS_PADDING_MASK_NAME.
        last_hidden_state_name (str): название колонки с последним скрытым состоянием.
            По умолчанию - DEFAULT_K_TOKENS_LAST_HIDDEN_STATE_NAME.
        k_tokens_empty_label (int): метка пустого токена.
            По умолчанию - DEFAULT_K_TOKENS_EMPTY_LABEL.
    """

    def __init__(
        self: Self,
        hidden_size: int,
        columns_meta: Dict[str, Dict[str, str | int | float | None]],
        horizon: int = 1,
        horizion_loss_weight: float = 1.0,
        feature_loss_weights: Dict[str, float] | None = None,
        numeric_loss: torch.nn.Module = DEFAULT_K_TOKENS_NUMERIC_LOSS,
        categorical_loss: torch.nn.Module = DEFAULT_K_TOKENS_CATEGORICAL_LOSS,
        seq_features_name: str = DEFAULT_K_TOKENS_SEQ_FEATURES_NAME,
        timestamp_name: str = DEFAULT_K_TOKENS_TIMESTAMP_NAME,
        event_ids_inp_name: str = DEFAULT_K_TOKENS_EVENT_IDS_INP_NAME,
        padding_mask_name: str = DEFAULT_K_TOKENS_PADDING_MASK_NAME,
        last_hidden_state_name: str = DEFAULT_K_TOKENS_LAST_HIDDEN_STATE_NAME,
        k_tokens_empty_label: int = DEFAULT_K_TOKENS_EMPTY_LABEL,
    ) -> None:
        super().__init__()
        self.model_columns_meta = deepcopy(columns_meta)

        if horizon != 1:
            msg: str = "The loss doesn`t support prediction for a horizon greather than 1."
            raise ValueError(msg)
        if horizion_loss_weight <= 0.0:
            msg: str = f"`horizon_loss_coef` should be positive, but got {horizion_loss_weight=}. "
            raise ValueError(msg)

        if hidden_size < 1:
            msg: str = f"Hidden size must be positive, but got {hidden_size=}. "
            raise ValueError(msg)
        elif hidden_size < 128:
            msg: str = f"Suspiciously low hidden size. Got {hidden_size=}."
            warnings.warn(msg, stacklevel=2)

        self.hidden_size: int = hidden_size

        self.lm_heads = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: torch.nn.Linear(
                            self.hidden_size,
                            meta.get("cardinality", 1),
                        )
                        for key, meta in self.model_columns_meta.items()
                    }
                )
                for _ in range(horizon)
            ]
        )

        self.feature_loss_weights = dict.fromkeys(self.model_columns_meta.keys(), 1.0)
        if feature_loss_weights is not None:
            assert all(key in (list(self.model_columns_meta) + ["timedelta"]) for key in feature_loss_weights)
            self.feature_loss_weights.update(feature_loss_weights)

        if "timedelta" not in self.feature_loss_weights:
            warnings.warn("No `timedelta` in `feature_loss_weights`.", stacklevel=2)
            self.feature_loss_weights["timedelta"] = 0.6

        if self.feature_loss_weights["timedelta"] != 0.0:
            for horizon_module in self.lm_heads:
                horizon_module["timedelta"] = torch.nn.Linear(self.hidden_size, 1)

        self.stats: MetricsType = {}

        self.horizon = horizon
        self.horizion_loss_weight = horizion_loss_weight
        self.coefs = [(coef + 1) ** horizion_loss_weight for coef in range(horizon)]
        self.numeric_loss = numeric_loss
        self.categorical_loss = categorical_loss

        self.k_tokens_empty_label: int = k_tokens_empty_label
        self.seq_features_name: str = seq_features_name
        self.timestamp_name: str = timestamp_name
        self.event_ids_inp_name: str = event_ids_inp_name
        self.padding_mask_name: str = padding_mask_name
        self.last_hidden_state_name: str = last_hidden_state_name

    def create_labels_with_horizon(
        self: Self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        horizon_offset: int,
        n_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Создает лейблы для следующих токенов.
        Возвращает сдвинутый и несдвинутый набор токенов.
        """
        with torch.no_grad():
            labels = input_ids.detach().clone()
            if n_classes > 1:
                pad_tokens = labels == 0
                labels[pad_tokens] = self.k_tokens_empty_label

        shifted_labels = labels[:, horizon_offset:].contiguous()  # shift labels
        shifted_logits = logits[:, :-horizon_offset, :].contiguous()  # shift logits
        return shifted_logits, shifted_labels

    def calculate_loss(
        self: Self,
        shifted_logits: torch.Tensor,
        shifted_labels: torch.Tensor,
        n_classes: int,
        padding_mask: torch.Tensor | None = None,
        horizon_offset: int | None = None,
    ):
        """
        n_classes == 1 -> numeric, else categorical
        """
        assert shifted_logits.shape[1] == shifted_labels.shape[1]
        if n_classes == 1:
            assert padding_mask is not None
            shifted_logits = shifted_logits.squeeze(dim=-1)
            loss = self.numeric_loss(shifted_logits, shifted_labels)

            num_items = padding_mask[:, horizon_offset:]
            loss *= num_items

            return loss.sum(), num_items.sum()
        elif n_classes > 1:
            loss = self.categorical_loss(
                shifted_logits.squeeze(0),
                shifted_labels.squeeze(0),
            )
            num_items = (shifted_labels != self.k_tokens_empty_label).sum()

            return loss, num_items
        else:
            msg: str = "Corrupted n_classes value."
            raise ValueError(msg)

    def get_timedeltas(self: Self, seq_features: Batch, horizon_offset: int, padding_mask: torch.Tensor) -> torch.Tensor:
        timestamps: torch.Tensor = seq_features.get(self.timestamp_name)
        time_deltas = timestamps[:, horizon_offset:] - timestamps[:, :-horizon_offset]
        time_deltas = F.pad(time_deltas, (horizon_offset, 0), value=0.0)
        time_deltas[~(padding_mask.bool())] = 0.0
        return time_deltas

    def calculate_delta_loss(
        self: Self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        horizon_offset: int,
        padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shifted_logits, shifted_labels = self.create_labels_with_horizon(
            input_ids=labels.get(self.timestamp_name),
            logits=logits,
            horizon_offset=horizon_offset,
            n_classes=1,
        )
        loss, head_num_items = self.calculate_loss(
            shifted_logits=shifted_logits,
            shifted_labels=shifted_labels,
            n_classes=1,
            padding_mask=padding_mask,
            horizon_offset=horizon_offset,
        )
        return loss, head_num_items

    def get_last_hidden_state(self: Self, outputs: Batch) -> torch.Tensor:
        return outputs.get(self.last_hidden_state_name)

    def get_timestamps(self: Self, seq_features: Batch) -> torch.Tensor | None:
        return seq_features.get(self.timestamp_name, None)

    def get_padding_mask(self: Self, seq_features: Batch) -> torch.Tensor:
        return seq_features.get(self.padding_mask_name)

    def get_event_attn_mask(self: Self, seq_features: Batch, event_id: int | list[int] | None = None) -> torch.LongTensor:
        if event_id is not None:
            if isinstance(event_id, int):
                event_id = [event_id]

            event_ids: torch.LongTensor = seq_features.get(self.event_ids_inp_name)
            evt_id: torch.LongTensor = torch.tensor(event_id).to(event_ids.device)
            mask: torch.LongTensor = torch.isin(event_ids, evt_id).long()
            return mask
        else:
            return self.get_padding_mask(seq_features)

    def get_seq_features(self: Self, targets: Batch) -> Batch:
        return targets[self.seq_features_name]

    @torch.jit.export
    def get_stats(self: Self) -> MetricsType:
        return self.stats

    def forward(self: Self, outputs: Batch, targets: Batch, **kwargs) -> torch.Tensor:
        losses = {}
        total_num_items = {}

        seq_features: Batch = self.get_seq_features(targets)
        last_hidden_state: torch.Tensor = self.get_last_hidden_state(outputs)

        for horizon_idx in range(self.horizon):
            horizon_offset: int = horizon_idx + 1
            if horizon_offset >= last_hidden_state.shape[1]:
                break

            for key, meta in self.model_columns_meta.items():
                input_ids: torch.Tensor = seq_features[key]
                logits: torch.Tensor = self.lm_heads[horizon_idx][key](last_hidden_state)

                # clipping implimentation
                if "clip" in meta:
                    input_ids = input_ids.clamp(max=meta["clip"] - 1)

                shifted_logits, shifted_labels = self.create_labels_with_horizon(
                    input_ids=input_ids,
                    logits=logits,
                    horizon_offset=horizon_offset,
                    n_classes=meta.get("cardinality", 1),
                )

                event_padding_mask = self.get_event_attn_mask(
                    event_id=meta.get("event_id", None),
                    seq_features=seq_features,
                )

                loss, head_num_items = self.calculate_loss(
                    shifted_logits=shifted_logits,
                    shifted_labels=shifted_labels,
                    n_classes=meta.get("cardinality", 1),
                    padding_mask=event_padding_mask,
                    horizon_offset=horizon_offset,
                )

                if self.training:
                    loss *= self.feature_loss_weights[key]
                    loss /= self.coefs[horizon_idx]
                total_num_items[f"{key}_head_{horizon_idx}"] = head_num_items
                losses[f"{key}_head_{horizon_idx}"] = loss

            if self.feature_loss_weights["timedelta"] != 0.0:
                if self.get_timestamps(seq_features) is None:
                    msg = "Timestamps are not produced by transforms."
                    raise ValueError(msg)
                logits = self.lm_heads[horizon_idx]["timedelta"](last_hidden_state)
                padding_mask: torch.Tensor = self.get_padding_mask(seq_features)

                loss, head_num_items = self.calculate_delta_loss(
                    logits=logits,
                    labels=seq_features,
                    horizon_offset=horizon_offset,
                    padding_mask=padding_mask,
                )
                total_num_items[f"timedelta_head_{horizon_idx}"] = head_num_items
                if self.training:
                    loss *= self.feature_loss_weights["timedelta"]

                loss = loss / self.coefs[horizon_idx]
                losses[f"timedelta_head_{horizon_idx}"] = loss

        self.stats = {
            key: value.detach().cpu().item() / total_num_items[key].detach().cpu().item()
            for key, value in losses.items()
            }
        all_losses = torch.cat([
            (loss_value / total_num_items[loss_name]).reshape(-1)
            for loss_name, loss_value in losses.items()
        ])
        return torch.mean(all_losses)
