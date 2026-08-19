from typing import Any, Dict, List, Literal, Optional, Self

import torch

from fmlib.utils.deprecation import deprecation_warning


@deprecation_warning("Please use `make_tabular_transform` instead.")
class TabularTransform(torch.nn.Module):
    def __init__(
        self: Self,
        cat_features_name: str = "cat_features",
        num_features_name: str = "num_features",
        seq_hidden_state_name: str = "seq_hidden_state",
        bypass: List[str] | None = None,
    ) -> None:
        super().__init__()

        if bypass is None:
            bypass = ["epk_id"]

        self.bypass: List[str] = sorted(bypass)
        self.cat_features_name: str = cat_features_name
        self.num_features_name: str = num_features_name
        self.seq_hidden_state_name: str = seq_hidden_state_name

    def forward(self: Self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "cat_features": batch[self.cat_features_name],
            "num_features": batch[self.num_features_name],
            "hidden_states": batch[self.seq_hidden_state_name].float(),
            **{name: batch[name] for name in sorted(self.bypass)},
        }


@deprecation_warning("Please use `make_uplift_transform` instead.")
class UpliftTabularTransform(torch.nn.Module):
    def __init__(
        self: Self,
        channel_type: int,
        cat_features_name: str = "cat_features",
        num_features_name: str = "num_features",
        seq_hidden_state_name: str = "seq_hidden_state",
        bypass: List[str] | None = None,
    ) -> None:
        super().__init__()
        if bypass is None:
            bypass = ["epk_id"]

        self.channel_type: int = channel_type
        self.bypass: List[str] = sorted(bypass)
        self.cat_features_name: str = cat_features_name
        self.num_features_name: str = num_features_name
        self.seq_hidden_state_name: str = seq_hidden_state_name

    def forward(self: Self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size: int = batch[self.cat_features_name].size(0)
        device: torch.device = batch[self.cat_features_name].device
        group: torch.LongTensor = torch.ones(batch_size, dtype=torch.long, device=device)
        return {
            "group": (group * self.channel_type),
            "cat_features": batch[self.cat_features_name],
            "num_features": batch[self.num_features_name],
            "hidden_states": batch[self.seq_hidden_state_name].float(),
            **{name: batch[name] for name in sorted(self.bypass)},
        }


@deprecation_warning("Please use `make_sequential_transform` instead.")
class SequentialTransform(torch.nn.Module):
    def __init__(
        self: Self,
        events_meta: Dict[str, Dict[str, Any]],
        timestamp_inp_name: str = "evt_dttm",
        time_encoding: Optional[Literal["absolute", "delta"]] = None,
        events_ids_inp_name: str = "event_ids",
        padding_mask_inp_name: str = "evt_attr_9_mask",
        bypass: List[str] | None = None,
    ) -> None:
        if bypass is None:
            bypass = ["epk_id"]
        super().__init__()
        self.bypass: List[str] = sorted(bypass)
        self.events_meta: Dict[str, Dict[str, Any]] = events_meta
        self.timestamp_inp_name: str = timestamp_inp_name
        self.time_encoding: Optional[Literal["absolute", "delta"]] = time_encoding
        self.events_ids_inp_name: str = events_ids_inp_name
        self.padding_mask_inp_name: str = padding_mask_inp_name

    def make_padded(self: Self, tensor: torch.Tensor, mask: torch.BoolTensor, padding_value: int) -> torch.Tensor:
        return torch.where(
            other=padding_value,
            condition=mask,
            input=tensor,
        )

    def forward_impl(self: Self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}

        padding_mask: torch.BoolTensor = batch[self.padding_mask_inp_name]
        device: torch.device = padding_mask.device

        result["events"] = {name: self.make_padded(batch[name], padding_mask, 0) for name in self.events_meta}
        for name in self.events_meta:
            if result["events"][name].dtype.is_floating_point:
                result["events"][name] = result["events"][name].float()

        result["events"]["padding_mask"] = padding_mask
        result["events"]["event_ids"] = batch[self.events_ids_inp_name]

        ######## event_encoder_attn_mask preparation start ########
        event_attn_mask = []
        for _, meta in self.events_meta.items():
            if meta.get("event_id") is None:
                mask = padding_mask
            else:
                event_id: List[int] = [meta["event_id"]] if isinstance(meta["event_id"], int) else meta["event_id"]
                mask = torch.isin(
                    batch[self.events_ids_inp_name],
                    torch.tensor(event_id, device=device),
                )
            event_attn_mask.append(mask[..., None])
        event_attn_mask.append(torch.ones_like(event_attn_mask[-1]))  # mask for time embedding
        event_attn_mask = torch.cat(event_attn_mask, dim=-1)  # (batch_size, seq_len, n_features)
        expanded_attn_mask = torch.einsum("bsi,bsj->bsij", [event_attn_mask, event_attn_mask])
        diag_matrix = torch.eye(len(self.events_meta) + 1, dtype=torch.bool, device=device)
        event_encoder_attn_mask = expanded_attn_mask | diag_matrix[None, None, ...]
        result["events"]["event_encoder_attention_mask"] = event_encoder_attn_mask
        ######## event_encoder_attn_mask preparation end ########

        ######## encoding_timestamps preparation start ########
        if self.time_encoding is None:
            result["events"]["encoding_timestamps"] = None
        elif self.time_encoding == "absolute":
            result["events"]["encoding_timestamps"] = batch[self.timestamp_inp_name]
        elif self.time_encoding == "delta":
            timestamps: torch.Tensor = batch[self.timestamp_inp_name]
            time_deltas = timestamps[:, 1:] - timestamps[:, :-1]
            time_deltas = torch.nn.functional.pad(time_deltas, (1, 0), value=0.0)
            time_deltas.masked_fill_(padding_mask.logical_not(), 0)
            result["events"]["encoding_timestamps"] = time_deltas
        ######## encoding_timestamps preparation end ########

        result["events"]["positional_timestamps"] = batch[self.timestamp_inp_name]
        return result

    def forward(self: Self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            **{name: batch[name] for name in self.bypass},
            **self.forward_impl(batch),
        }
