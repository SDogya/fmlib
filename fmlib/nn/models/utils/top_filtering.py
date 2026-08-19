import warnings

import torch
import torch.nn.functional as func


def top_k_filtering(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    filter_value = torch.finfo(logits.dtype).min

    if top_k > 0:
        idx_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1]
        logits[idx_to_remove] = filter_value
    elif top_k < 0:
        msg: str = f"Top-k must be positive or zero. Got: {top_k=}."
        raise ValueError(msg)

    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float = 0.0) -> torch.Tensor:
    filter_value = torch.finfo(logits.dtype).min

    if top_p > 0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(func.softmax(sorted_logits, -1), -1)
        remove_idx = cum_probs > top_p
        remove_idx[..., 1:] = remove_idx[..., :-1].clone()
        remove_idx[..., 0] = 0
        sorted_logits[remove_idx] = filter_value
        logits = torch.gather(sorted_logits, -1, sorted_idx.argsort(-1))
    elif top_p < 0:
        msg: str = f"Top-p must be positive or zero. Got: {top_p=}"
        raise ValueError(msg)

    return logits


def top_p_top_k_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)

    if (top_k == 0) and (top_p == 0.0):
        msg: str = f"Top-k and top-p combination does not do anything. Got: {top_k=}, {top_p=}."
        warnings.warn(msg, stacklevel=2)

    return logits
