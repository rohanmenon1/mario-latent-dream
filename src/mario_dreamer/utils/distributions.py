from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def two_hot_symlog(
    target: torch.Tensor,
    *,
    bins: int,
    low: float = -20.0,
    high: float = 20.0,
) -> torch.Tensor:
    if bins < 2:
        raise ValueError("bins must be at least 2")

    target = symlog(target).clamp(low, high)
    position = (target - low) / (high - low) * (bins - 1)
    lower = position.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp(0, bins - 1)
    upper_weight = position - lower.float()
    lower_weight = 1.0 - upper_weight

    output = torch.zeros(*target.shape, bins, device=target.device, dtype=target.dtype)
    output.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    output.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return output


def two_hot_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    low: float = -20.0,
    high: float = 20.0,
    reduction: str = "mean",
) -> torch.Tensor:
    target_dist = two_hot_symlog(target, bins=logits.shape[-1], low=low, high=high)
    loss = -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"reduction must be 'none' or 'mean', got {reduction!r}")


def bins_to_scalar(
    logits: torch.Tensor,
    *,
    low: float = -20.0,
    high: float = 20.0,
) -> torch.Tensor:
    bins = torch.linspace(low, high, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    count = logits.shape[-1]
    midpoint = count // 2
    if count % 2:
        symlog_value = probs[..., midpoint] * bins[midpoint]
        symlog_value = symlog_value + (
            probs[..., :midpoint].flip(-1) * bins[:midpoint].flip(0)
            + probs[..., midpoint + 1 :] * bins[midpoint + 1 :]
        ).sum(dim=-1)
    else:
        symlog_value = (
            probs[..., :midpoint].flip(-1) * bins[:midpoint].flip(0)
            + probs[..., midpoint:] * bins[midpoint:]
        ).sum(dim=-1)
    return symexp(symlog_value)
