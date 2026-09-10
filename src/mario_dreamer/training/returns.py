from __future__ import annotations

import torch


def replay_transition_discounts(
    continues: torch.Tensor,
    is_first: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Discount state-to-next-state replay transitions without crossing resets."""

    if continues.shape != is_first.shape:
        raise ValueError("continues and is_first must have matching [B,T] shapes")
    if continues.ndim != 2:
        raise ValueError("continues and is_first must have shape [B,T]")
    return discount * continues[:, 1:] * (1.0 - is_first[:, 1:])


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute temporal lambda returns.

    Shapes:
    - rewards: `[B, T]`
    - values: `[B, T]`
    - continues: `[B, T]`, usually discount * continuation probability
    - bootstrap: `[B]`
    """

    if rewards.shape != values.shape or rewards.shape != continues.shape:
        raise ValueError("rewards, values, and continues must have matching [B,T] shapes")
    if bootstrap.shape != rewards[:, 0].shape:
        raise ValueError("bootstrap must have shape [B]")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    next_values = torch.cat([values[:, 1:], bootstrap[:, None]], dim=1)
    inputs = rewards + continues * next_values * (1.0 - lambda_)
    returns = []
    last = bootstrap
    for index in reversed(range(rewards.shape[1])):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        returns.append(last)
    returns.reverse()
    return torch.stack(returns, dim=1)
