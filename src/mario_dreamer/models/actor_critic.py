from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mario_dreamer.models.layers import RMSNorm, init_dreamer_module, scale_output_
from mario_dreamer.utils.distributions import bins_to_scalar


@dataclass(frozen=True)
class ActorCriticConfig:
    feature_dim: int
    action_dim: int
    hidden_dim: int = 512
    layers: int = 3
    value_bins: int = 255
    unimix_ratio: float = 0.01


def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    *,
    output_scale: float,
) -> nn.Sequential:
    if layers <= 0:
        raise ValueError("layers must be positive")
    modules: list[nn.Module] = []
    dim = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU()])
        dim = hidden_dim
    output = nn.Linear(dim, output_dim)
    modules.append(output)
    network = nn.Sequential(*modules)
    network.apply(init_dreamer_module)
    scale_output_(output, output_scale)
    return network


class Actor(nn.Module):
    """Categorical actor over discrete Mario actions."""

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.net = mlp(
            config.feature_dim,
            config.hidden_dim,
            config.action_dim,
            config.layers,
            output_scale=0.01,
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        leading_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        logits = self.net(flat)
        return logits.reshape(*leading_shape, self.config.action_dim)

    def distribution(self, features: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.logits(features)
        probs = torch.softmax(logits, dim=-1)
        uniform = torch.full_like(probs, 1.0 / self.config.action_dim)
        probs = (1.0 - self.config.unimix_ratio) * probs + self.config.unimix_ratio * uniform
        return torch.distributions.Categorical(probs=probs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)


class Critic(nn.Module):
    """Distributional value critic over RSSM features."""

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.net = mlp(
            config.feature_dim,
            config.hidden_dim,
            config.value_bins,
            config.layers,
            output_scale=0.0,
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        leading_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        logits = self.net(flat)
        return logits.reshape(*leading_shape, self.config.value_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return bins_to_scalar(self.logits(features)).unsqueeze(-1)
