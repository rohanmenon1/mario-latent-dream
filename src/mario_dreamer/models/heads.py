from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mario_dreamer.models.layers import RMSNorm, init_dreamer_module, scale_output_
from mario_dreamer.utils.distributions import bins_to_scalar


@dataclass(frozen=True)
class MLPHeadConfig:
    input_dim: int
    hidden_dim: int = 512
    output_dim: int = 1
    layers: int = 1
    output_scale: float = 1.0


class MLPHead(nn.Module):
    """Small MLP prediction head for latent RSSM features."""

    def __init__(self, config: MLPHeadConfig) -> None:
        super().__init__()
        if config.layers <= 0:
            raise ValueError("layers must be positive")
        self.config = config

        modules: list[nn.Module] = []
        dim = config.input_dim
        for _ in range(config.layers):
            modules.extend(
                [
                    nn.Linear(dim, config.hidden_dim),
                    RMSNorm(config.hidden_dim),
                    nn.SiLU(),
                ]
            )
            dim = config.hidden_dim
        output = nn.Linear(dim, config.output_dim)
        modules.append(output)
        self.net = nn.Sequential(*modules)
        self.net.apply(init_dreamer_module)
        scale_output_(output, config.output_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2:
            raise ValueError(f"expected features with final dim, got {tuple(features.shape)}")
        leading_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        output = self.net(flat)
        return output.reshape(*leading_shape, self.config.output_dim)


class RewardPredictor(MLPHead):
    """Predicts rewards from RSSM features.

    If the configured output dimension is greater than 1, the head represents a
    two-hot symlog distribution and `forward()` returns its scalar expectation.
    """

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.logits(features)
        if logits.shape[-1] == 1:
            return logits
        return bins_to_scalar(logits).unsqueeze(-1)


class ContinuationPredictor(MLPHead):
    """Predicts logits for whether an episode continues after a transition."""
