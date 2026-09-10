from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mario_dreamer.models.layers import BlockLinear, RMSNorm, init_dreamer_module


@dataclass(frozen=True)
class ConvEncoderConfig:
    in_channels: int = 3
    obs_size: int = 64
    depth: int = 32
    embed_dim: int = 1024


@dataclass(frozen=True)
class ConvDecoderConfig:
    out_channels: int = 3
    obs_size: int = 64
    depth: int = 32
    feature_dim: int = 2048
    deter_dim: int = 0
    hidden_dim: int = 512
    blocks: int = 8


class ConvEncoder(nn.Module):
    """CNN encoder for square image observations divisible by 16."""

    def __init__(self, config: ConvEncoderConfig = ConvEncoderConfig()) -> None:
        super().__init__()
        if config.obs_size % 16 != 0:
            raise ValueError("ConvEncoder expects obs_size divisible by 16")

        depth = config.depth
        latent_size = config.obs_size // 16
        self.config = config
        depths = (depth * 2, depth * 3, depth * 4, depth * 4)
        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for out_channels in depths:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    _ChannelRMSNorm(out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        raw_dim = depths[-1] * latent_size * latent_size
        self.output_dim = config.embed_dim if config.embed_dim > 0 else raw_dim
        layers.append(nn.Flatten())
        if config.embed_dim > 0:
            layers.extend(
                [
                    nn.Linear(raw_dim, config.embed_dim),
                    RMSNorm(config.embed_dim),
                    nn.SiLU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.net.apply(init_dreamer_module)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"expected obs shape [B,C,H,W], got {tuple(obs.shape)}")
        return self.net(obs - 0.5)


class ConvDecoder(nn.Module):
    """CNN decoder that maps RSSM features back to image logits."""

    def __init__(self, config: ConvDecoderConfig = ConvDecoderConfig()) -> None:
        super().__init__()
        if config.obs_size % 16 != 0:
            raise ValueError("ConvDecoder expects obs_size divisible by 16")

        depth = config.depth
        latent_size = config.obs_size // 16
        self.config = config
        self.latent_size = latent_size
        depths = (depth * 2, depth * 3, depth * 4, depth * 4)
        self.deter_dim = config.deter_dim or config.feature_dim // 2
        if self.deter_dim >= config.feature_dim:
            raise ValueError("decoder deter_dim must be smaller than feature_dim")
        spatial_dim = depths[-1] * latent_size * latent_size
        self.deter_input = BlockLinear(
            self.deter_dim,
            spatial_dim,
            config.blocks,
        )
        self.stoch_input = nn.Sequential(
            nn.Linear(config.feature_dim - self.deter_dim, 2 * config.hidden_dim),
            RMSNorm(2 * config.hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_dim, spatial_dim),
        )
        self.spatial_norm = _ChannelRMSNorm(depths[-1])
        layers = []
        in_channels = depths[-1]
        for out_channels in reversed(depths[:-1]):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    _ChannelRMSNorm(out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, config.out_channels, kernel_size=5, padding=2),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.stoch_input.apply(init_dreamer_module)
        self.net.apply(init_dreamer_module)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"expected feature shape [B,D], got {tuple(features.shape)}")
        deter, stoch = features.split(
            [self.deter_dim, self.config.feature_dim - self.deter_dim],
            dim=-1,
        )
        channels = self.config.depth * 4
        deter_hidden = self.deter_input(deter).view(
            features.shape[0],
            self.config.blocks,
            self.latent_size,
            self.latent_size,
            channels // self.config.blocks,
        )
        deter_hidden = deter_hidden.permute(0, 2, 3, 1, 4).reshape(
            features.shape[0],
            self.latent_size,
            self.latent_size,
            channels,
        ).permute(0, 3, 1, 2)
        stoch_hidden = self.stoch_input(stoch).view(
            features.shape[0],
            channels,
            self.latent_size,
            self.latent_size,
        )
        hidden = torch.nn.functional.silu(self.spatial_norm(deter_hidden + stoch_hidden))
        return self.net(hidden)

    def reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(features))


class _ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 3, 1)
        value = self.norm(value)
        return value.permute(0, 3, 1, 2)
