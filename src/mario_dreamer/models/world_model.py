from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mario_dreamer.models.cnn import ConvDecoder, ConvDecoderConfig, ConvEncoder, ConvEncoderConfig
from mario_dreamer.models.heads import ContinuationPredictor, MLPHeadConfig, RewardPredictor
from mario_dreamer.models.rssm import RSSM, RSSMConfig, RSSMState
from mario_dreamer.training.world_model_loss import (
    WorldModelLossConfig,
    WorldModelLossOutput,
    world_model_loss,
)


@dataclass(frozen=True)
class WorldModelConfig:
    action_dim: int
    obs_channels: int = 3
    obs_size: int = 64
    depth: int = 32
    embed_dim: int = 1024
    deter_dim: int = 1024
    stoch_classes: int = 32
    stoch_categories: int = 32
    hidden_dim: int = 1024
    head_layers: int = 1
    reward_bins: int = 255
    unimix_ratio: float = 0.01
    rssm_blocks: int = 8


@dataclass
class WorldModelOutput:
    posterior: RSSMState
    prior: RSSMState
    features: torch.Tensor
    image_logits: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    losses: WorldModelLossOutput | None = None


class WorldModel(nn.Module):
    """Encoder, RSSM, decoder, reward head, and continuation head."""

    def __init__(
        self,
        config: WorldModelConfig,
        loss_config: WorldModelLossConfig = WorldModelLossConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.loss_config = loss_config
        self.encoder = ConvEncoder(
            ConvEncoderConfig(
                in_channels=config.obs_channels,
                obs_size=config.obs_size,
                depth=config.depth,
                embed_dim=config.embed_dim,
            )
        )
        self.rssm = RSSM(
            RSSMConfig(
                action_dim=config.action_dim,
                embed_dim=self.encoder.output_dim,
                deter_dim=config.deter_dim,
                stoch_classes=config.stoch_classes,
                stoch_categories=config.stoch_categories,
                hidden_dim=config.hidden_dim,
                unimix_ratio=config.unimix_ratio,
                blocks=config.rssm_blocks,
            )
        )
        self.decoder = ConvDecoder(
            ConvDecoderConfig(
                out_channels=config.obs_channels,
                obs_size=config.obs_size,
                depth=config.depth,
                feature_dim=self.rssm.feature_dim,
                deter_dim=config.deter_dim,
                hidden_dim=config.hidden_dim,
                blocks=config.rssm_blocks,
            )
        )
        head_config = MLPHeadConfig(
            input_dim=self.rssm.feature_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.reward_bins,
            layers=config.head_layers,
            output_scale=0.0,
        )
        self.reward = RewardPredictor(head_config)
        self.continuation = ContinuationPredictor(
            MLPHeadConfig(
                input_dim=self.rssm.feature_dim,
                hidden_dim=config.hidden_dim,
                output_dim=1,
                layers=config.head_layers,
                output_scale=1.0,
            )
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
        continues: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
        is_first: torch.Tensor | None = None,
    ) -> WorldModelOutput:
        if obs.ndim != 5:
            raise ValueError(f"expected obs shape [B,T,C,H,W], got {tuple(obs.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"expected integer actions shape [B,T], got {tuple(actions.shape)}")
        batch_size, time_steps = obs.shape[:2]

        flat_obs = obs.reshape(batch_size * time_steps, *obs.shape[2:])
        embeds = self.encoder(flat_obs).reshape(batch_size, time_steps, -1)
        if prev_actions is None:
            action_one_hot = F.one_hot(actions.long(), num_classes=self.config.action_dim).float()
            prev_action_one_hot = torch.zeros_like(action_one_hot)
            prev_action_one_hot[:, 1:] = action_one_hot[:, :-1]
        else:
            if prev_actions.shape != actions.shape:
                raise ValueError(
                    "expected prev_actions to match actions shape "
                    f"{tuple(actions.shape)}, got {tuple(prev_actions.shape)}"
                )
            prev_action_one_hot = F.one_hot(
                prev_actions.long(),
                num_classes=self.config.action_dim,
            ).float()
        if is_first is None:
            posterior, prior = self.rssm.observe(embeds, prev_action_one_hot)
        else:
            posterior, prior = self.rssm.observe(
                embeds,
                prev_action_one_hot,
                is_first=is_first,
            )
        features = self.rssm.get_features(posterior)

        flat_features = features.reshape(batch_size * time_steps, -1)
        image_logits = self.decoder(flat_features).reshape(batch_size, time_steps, *obs.shape[2:])
        reward_logits = self.reward.logits(features)
        reward = self.reward(features)
        continuation = self.continuation(features)

        losses = None
        if rewards is not None and continues is not None:
            dyn_kl, rep_kl = self.rssm.kl_losses(posterior, prior)
            losses = world_model_loss(
                image_logits=image_logits,
                target_obs=obs,
                reward_logits=reward_logits,
                target_reward=rewards,
                continuation_logits=continuation,
                target_continue=continues,
                posterior=posterior,
                prior=prior,
                dyn_kl_values=dyn_kl,
                rep_kl_values=rep_kl,
                config=self.loss_config,
            )

        return WorldModelOutput(
            posterior=posterior,
            prior=prior,
            features=features,
            image_logits=image_logits,
            reward=reward,
            continuation=continuation,
            losses=losses,
        )
