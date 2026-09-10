from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mario_dreamer.models.rssm import RSSMState
from mario_dreamer.utils.distributions import two_hot_loss


@dataclass(frozen=True)
class WorldModelLossConfig:
    image_scale: float = 1.0
    image_edge_scale: float = 0.0
    reward_scale: float = 1.0
    continuation_scale: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    kl_free_nats: float = 1.0
    continuation_discount: float = 0.997


@dataclass(frozen=True)
class WorldModelLossOutput:
    total: torch.Tensor
    image: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    kl: torch.Tensor
    dyn_kl: torch.Tensor
    rep_kl: torch.Tensor


def free_nats(kl: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.maximum(kl, torch.full_like(kl, threshold))


def world_model_loss(
    *,
    image_logits: torch.Tensor,
    target_obs: torch.Tensor,
    reward_logits: torch.Tensor,
    target_reward: torch.Tensor,
    continuation_logits: torch.Tensor,
    target_continue: torch.Tensor,
    posterior: RSSMState,
    prior: RSSMState,
    dyn_kl_values: torch.Tensor,
    rep_kl_values: torch.Tensor,
    config: WorldModelLossConfig = WorldModelLossConfig(),
) -> WorldModelLossOutput:
    del posterior, prior

    image_error = (torch.sigmoid(image_logits) - target_obs).square()
    if config.image_edge_scale > 0.0:
        image_weight = _edge_weight(target_obs, scale=config.image_edge_scale)
        image_per_state = (image_error * image_weight).sum(dim=(-3, -2, -1))
    else:
        image_per_state = image_error.sum(dim=(-3, -2, -1))
    image_loss = image_per_state.mean()
    if reward_logits.shape[-1] == 1:
        reward_loss = F.mse_loss(reward_logits.squeeze(-1), target_reward)
    else:
        reward_loss = two_hot_loss(reward_logits, target_reward)
    continuation_loss = F.binary_cross_entropy_with_logits(
        continuation_logits.squeeze(-1),
        target_continue * config.continuation_discount,
    )
    dyn_kl = free_nats(dyn_kl_values, config.kl_free_nats).mean()
    rep_kl = free_nats(rep_kl_values, config.kl_free_nats).mean()
    kl_loss = dyn_kl + rep_kl
    total = (
        config.image_scale * image_loss
        + config.reward_scale * reward_loss
        + config.continuation_scale * continuation_loss
        + config.dyn_scale * dyn_kl
        + config.rep_scale * rep_kl
    )
    return WorldModelLossOutput(
        total=total,
        image=image_loss,
        reward=reward_loss,
        continuation=continuation_loss,
        kl=kl_loss,
        dyn_kl=dyn_kl,
        rep_kl=rep_kl,
    )


def _edge_weight(target_obs: torch.Tensor, scale: float) -> torch.Tensor:
    gray = target_obs.mean(dim=-3, keepdim=True)
    dx = F.pad(torch.abs(gray[..., :, 1:] - gray[..., :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[..., 1:, :] - gray[..., :-1, :]), (0, 0, 0, 1))
    edge = torch.maximum(dx, dy)
    flat = edge.reshape(-1, 1, *edge.shape[-2:])
    edge = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1).reshape_as(edge)
    return 1.0 + scale * edge.expand_as(target_obs)
