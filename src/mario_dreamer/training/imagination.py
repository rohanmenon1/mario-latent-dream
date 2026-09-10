from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from mario_dreamer.models.actor_critic import Actor
from mario_dreamer.models.rssm import RSSMState


@dataclass
class ImaginationOutput:
    states: RSSMState
    features: torch.Tensor
    actions: torch.Tensor
    action_logits: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor


def imagine_rollout(
    *,
    world_model: Any,
    actor: Actor,
    start: RSSMState,
    horizon: int,
) -> ImaginationOutput:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    state = start
    states = [state]
    start_feature = world_model.rssm.get_features(state)
    features = [start_feature]
    actions = []
    action_logits = []
    rewards = [world_model.reward(start_feature).squeeze(-1)]
    continues = [torch.sigmoid(world_model.continuation(start_feature).squeeze(-1))]

    for _ in range(horizon):
        feature = world_model.rssm.get_features(state).detach()
        dist = actor.distribution(feature)
        action = dist.sample()
        one_hot = F.one_hot(action, num_classes=world_model.config.action_dim).float()
        state = world_model.rssm.img_step(state, one_hot)
        imagined_feature = world_model.rssm.get_features(state)

        states.append(state)
        features.append(imagined_feature)
        actions.append(action)
        action_logits.append(dist.logits)
        rewards.append(world_model.reward(imagined_feature).squeeze(-1))
        continues.append(torch.sigmoid(world_model.continuation(imagined_feature).squeeze(-1)))

    return ImaginationOutput(
        states=world_model.rssm._stack_states(states),
        features=torch.stack(features, dim=1),
        actions=torch.stack(actions, dim=1),
        action_logits=torch.stack(action_logits, dim=1),
        rewards=torch.stack(rewards, dim=1),
        continues=torch.stack(continues, dim=1),
    )
