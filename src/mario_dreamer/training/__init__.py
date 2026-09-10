"""Training loops, losses, and evaluation entry points."""

from mario_dreamer.training.imagination import ImaginationOutput, imagine_rollout
from mario_dreamer.training.returns import lambda_returns, replay_transition_discounts
from mario_dreamer.training.world_model_loss import (
    WorldModelLossConfig,
    WorldModelLossOutput,
    free_nats,
    world_model_loss,
)

__all__ = [
    "ImaginationOutput",
    "WorldModelLossConfig",
    "WorldModelLossOutput",
    "free_nats",
    "imagine_rollout",
    "lambda_returns",
    "replay_transition_discounts",
    "world_model_loss",
]
