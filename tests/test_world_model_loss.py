import torch

from mario_dreamer.models import ContinuationPredictor, MLPHeadConfig, RewardPredictor, RSSMState
from mario_dreamer.training import WorldModelLossConfig, free_nats, world_model_loss


def test_reward_and_continuation_heads_shape_and_gradients():
    reward = RewardPredictor(MLPHeadConfig(input_dim=64, hidden_dim=32, layers=2))
    continuation = ContinuationPredictor(MLPHeadConfig(input_dim=64, hidden_dim=32, layers=2))
    features = torch.randn(3, 5, 64)

    reward_pred = reward(features)
    continuation_logits = continuation(features)
    loss = reward_pred.square().mean() + continuation_logits.square().mean()
    loss.backward()

    assert reward_pred.shape == (3, 5, 1)
    assert continuation_logits.shape == (3, 5, 1)
    assert any(param.grad is not None for param in reward.parameters())
    assert any(param.grad is not None for param in continuation.parameters())


def test_free_nats_floors_kl_values():
    kl = torch.tensor([0.0, 0.5, 2.0])

    floored = free_nats(kl, threshold=1.0)

    assert floored.tolist() == [1.0, 1.0, 2.0]


def test_world_model_loss_shapes_and_gradients():
    batch, time = 2, 4
    image_logits = torch.randn(batch, time, 3, 64, 64, requires_grad=True)
    target_obs = torch.rand(batch, time, 3, 64, 64)
    reward_logits = torch.randn(batch, time, 31, requires_grad=True)
    target_reward = torch.randn(batch, time)
    continuation_logits = torch.randn(batch, time, 1, requires_grad=True)
    target_continue = torch.ones(batch, time)
    posterior = RSSMState(
        deter=torch.randn(batch, time, 8),
        stoch=torch.randn(batch, time, 2, 4),
        logits=torch.randn(batch, time, 2, 4),
    )
    prior = RSSMState(
        deter=torch.randn(batch, time, 8),
        stoch=torch.randn(batch, time, 2, 4),
        logits=torch.randn(batch, time, 2, 4),
    )
    dyn_kl_values = torch.rand(batch, time, requires_grad=True)
    rep_kl_values = torch.rand(batch, time, requires_grad=True)

    losses = world_model_loss(
        image_logits=image_logits,
        target_obs=target_obs,
        reward_logits=reward_logits,
        target_reward=target_reward,
        continuation_logits=continuation_logits,
        target_continue=target_continue,
        posterior=posterior,
        prior=prior,
        dyn_kl_values=dyn_kl_values,
        rep_kl_values=rep_kl_values,
        config=WorldModelLossConfig(kl_free_nats=0.0),
    )
    losses.total.backward()

    assert losses.total.ndim == 0
    assert losses.image.ndim == 0
    assert losses.reward.ndim == 0
    assert losses.continuation.ndim == 0
    assert losses.kl.ndim == 0
    assert losses.dyn_kl.ndim == 0
    assert losses.rep_kl.ndim == 0
    assert image_logits.grad is not None
    assert reward_logits.grad is not None
    assert continuation_logits.grad is not None
    assert dyn_kl_values.grad is not None
    assert rep_kl_values.grad is not None


def test_edge_weighted_image_loss_changes_image_term():
    batch, time = 1, 2
    target_obs = torch.zeros(batch, time, 3, 16, 16)
    target_obs[..., 4:8, 4:8] = 1.0
    image_logits = torch.randn_like(target_obs, requires_grad=True)
    reward_logits = torch.randn(batch, time, 31, requires_grad=True)
    continuation_logits = torch.randn(batch, time, 1, requires_grad=True)
    target_reward = torch.zeros(batch, time)
    target_continue = torch.ones(batch, time)
    posterior = RSSMState(
        deter=torch.randn(batch, time, 8),
        stoch=torch.randn(batch, time, 2, 4),
        logits=torch.randn(batch, time, 2, 4),
    )
    prior = RSSMState(
        deter=torch.randn(batch, time, 8),
        stoch=torch.randn(batch, time, 2, 4),
        logits=torch.randn(batch, time, 2, 4),
    )
    dyn_kl_values = torch.rand(batch, time, 2)
    rep_kl_values = torch.rand(batch, time, 2)

    plain = world_model_loss(
        image_logits=image_logits,
        target_obs=target_obs,
        reward_logits=reward_logits,
        target_reward=target_reward,
        continuation_logits=continuation_logits,
        target_continue=target_continue,
        posterior=posterior,
        prior=prior,
        dyn_kl_values=dyn_kl_values,
        rep_kl_values=rep_kl_values,
        config=WorldModelLossConfig(kl_free_nats=0.0, image_edge_scale=0.0),
    )
    weighted = world_model_loss(
        image_logits=image_logits,
        target_obs=target_obs,
        reward_logits=reward_logits,
        target_reward=target_reward,
        continuation_logits=continuation_logits,
        target_continue=target_continue,
        posterior=posterior,
        prior=prior,
        dyn_kl_values=dyn_kl_values,
        rep_kl_values=rep_kl_values,
        config=WorldModelLossConfig(kl_free_nats=0.0, image_edge_scale=5.0),
    )

    assert torch.isfinite(weighted.image)
    assert weighted.image != plain.image
