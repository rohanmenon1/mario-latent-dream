import torch

from mario_dreamer.models import Actor, ActorCriticConfig, Critic, RSSMState
from mario_dreamer.models.layers import RMSNorm
from mario_dreamer.training import WorldModelLossConfig, world_model_loss
from mario_dreamer.utils.optim import DreamerOptimizer, adaptive_gradient_clip


def _state(batch: int, time: int) -> RSSMState:
    return RSSMState(
        deter=torch.zeros(batch, time, 8),
        stoch=torch.zeros(batch, time, 2, 4),
        logits=torch.zeros(batch, time, 2, 4),
    )


def test_rms_norm_preserves_unit_root_mean_square_at_initialization():
    norm = RMSNorm(4)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = norm(value)

    assert torch.allclose(output.square().mean(-1), torch.ones(1), atol=2e-4)


def test_actor_starts_near_uniform_and_critic_starts_at_zero():
    config = ActorCriticConfig(feature_dim=12, action_dim=7, hidden_dim=16, value_bins=255)
    actor = Actor(config)
    critic = Critic(config)
    features = torch.randn(32, 12)

    probabilities = actor.distribution(features).probs
    values = critic(features)

    assert (probabilities - 1.0 / 7).abs().max() < 0.02
    assert torch.allclose(values, torch.zeros_like(values), atol=1e-6)


def test_image_reconstruction_is_pixel_summed_mse_after_sigmoid():
    image_logits = torch.zeros(1, 1, 3, 4, 4, requires_grad=True)
    state = _state(1, 1)
    losses = world_model_loss(
        image_logits=image_logits,
        target_obs=torch.zeros_like(image_logits),
        reward_logits=torch.zeros(1, 1, 5, requires_grad=True),
        target_reward=torch.zeros(1, 1),
        continuation_logits=torch.zeros(1, 1, 1, requires_grad=True),
        target_continue=torch.ones(1, 1),
        posterior=state,
        prior=state,
        dyn_kl_values=torch.zeros(1, 1, 2),
        rep_kl_values=torch.zeros(1, 1, 2),
        config=WorldModelLossConfig(kl_free_nats=0.0),
    )

    assert torch.allclose(losses.image, torch.tensor(12.0))


def test_dreamer_optimizer_uses_warmup_and_rms_momentum_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DreamerOptimizer([parameter], lr=0.1, warmup_steps=10, agc=0.0)
    parameter.grad = torch.tensor([1.0])

    optimizer.step()

    state = optimizer.state[parameter]
    assert state["step"] == 1
    assert "rms" in state and "momentum" in state
    assert torch.allclose(parameter.detach(), torch.tensor([0.99]), atol=1e-6)


def test_agc_uses_the_norm_of_the_whole_parameter_tensor():
    parameter = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    gradient = torch.tensor([[10.0, 0.0], [0.0, 0.0]])

    clipped = adaptive_gradient_clip(parameter, gradient, clipping=0.3)

    assert torch.allclose(clipped, gradient * 0.15)
