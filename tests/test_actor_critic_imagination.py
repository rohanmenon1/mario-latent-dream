import torch

from mario_dreamer.models import Actor, ActorCriticConfig, Critic, WorldModel, WorldModelConfig
from mario_dreamer.training import imagine_rollout, lambda_returns, replay_transition_discounts
from mario_dreamer.training.trainer import DreamerTrainer
from mario_dreamer.utils.distributions import bins_to_scalar, symexp, symlog, two_hot_symlog


def small_world_model() -> WorldModel:
    return WorldModel(
        WorldModelConfig(
            action_dim=7,
            depth=4,
            embed_dim=32,
            deter_dim=32,
            stoch_classes=4,
            stoch_categories=4,
            hidden_dim=64,
        )
    )


def test_actor_and_critic_shapes_and_gradients():
    config = ActorCriticConfig(feature_dim=48, action_dim=7, hidden_dim=32)
    actor = Actor(config)
    critic = Critic(config)
    features = torch.randn(3, 5, 48)

    logits = actor(features)
    dist = actor.distribution(features)
    values = critic(features)
    actions = dist.sample()
    loss = -dist.log_prob(actions).mean() + values.square().mean()
    loss.backward()

    assert logits.shape == (3, 5, 7)
    assert actions.shape == (3, 5)
    assert values.shape == (3, 5, 1)
    assert any(param.grad is not None for param in actor.parameters())
    assert any(param.grad is not None for param in critic.parameters())


def test_discrete_actor_unimix_keeps_every_action_possible():
    config = ActorCriticConfig(feature_dim=4, action_dim=3, hidden_dim=8, unimix_ratio=0.01)
    actor = Actor(config)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.net[-1].bias.copy_(torch.tensor([100.0, -100.0, -100.0]))

    probabilities = actor.distribution(torch.zeros(1, 4)).probs

    assert torch.all(probabilities >= 0.01 / 3.0)


def test_critic_has_distributional_logits_and_scalar_value():
    critic = Critic(ActorCriticConfig(feature_dim=10, action_dim=3, hidden_dim=16, value_bins=31))
    features = torch.randn(2, 4, 10)

    logits = critic.logits(features)
    values = critic(features)

    assert logits.shape == (2, 4, 31)
    assert values.shape == (2, 4, 1)


def test_symlog_two_hot_round_trip_is_close_for_bin_center():
    x = torch.tensor([0.0])
    encoded = two_hot_symlog(x, bins=21, low=-10.0, high=10.0)
    logits = encoded.clamp_min(1e-6).log()
    decoded = bins_to_scalar(logits, low=-10.0, high=10.0)

    assert torch.allclose(symexp(symlog(x)), x)
    assert torch.allclose(decoded, x, atol=1e-5)


def test_imagine_rollout_shapes():
    torch.manual_seed(0)
    world_model = small_world_model()
    actor = Actor(ActorCriticConfig(feature_dim=world_model.rssm.feature_dim, action_dim=7, hidden_dim=32))
    start = world_model.rssm.initial(batch_size=2)

    imagined = imagine_rollout(world_model=world_model, actor=actor, start=start, horizon=5)

    assert imagined.states.deter.shape == (2, 6, 32)
    assert imagined.states.stoch.shape == (2, 6, 4, 4)
    assert imagined.features.shape == (2, 6, 48)
    assert imagined.actions.shape == (2, 5)
    assert imagined.action_logits.shape == (2, 5, 7)
    assert imagined.rewards.shape == (2, 6)
    assert imagined.continues.shape == (2, 6)
    assert torch.equal(imagined.features[:, 0], world_model.rssm.get_features(start))
    assert torch.all((imagined.continues >= 0.0) & (imagined.continues <= 1.0))


def test_arrival_rewards_target_pre_action_states():
    all_values = torch.tensor([[10.0, 20.0, 30.0]])
    arrival_rewards = torch.tensor([[999.0, 1.0, 2.0]])
    transition_continues = torch.tensor([[0.9, 0.9]])

    returns = lambda_returns(
        arrival_rewards[:, 1:],
        all_values[:, :-1],
        transition_continues,
        all_values[:, -1],
        lambda_=0.0,
    )

    assert torch.allclose(returns, torch.tensor([[1.0 + 0.9 * 20.0, 2.0 + 0.9 * 30.0]]))


def test_lambda_returns_matches_one_step_when_lambda_zero():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[10.0, 20.0, 30.0]])
    continues = torch.tensor([[0.9, 0.9, 0.9]])
    bootstrap = torch.tensor([40.0])

    returns = lambda_returns(rewards, values, continues, bootstrap, lambda_=0.0)

    expected = torch.tensor([[1.0 + 0.9 * 20.0, 2.0 + 0.9 * 30.0, 3.0 + 0.9 * 40.0]])
    assert torch.allclose(returns, expected)


def test_lambda_returns_matches_monte_carlo_when_lambda_one():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.zeros_like(rewards)
    continues = torch.tensor([[0.5, 0.5]])
    bootstrap = torch.tensor([4.0])

    returns = lambda_returns(rewards, values, continues, bootstrap, lambda_=1.0)

    expected_t1 = 2.0 + 0.5 * 4.0
    expected_t0 = 1.0 + 0.5 * expected_t1
    assert torch.allclose(returns, torch.tensor([[expected_t0, expected_t1]]))


def test_replay_transition_discounts_do_not_cross_environment_resets():
    continues = torch.ones(1, 4)
    is_first = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    discounts = replay_transition_discounts(continues, is_first, discount=0.99)

    assert torch.allclose(discounts, torch.tensor([[0.99, 0.0, 0.99]]))


def test_trajectory_weights_discount_later_imagined_losses():
    continues = torch.tensor([[0.9, 0.5, 0.0, 1.0]])

    weights = DreamerTrainer._trajectory_weights(continues)

    assert torch.allclose(weights, torch.tensor([[1.0, 0.9, 0.45, 0.0]]))
