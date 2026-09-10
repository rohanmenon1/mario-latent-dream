import torch
import torch.nn.functional as F

from mario_dreamer.models import WorldModel, WorldModelConfig
from mario_dreamer.training import WorldModelLossConfig


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
        ),
        loss_config=WorldModelLossConfig(kl_free_nats=0.0),
    )


def synthetic_batch(batch_size: int = 2, time_steps: int = 3):
    obs = torch.rand(batch_size, time_steps, 3, 64, 64)
    actions = torch.randint(0, 7, (batch_size, time_steps))
    rewards = torch.randn(batch_size, time_steps)
    continues = torch.ones(batch_size, time_steps)
    return obs, actions, rewards, continues


def test_world_model_forward_shapes_and_gradients():
    model = small_world_model()
    obs, actions, rewards, continues = synthetic_batch()

    output = model(obs, actions, rewards, continues)
    assert output.losses is not None
    output.losses.total.backward()

    assert output.features.shape == (2, 3, 48)
    assert output.image_logits.shape == (2, 3, 3, 64, 64)
    assert output.reward.shape == (2, 3, 1)
    assert output.continuation.shape == (2, 3, 1)
    assert output.losses.total.ndim == 0
    assert any(param.grad is not None for param in model.parameters())


def test_world_model_observes_with_previous_actions():
    model = small_world_model()
    obs = torch.rand(1, 3, 3, 64, 64)
    actions = torch.tensor([[1, 2, 3]])
    captured = {}
    original_observe = model.rssm.observe

    def observe_with_capture(embeds, action_inputs, start=None):
        captured["actions"] = action_inputs.detach().clone()
        return original_observe(embeds, action_inputs, start)

    model.rssm.observe = observe_with_capture

    model(obs, actions)

    expected = torch.zeros(1, 3, model.config.action_dim)
    expected[:, 1:] = F.one_hot(actions[:, :-1], num_classes=model.config.action_dim).float()
    assert torch.equal(captured["actions"], expected)


def test_world_model_accepts_explicit_previous_actions():
    model = small_world_model()
    obs = torch.rand(1, 3, 3, 64, 64)
    actions = torch.tensor([[1, 2, 3]])
    prev_actions = torch.tensor([[0, 4, 5]])
    captured = {}
    original_observe = model.rssm.observe

    def observe_with_capture(embeds, action_inputs, start=None):
        captured["actions"] = action_inputs.detach().clone()
        return original_observe(embeds, action_inputs, start)

    model.rssm.observe = observe_with_capture

    model(obs, actions, prev_actions=prev_actions)

    expected = F.one_hot(prev_actions, num_classes=model.config.action_dim).float()
    assert torch.equal(captured["actions"], expected)


def test_world_model_forwards_replay_reset_markers_to_rssm():
    model = small_world_model()
    obs = torch.rand(1, 3, 3, 64, 64)
    actions = torch.tensor([[1, 2, 3]])
    is_first = torch.tensor([[0.0, 1.0, 0.0]])
    captured = {}
    original_observe = model.rssm.observe

    def observe_with_capture(embeds, action_inputs, start=None, is_first=None):
        captured["is_first"] = is_first.detach().clone()
        return original_observe(embeds, action_inputs, start, is_first)

    model.rssm.observe = observe_with_capture
    model(obs, actions, is_first=is_first)

    assert torch.equal(captured["is_first"], is_first)


def test_world_model_can_reduce_loss_on_tiny_fixed_batch():
    torch.manual_seed(0)
    model = small_world_model()
    obs, actions, rewards, continues = synthetic_batch(batch_size=1, time_steps=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        output = model(obs, actions, rewards, continues)
        assert output.losses is not None
        output.losses.total.backward()
        optimizer.step()
        losses.append(float(output.losses.total.detach()))

    assert losses[-1] < losses[0]
