import math

from mario_dreamer.models import WorldModelConfig
from mario_dreamer.training.trainer import DreamerTrainer, TrainerConfig
from mario_dreamer.training.world_model_loss import WorldModelLossConfig


def small_trainer(**trainer_overrides) -> DreamerTrainer:
    return DreamerTrainer(
        TrainerConfig(
            seed=0,
            replay_capacity=100,
            batch_size=1,
            sequence_length=4,
            **trainer_overrides,
        ),
        WorldModelConfig(
            action_dim=7,
            depth=4,
            embed_dim=32,
            deter_dim=32,
            stoch_classes=4,
            stoch_categories=4,
            hidden_dim=64,
        ),
        WorldModelLossConfig(kl_free_nats=0.0),
    )


def test_collect_with_exploration_logs_action_metrics():
    trainer = small_trainer()
    try:
        metrics = trainer.collect_with_exploration(steps=10, epsilon=1.0)

        assert metrics["collect_epsilon"] == 1.0
        assert metrics["collect_random_action_steps"] == 10.0
        assert "collect_action_0_name" in metrics
        assert "collect_action_0_frac" in metrics
        assert sum(metrics[f"collect_action_{idx}_count"] for idx in range(7)) == 10.0
    finally:
        trainer.close()


def test_collect_with_exploration_rejects_invalid_epsilon():
    trainer = small_trainer()
    try:
        try:
            trainer.collect_with_exploration(steps=1, epsilon=1.1)
        except ValueError as exc:
            assert "epsilon" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    finally:
        trainer.close()


def test_persistent_exploration_holds_random_actions_for_configured_chunks():
    trainer = small_trainer(exploration_repeat_min=3, exploration_repeat_max=3)
    try:
        metrics = trainer.collect_with_exploration(steps=10, epsilon=1.0)

        assert metrics["collect_random_action_steps"] == 10.0
        assert metrics["collect_random_action_chunks"] == 4.0
        assert metrics["collect_exploration_repeat_min"] == 3.0
        assert metrics["collect_exploration_repeat_max"] == 3.0
        assert 0.0 <= metrics["collect_reward_nonzero_frac"] <= 1.0
        assert 0.0 <= metrics["collect_action_change_frac"] <= 1.0
        assert metrics["collect_max_x"] >= 0.0
    finally:
        trainer.close()


def test_informative_replay_runs_world_model_and_behavior_updates_end_to_end():
    trainer = small_trainer(
        informative_replay_fraction=0.5,
        informative_replay_candidates=4,
        imagination_horizon=3,
    )
    try:
        trainer.collect_with_exploration(steps=12, epsilon=1.0)
        trainer.replay.finish_episode()

        metrics = {
            **trainer.train_world_model(updates=1),
            **trainer.train_actor_critic(updates=1),
        }

        assert all(math.isfinite(value) for value in metrics.values())
        assert "critic_prediction_loss" in metrics
        assert "critic_slow_reg" in metrics
        assert "critic_abs_error" in metrics
        assert metrics["imagination_starts"] == 4.0
        assert metrics["imagination_transitions"] == 12.0
        assert "wm_replay_visual_change" in metrics
        assert "wm_replay_reward_nonzero_frac" in metrics
    finally:
        trainer.close()


def test_joint_update_reports_world_model_imagination_and_replay_value_losses():
    trainer = small_trainer(imagination_horizon=3, optimizer_warmup_steps=0)
    try:
        trainer.collect_with_exploration(steps=12, epsilon=1.0)
        trainer.replay.finish_episode()

        metrics = trainer.train_joint(updates=1)

        assert all(math.isfinite(value) for value in metrics.values())
        assert "wm_total" in metrics
        assert "actor_loss" in metrics
        assert "critic_prediction_loss" in metrics
        assert "critic_replay_value_loss" in metrics
        assert "entropy" in metrics
        assert "weighted_entropy" in metrics
        assert "wm_replay_terminal_frac" in metrics
        assert "wm_replay_reset_frac" in metrics
    finally:
        trainer.close()


def test_joint_world_model_receives_discounted_continuation_targets():
    trainer = small_trainer(imagination_horizon=3, optimizer_warmup_steps=0)
    try:
        trainer.collect_with_exploration(steps=12, epsilon=1.0)
        trainer.replay.finish_episode()
        captured = {}
        original_forward = trainer.world_model.forward

        def forward_with_capture(obs, actions, rewards=None, continues=None, **kwargs):
            captured["continues"] = continues.detach().clone()
            return original_forward(obs, actions, rewards, continues, **kwargs)

        trainer.world_model.forward = forward_with_capture
        trainer.train_joint(updates=1)

        assert captured["continues"].max().item() <= trainer.config.discount
    finally:
        trainer.close()
