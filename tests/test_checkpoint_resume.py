import numpy as np
import torch
import json

from mario_dreamer.models import WorldModelConfig
from mario_dreamer.training.trainer import DreamerTrainer, TrainerConfig
from mario_dreamer.training.world_model_loss import WorldModelLossConfig


def test_trainer_checkpoint_restores_replay_and_counters(tmp_path):
    config = TrainerConfig(
        seed=0,
        replay_capacity=100,
        batch_size=1,
        sequence_length=4,
    )
    world_config = WorldModelConfig(
        action_dim=7,
        depth=4,
        embed_dim=32,
        deter_dim=32,
        stoch_classes=4,
        stoch_categories=4,
        hidden_dim=64,
    )
    path = tmp_path / "checkpoint.pt"

    trainer = DreamerTrainer(config, world_config, WorldModelLossConfig(kl_free_nats=0.0))
    try:
        trainer.collect(12, random=True)
        trainer.replay.finish_episode()
        trainer.env_steps = 12
        trainer.episodes = 3
        trainer.episode_return = 4.5
        trainer.episode_length = 6
        trainer.save_checkpoint(path)
        metadata = json.loads(path.with_suffix(".pt.json").read_text(encoding="utf-8"))
        assert metadata["env_steps"] == 12
        assert metadata["replay_steps"] == 12
    finally:
        trainer.close()

    restored = DreamerTrainer(config, world_config, WorldModelLossConfig(kl_free_nats=0.0))
    try:
        restored.load_checkpoint(path)

        assert restored.env_steps == 12
        assert restored.episodes == 3
        assert restored.episode_return == 4.5
        assert restored.episode_length == 6
        assert len(restored.replay) == 12
        assert restored.replay.num_complete_episodes == 1
        assert isinstance(restored.obs, np.ndarray)
        assert restored.obs.shape == (3, 64, 64)
    finally:
        restored.close()


def test_reset_behavior_keeps_world_model_and_replay_but_resets_actor_critic():
    config = TrainerConfig(
        seed=1,
        replay_capacity=100,
        batch_size=1,
        sequence_length=4,
    )
    world_config = WorldModelConfig(
        action_dim=7,
        depth=4,
        embed_dim=32,
        deter_dim=32,
        stoch_classes=4,
        stoch_categories=4,
        hidden_dim=64,
    )

    trainer = DreamerTrainer(config, world_config, WorldModelLossConfig(kl_free_nats=0.0))
    try:
        trainer.collect(8, random=True)
        trainer.replay.finish_episode()
        old_world = {key: value.clone() for key, value in trainer.world_model.state_dict().items()}
        old_actor = {key: value.clone() for key, value in trainer.actor.state_dict().items()}
        old_critic = {key: value.clone() for key, value in trainer.critic.state_dict().items()}

        trainer.reset_behavior()

        assert len(trainer.replay) == 8
        assert all(
            torch.equal(value, trainer.world_model.state_dict()[key])
            for key, value in old_world.items()
        )
        assert any(
            not torch.equal(value, trainer.actor.state_dict()[key])
            for key, value in old_actor.items()
        )
        assert any(
            not torch.equal(value, trainer.critic.state_dict()[key])
            for key, value in old_critic.items()
        )
        assert float(trainer.return_scale) == 1.0
    finally:
        trainer.close()
