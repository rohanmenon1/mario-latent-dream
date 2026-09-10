import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from mario_dreamer.baselines.ppo import PPOBaselineConfig, evaluate_ppo, ppo_model_kwargs


class _ConstantModel:
    def predict(self, obs, deterministic=True):
        assert deterministic is True
        return np.asarray([2]), None


class _ShortEpisodeVecEnv:
    num_envs = 1
    action_space = spaces.Discrete(7)

    def __init__(self):
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return np.zeros((1, 12, 96, 96), dtype=np.float32)

    def step(self, action):
        assert int(np.asarray(action).reshape(-1)[0]) == 2
        self.step_index += 1
        done = self.step_index == 3
        info = {
            "x_pos": 40 + 10 * self.step_index,
            "flag_get": done,
            "powerup_level": int(self.step_index >= 2),
        }
        return (
            np.zeros((1, 12, 96, 96), dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([done]),
            [info],
        )


def test_ppo_config_requires_exact_minibatches():
    config = PPOBaselineConfig(num_envs=2, n_steps=64, batch_size=48)
    with pytest.raises(ValueError, match="must divide"):
        config.validate()


def test_ppo_kwargs_are_explicit_and_use_float_images():
    config = PPOBaselineConfig(num_envs=2, n_steps=64, batch_size=32)
    kwargs = ppo_model_kwargs(config)

    assert kwargs["policy"] == "CnnPolicy"
    assert kwargs["n_steps"] == 64
    assert kwargs["batch_size"] == 32
    assert kwargs["policy_kwargs"] == {"normalize_images": False}


def test_evaluate_ppo_reports_progress_completion_powerups_and_actions():
    metrics = evaluate_ppo(
        _ConstantModel(),
        _ShortEpisodeVecEnv(),
        episodes=2,
        max_steps=10,
    )

    assert metrics["eval_return"] == 3.0
    assert metrics["eval_length"] == 3.0
    assert metrics["eval_max_x"] == 70.0
    assert metrics["eval_completion_rate"] == 1.0
    assert metrics["eval_powerup_rate"] == 1.0
    assert metrics["eval_action_2_count"] == 6.0
    assert metrics["eval_action_2_frac"] == 1.0
