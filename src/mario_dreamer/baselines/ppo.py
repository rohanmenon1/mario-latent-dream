from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from mario_dreamer.envs import MarioEnvConfig, make_mario_env


@dataclass(frozen=True)
class PPOBaselineConfig:
    """Configuration for the model-free PPO comparison."""

    env_id: str = "SuperMarioBros-1-1-v0"
    action_set: str = "simple"
    obs_size: int = 96
    frame_stack: int = 4
    num_envs: int = 4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    max_episode_steps: int = 18000
    seed: int = 0

    def validate(self) -> None:
        if self.obs_size < 36:
            raise ValueError("obs_size must be at least 36 for the SB3 NatureCNN extractor")
        if self.frame_stack <= 0 or self.num_envs <= 0 or self.n_steps <= 0:
            raise ValueError("frame_stack, num_envs, and n_steps must be positive")
        if self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("batch_size and n_epochs must be positive")
        rollout_size = self.num_envs * self.n_steps
        if self.batch_size > rollout_size or rollout_size % self.batch_size != 0:
            raise ValueError(
                "batch_size must divide num_envs * n_steps exactly; "
                f"got {self.batch_size} and rollout size {rollout_size}"
            )
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma must be in (0,1] and gae_lambda must be in [0,1]")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must be in (0,1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_ppo_vec_env(config: PPOBaselineConfig, *, num_envs: int | None = None):
    """Create vectorized, monitored Mario environments with stacked CHW observations."""

    config.validate()
    count = config.num_envs if num_envs is None else int(num_envs)
    if count <= 0:
        raise ValueError("num_envs must be positive")

    def make_one(rank: int):
        def factory():
            env = make_mario_env(
                MarioEnvConfig(
                    env_id=config.env_id,
                    action_set=config.action_set,
                    obs_size=config.obs_size,
                    max_episode_steps=config.max_episode_steps,
                )
            )
            env.reset(seed=config.seed + rank)
            return Monitor(
                env,
                info_keywords=("x_pos", "flag_get", "status", "powerup_level"),
            )

        return factory

    vector_env = DummyVecEnv([make_one(rank) for rank in range(count)])
    return VecFrameStack(vector_env, n_stack=config.frame_stack, channels_order="first")


def ppo_model_kwargs(config: PPOBaselineConfig) -> dict[str, Any]:
    """Return explicit SB3 PPO arguments so run metadata captures every update choice."""

    config.validate()
    return {
        "policy": "CnnPolicy",
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "ent_coef": config.entropy_coef,
        "vf_coef": config.value_coef,
        "max_grad_norm": config.max_grad_norm,
        "policy_kwargs": {"normalize_images": False},
        "seed": config.seed,
    }


def evaluate_ppo(model, env, *, episodes: int, max_steps: int) -> dict[str, float]:
    """Evaluate a PPO policy and expose Mario progress alongside standard returns."""

    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if int(env.num_envs) != 1:
        raise ValueError("evaluate_ppo requires a single vectorized environment")

    returns: list[float] = []
    lengths: list[float] = []
    max_x_positions: list[float] = []
    completions: list[float] = []
    powerup_pickups: list[float] = []
    action_count = int(env.action_space.n)
    action_counts = np.zeros(action_count, dtype=np.int64)

    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        max_x = 0.0
        completed = 0.0
        max_powerup_level = 0.0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            action_index = int(np.asarray(action).reshape(-1)[0])
            action_counts[action_index] += 1
            obs, reward, done, infos = env.step(action)
            info = infos[0]
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            max_x = max(max_x, float(info.get("x_pos", 0.0)))
            completed = max(completed, float(bool(info.get("flag_get", False))))
            max_powerup_level = max(
                max_powerup_level,
                float(info.get("powerup_level", 0.0)),
            )
            if bool(np.asarray(done).reshape(-1)[0]):
                lengths.append(float(step + 1))
                break
        else:
            lengths.append(float(max_steps))

        returns.append(total_reward)
        max_x_positions.append(max_x)
        completions.append(completed)
        powerup_pickups.append(float(max_powerup_level > 0.0))

    metrics = {
        "eval_return": float(np.mean(returns)),
        "eval_length": float(np.mean(lengths)),
        "eval_max_x": float(np.mean(max_x_positions)),
        "eval_completion_rate": float(np.mean(completions)),
        "eval_powerup_rate": float(np.mean(powerup_pickups)),
    }
    total_actions = max(int(action_counts.sum()), 1)
    for index, count in enumerate(action_counts):
        metrics[f"eval_action_{index}_count"] = float(count)
        metrics[f"eval_action_{index}_frac"] = float(count / total_actions)
    return metrics
