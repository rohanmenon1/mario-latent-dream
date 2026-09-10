from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_dreamer.envs.preprocessing import FramePreprocessor, PreprocessConfig


@dataclass(frozen=True)
class MarioEnvConfig:
    env_id: str = "SuperMarioBros-1-1-v0"
    action_set: str = "simple"
    obs_size: int = 64
    grayscale: bool = False
    max_episode_steps: int | None = 18000


ACTION_SET_NAMES = ("right_only", "simple", "complex")
ACTION_SETS: dict[str, list[list[str]]] = {
    "right_only": [["NOOP"], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"]],
    "simple": [
        ["NOOP"],
        ["right"],
        ["right", "A"],
        ["right", "B"],
        ["right", "A", "B"],
        ["A"],
        ["left"],
    ],
    "complex": [],
}


class GymnasiumMarioAdapter(gym.Env):
    """Small compatibility layer around the older nes-py Gym API."""

    metadata = {"render_modes": []}

    def __init__(self, env: Any, preprocessor: FramePreprocessor) -> None:
        super().__init__()
        self.env = env
        self.preprocessor = preprocessor
        self.action_space = spaces.Discrete(int(env.action_space.n))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=preprocessor.output_shape,
            dtype=np.float32,
        )
        self.step_count = 0
        self.max_episode_steps: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)

        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        self.step_count = 0
        return self.preprocessor(obs), {}

    def step(self, action: int):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False

        self.step_count += 1
        if self.max_episode_steps is not None and self.step_count >= self.max_episode_steps:
            truncated = True

        processed = self.preprocessor(obs)
        return processed, np.float32(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


def make_mario_env(config: MarioEnvConfig) -> GymnasiumMarioAdapter:
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise ImportError(
            "Mario dependencies are missing. Install them with `py -m pip install -r requirements.txt`."
        ) from exc

    action_sets = {
        "right_only": RIGHT_ONLY,
        "simple": SIMPLE_MOVEMENT,
        "complex": COMPLEX_MOVEMENT,
    }
    if config.action_set not in action_sets:
        raise ValueError(f"action_set must be one of {ACTION_SET_NAMES}, got {config.action_set!r}")

    raw_env = gym_super_mario_bros.make(config.env_id)
    joypad_env = JoypadSpace(raw_env, action_sets[config.action_set])
    preprocessor = FramePreprocessor(
        PreprocessConfig(obs_size=config.obs_size, grayscale=config.grayscale)
    )
    adapter = GymnasiumMarioAdapter(joypad_env, preprocessor)
    adapter.max_episode_steps = config.max_episode_steps
    return adapter
