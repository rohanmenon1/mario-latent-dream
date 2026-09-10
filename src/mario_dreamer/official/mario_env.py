from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import cv2
import elements
import embodied
import numpy as np


class MarioEmbodiedEnv(embodied.Env):
    """Super Mario Bros. adapter for the upstream DreamerV3 Embodied API."""

    def __init__(
        self,
        task: str = "1_1",
        *,
        size: tuple[int, int] | list[int] = (64, 64),
        repeat: int = 4,
        action_set: str = "right_only",
        max_episode_steps: int = 4500,
        log_render: bool = False,
        seed: int = 0,
    ) -> None:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        if repeat <= 0:
            raise ValueError("repeat must be positive")
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("size must contain two positive dimensions")
        if action_set not in ("right_only", "simple"):
            raise ValueError("action_set must be 'right_only' or 'simple'")

        world, stage = task.split("_", 1)
        env_id = f"SuperMarioBros-{int(world)}-{int(stage)}-v0"
        movement = RIGHT_ONLY if action_set == "right_only" else SIMPLE_MOVEMENT
        self._env = JoypadSpace(
            gym_super_mario_bros.make(env_id, disable_env_checker=True), movement
        )
        self._size = (int(size[0]), int(size[1]))
        self._repeat = int(repeat)
        self._max_episode_steps = int(max_episode_steps)
        self._log_render = bool(log_render)
        self._render_shape = tuple(self._env.observation_space.shape)
        self._seed = int(seed)
        self._episode_index = 0
        self._episode_steps = 0
        self._done = True
        self._last_info: dict = {}

    @functools.cached_property
    def obs_space(self):
        height, width = self._size
        spaces = {
            "image": elements.Space(np.uint8, (height, width, 3), 0, 255),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/x_pos": elements.Space(np.float32),
            "log/flag_get": elements.Space(np.float32),
        }
        if self._log_render:
            spaces["log/render"] = elements.Space(
                np.uint8, self._render_shape, 0, 255
            )
        return spaces

    @functools.cached_property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (), 0, int(self._env.action_space.n)),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if bool(action["reset"]) or self._done:
            observation, _ = _reset_compat(
                self._env, self._seed + self._episode_index
            )
            self._episode_index += 1
            self._episode_steps = 0
            self._done = False
            self._last_info = {}
            return self._observation(observation, 0.0, is_first=True)

        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None
        info = {}
        for _ in range(self._repeat):
            observation, reward, terminated, truncated, info = _step_compat(
                self._env, int(action["action"])
            )
            total_reward += float(reward)
            if terminated or truncated:
                break

        self._episode_steps += 1
        if self._max_episode_steps and self._episode_steps >= self._max_episode_steps:
            truncated = True
        self._done = bool(terminated or truncated)
        self._last_info = dict(info)
        assert observation is not None
        return self._observation(
            observation,
            total_reward,
            is_last=self._done,
            is_terminal=bool(terminated),
        )

    def _observation(
        self,
        image: np.ndarray,
        reward: float,
        *,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> dict[str, np.ndarray | np.float32 | bool]:
        height, width = self._size
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        result = {
            "image": np.asarray(resized, dtype=np.uint8),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log/x_pos": np.float32(self._last_info.get("x_pos", 0.0)),
            "log/flag_get": np.float32(bool(self._last_info.get("flag_get", False))),
        }
        if self._log_render:
            result["log/render"] = np.asarray(image, dtype=np.uint8)
        return result

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()


def _reset_compat(env: Any, seed: int) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Normalize legacy Gym and Gymnasium reset results."""

    try:
        result = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], Mapping)
    ):
        observation, info = result
        return observation, info
    return result, {}


def _step_compat(
    env: Any, action: int
) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
    """Normalize legacy Gym and Gymnasium step results."""

    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return observation, reward, terminated, truncated, info
    raise RuntimeError(f"Mario step returned {len(result)} values; expected 4 or 5")
