from __future__ import annotations

import sys
import types

import numpy as np


def _load_compat_functions(monkeypatch):
    elements = types.ModuleType("elements")
    elements.Space = object
    embodied = types.ModuleType("embodied")
    embodied.Env = object
    monkeypatch.setitem(sys.modules, "elements", elements)
    monkeypatch.setitem(sys.modules, "embodied", embodied)

    from mario_dreamer.official.mario_env import _reset_compat, _step_compat

    return _reset_compat, _step_compat


def test_legacy_gym_api_is_normalized(monkeypatch):
    reset_compat, step_compat = _load_compat_functions(monkeypatch)
    observation = np.zeros((240, 256, 3), dtype=np.uint8)

    class LegacyEnv:
        def __init__(self):
            self.seed_value = None

        def reset(self):
            return observation

        def seed(self, value):
            self.seed_value = value

        def step(self, action):
            assert action == 2
            return observation, 1.5, True, {"x_pos": 41}

    env = LegacyEnv()
    reset_observation, reset_info = reset_compat(env, 51)
    step_observation, reward, terminated, truncated, info = step_compat(env, 2)

    assert reset_observation is observation
    assert reset_info == {}
    assert env.seed_value == 51
    assert step_observation is observation
    assert reward == 1.5
    assert terminated is True
    assert truncated is False
    assert info["x_pos"] == 41


def test_modern_api_and_time_limit_are_normalized(monkeypatch):
    reset_compat, step_compat = _load_compat_functions(monkeypatch)
    observation = np.zeros((240, 256, 3), dtype=np.uint8)

    class ModernEnv:
        def reset(self, *, seed):
            assert seed == 7
            return observation, {"seeded": True}

        def step(self, action):
            assert action == 1
            return observation, -1.0, False, True, {"x_pos": 40}

    reset_observation, reset_info = reset_compat(ModernEnv(), 7)
    step_observation, reward, terminated, truncated, info = step_compat(
        ModernEnv(), 1
    )

    assert reset_observation is observation
    assert reset_info == {"seeded": True}
    assert step_observation is observation
    assert reward == -1.0
    assert terminated is False
    assert truncated is True
    assert info["x_pos"] == 40
