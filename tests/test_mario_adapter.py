import numpy as np
from gymnasium import spaces

from mario_dreamer.envs.mario import GymnasiumMarioAdapter
from mario_dreamer.envs.preprocessing import FramePreprocessor, PreprocessConfig


class DummyActionSpace:
    n = 2

    def sample(self):
        return 1


class DummyOldGymEnv:
    action_space = DummyActionSpace()

    def __init__(self):
        self.closed = False

    def reset(self):
        return np.zeros((240, 256, 3), dtype=np.uint8)

    def step(self, action):
        del action
        obs = np.ones((240, 256, 3), dtype=np.uint8) * 255
        return obs, 1.0, False, {"x_pos": 1}

    def close(self):
        self.closed = True


def test_adapter_exposes_gymnasium_step_api():
    adapter = GymnasiumMarioAdapter(
        DummyOldGymEnv(),
        FramePreprocessor(PreprocessConfig(obs_size=32)),
    )
    adapter.max_episode_steps = 1

    obs, info = adapter.reset(seed=0)
    next_obs, reward, terminated, truncated, step_info = adapter.step(0)

    assert obs.shape == (3, 32, 32)
    assert adapter.observation_space == spaces.Box(
        low=0.0,
        high=1.0,
        shape=(3, 32, 32),
        dtype=np.float32,
    )
    assert isinstance(adapter.action_space, spaces.Discrete)
    assert adapter.action_space.n == 2
    assert info == {}
    assert next_obs.shape == (3, 32, 32)
    assert reward == np.float32(1.0)
    assert terminated is False
    assert truncated is True
    assert step_info["x_pos"] == 1
