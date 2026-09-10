import numpy as np
import pytest

from mario_dreamer.replay import EpisodeReplayBuffer


def obs(value: float) -> np.ndarray:
    return np.full((3, 4, 4), value, dtype=np.float32)


def add_episode(buffer: EpisodeReplayBuffer, length: int, start: int = 0, terminal: bool = True):
    for index in range(length):
        value = start + index
        buffer.add(
            obs(value),
            action=value % 7,
            reward=float(value),
            terminated=terminal and index == length - 1,
            truncated=False,
        )


def test_replay_samples_sequence_shapes():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=0)
    add_episode(buffer, length=8)

    batch = buffer.sample(batch_size=3, sequence_length=4)

    assert batch.obs.shape == (3, 4, 3, 4, 4)
    assert batch.actions.shape == (3, 4)
    assert batch.prev_actions.shape == (3, 4)
    assert batch.rewards.shape == (3, 4)
    assert batch.continues.shape == (3, 4)
    assert batch.is_first.shape == (3, 4)
    assert batch.obs.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.prev_actions.dtype == np.int64


def test_replay_stores_normalized_images_as_bytes_and_decodes_for_training():
    buffer = EpisodeReplayBuffer(capacity_steps=10, seed=9)
    expected = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(3, 4, 4)
    for index in range(4):
        buffer.add(
            expected,
            action=index,
            reward=0.0,
            terminated=index == 3,
        )

    assert buffer.episodes[0]["obs"].dtype == np.uint8
    batch = buffer.sample(batch_size=1, sequence_length=4)
    assert batch.obs.dtype == np.float32
    assert np.allclose(batch.obs[0, 0], expected, atol=1.0 / 255.0)


def test_replay_returns_state_indexed_rewards_and_previous_actions():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=0)
    for index in range(6):
        buffer.add_state(
            obs=obs(index),
            action=index % 7,
            prev_action=(index - 1) % 7 if index > 0 else 0,
            reward=float(index - 1) if index > 0 else 0.0,
            continue_=1.0,
            terminal=False,
        )
    buffer.finish_episode()

    batch = buffer.sample(batch_size=1, sequence_length=4)
    values = batch.obs[0, :, 0, 0, 0].astype(np.int64)

    expected_prev_actions = np.zeros(4, dtype=np.int64)
    expected_rewards = np.zeros(4, dtype=np.float32)
    for index, value in enumerate(values):
        if value > 0:
            expected_prev_actions[index] = (value - 1) % 7
            expected_rewards[index] = float(value - 1)

    assert batch.prev_actions[0].tolist() == expected_prev_actions.tolist()
    assert batch.rewards[0].tolist() == expected_rewards.tolist()


def test_terminal_transition_has_zero_continue():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=1)
    add_episode(buffer, length=4)

    batch = buffer.sample(batch_size=1, sequence_length=4)

    assert batch.continues[0, -1] == 0.0
    assert np.all(batch.continues[0, :-1] == 1.0)


def test_truncated_episode_does_not_zero_continue():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=2)
    for index in range(4):
        buffer.add(
            obs(index),
            action=index,
            reward=float(index),
            terminated=False,
            truncated=index == 3,
        )

    batch = buffer.sample(batch_size=1, sequence_length=4)

    assert np.all(batch.continues == 1.0)


def test_sampling_crosses_real_episode_boundaries_with_reset_marker():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=3)
    add_episode(buffer, length=3, start=0)
    add_episode(buffer, length=3, start=100)

    observed_crossing = False
    for _ in range(20):
        batch = buffer.sample(batch_size=1, sequence_length=3)
        values = batch.obs[0, :, 0, 0, 0]
        if 100.0 in values and values[0] < 100.0:
            reset_index = int(np.flatnonzero(values == 100.0)[0])
            assert batch.is_first[0, reset_index] == 1.0
            observed_crossing = True
    assert observed_crossing


def test_sampling_crosses_artificial_storage_chunks_without_reset():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=30)
    for index in range(3):
        buffer.add_state(obs(index), index, 0, 0.0, 1.0, False)
    buffer.finish_episode()
    for index in range(3, 6):
        buffer.add_state(obs(index), index, 0, 0.0, 1.0, False)
    buffer.finish_episode()

    batch = buffer.sample(batch_size=1, sequence_length=6)

    assert batch.obs[0, :, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert batch.is_first[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_capacity_trims_oldest_complete_episodes():
    buffer = EpisodeReplayBuffer(capacity_steps=5, seed=4)
    add_episode(buffer, length=3, start=0)
    add_episode(buffer, length=3, start=100)

    assert buffer.num_complete_episodes == 1
    assert len(buffer) == 3
    batch = buffer.sample(batch_size=1, sequence_length=3)
    assert batch.obs[0, :, 0, 0, 0].tolist() == [100.0, 101.0, 102.0]


def test_raises_when_no_episode_is_long_enough():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=5)
    add_episode(buffer, length=2)

    with pytest.raises(ValueError, match="fewer than 3 replay steps"):
        buffer.sample(batch_size=1, sequence_length=3)


def test_terminal_states_are_not_starved_by_sequence_sampling():
    buffer = EpisodeReplayBuffer(capacity_steps=1000, seed=31)
    for episode in range(20):
        add_episode(buffer, length=10, start=episode * 20)

    batch = buffer.sample(batch_size=1000, sequence_length=8)
    sampled_terminal_fraction = float((batch.continues < 0.5).mean())

    assert sampled_terminal_fraction == pytest.approx(0.1, abs=0.02)


def test_loads_old_replay_state_without_explicit_reset_markers():
    original = EpisodeReplayBuffer(capacity_steps=100, seed=32)
    add_episode(original, length=3)
    add_episode(original, length=3, start=100)
    state = original.state_dict()
    for episode in state["episodes"]:
        episode.pop("resets")
    state["active"].pop("resets")

    restored = EpisodeReplayBuffer(capacity_steps=1, seed=0)
    restored.load_state_dict(state)
    batch = restored.sample(batch_size=1, sequence_length=6)

    assert batch.is_first[0].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_sampling_with_replacement_does_not_require_one_episode_per_batch_item():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=6)
    add_episode(buffer, length=4)

    assert buffer.can_sample(batch_size=8, sequence_length=4)
    batch = buffer.sample(batch_size=8, sequence_length=4)
    assert batch.obs.shape[0] == 8


def test_informative_sampling_prefers_eventful_candidate_windows():
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=7)
    for index in range(4):
        buffer.add(
            obs(0.0),
            action=0,
            reward=0.0,
            terminated=index == 3,
        )
    for index in range(4):
        buffer.add(
            obs(float(index % 2)),
            action=index % 2,
            reward=10.0 if index == 2 else 0.0,
            terminated=index == 3,
        )

    batch = buffer.sample(
        batch_size=1,
        sequence_length=4,
        informative_fraction=1.0,
        informative_candidates=32,
    )

    assert batch.rewards.max() == 10.0
    assert np.any(batch.actions[0, 1:] != batch.actions[0, :-1])


@pytest.mark.parametrize(
    ("fraction", "candidates", "message"),
    [(-0.1, 16, "informative_fraction"), (1.1, 16, "informative_fraction"), (0.5, 0, "informative_candidates")],
)
def test_informative_sampling_validates_configuration(fraction, candidates, message):
    buffer = EpisodeReplayBuffer(capacity_steps=100, seed=8)
    add_episode(buffer, length=4)

    with pytest.raises(ValueError, match=message):
        buffer.sample(
            batch_size=1,
            sequence_length=4,
            informative_fraction=fraction,
            informative_candidates=candidates,
        )
