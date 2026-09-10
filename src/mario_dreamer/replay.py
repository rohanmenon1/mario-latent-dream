from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    obs: np.ndarray
    actions: np.ndarray
    prev_actions: np.ndarray
    rewards: np.ndarray
    continues: np.ndarray
    is_first: np.ndarray


@dataclass
class _EpisodeBuilder:
    obs: list[np.ndarray]
    actions: list[int]
    prev_actions: list[int]
    rewards: list[float]
    continues: list[float]
    terminals: list[bool]
    resets: list[bool]

    @classmethod
    def empty(cls) -> "_EpisodeBuilder":
        return cls(
            obs=[],
            actions=[],
            prev_actions=[],
            rewards=[],
            continues=[],
            terminals=[],
            resets=[],
        )

    def append(
        self,
        obs: np.ndarray,
        action: int,
        prev_action: int,
        reward: float,
        continue_: float,
        terminal: bool,
        reset: bool,
    ) -> None:
        observation = np.asarray(obs)
        normalized_image = (
            np.issubdtype(observation.dtype, np.floating)
            and float(observation.min()) >= 0.0
            and float(observation.max()) <= 1.0
        )
        if self.obs and self.obs[0].dtype != np.uint8:
            stored_obs = observation.astype(np.float32)
        elif normalized_image:
            stored_obs = _encode_observation(observation)
        else:
            self.obs[:] = [_decode_observations(previous) for previous in self.obs]
            stored_obs = observation.astype(np.float32)
        self.obs.append(stored_obs)
        self.actions.append(int(action))
        self.prev_actions.append(int(prev_action))
        self.rewards.append(float(reward))
        self.continues.append(float(continue_))
        self.terminals.append(bool(terminal))
        self.resets.append(bool(reset))

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.prev_actions.clear()
        self.rewards.clear()
        self.continues.clear()
        self.terminals.clear()
        self.resets.clear()

    def to_episode(self) -> dict[str, np.ndarray]:
        return {
            "obs": np.stack(self.obs, axis=0),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "prev_actions": np.asarray(self.prev_actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "continues": np.asarray(self.continues, dtype=np.float32),
            "terminals": np.asarray(self.terminals, dtype=np.bool_),
            "resets": np.asarray(self.resets, dtype=np.bool_),
        }


class EpisodeReplayBuffer:
    """Episode-aware sequence replay buffer for Dreamer-style world-model training."""

    def __init__(self, capacity_steps: int, seed: int | None = None) -> None:
        if capacity_steps <= 0:
            raise ValueError("capacity_steps must be positive")
        self.capacity_steps = int(capacity_steps)
        self.rng = np.random.default_rng(seed)
        self.episodes: list[dict[str, np.ndarray]] = []
        self.active = _EpisodeBuilder.empty()
        self.num_steps = 0

    def __len__(self) -> int:
        return self.num_steps + len(self.active.obs)

    @property
    def num_complete_episodes(self) -> int:
        return len(self.episodes)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        terminal = bool(terminated)
        self.active.append(
            obs=obs,
            action=action,
            prev_action=0,
            reward=reward,
            continue_=1.0 - float(terminal),
            terminal=terminal,
            reset=terminal or bool(truncated),
        )
        if terminal or truncated:
            self._finish_active_episode()

    def add_state(
        self,
        obs: np.ndarray,
        action: int,
        prev_action: int,
        reward: float,
        continue_: float,
        terminal: bool,
        truncated: bool = False,
    ) -> None:
        self.active.append(
            obs=obs,
            action=action,
            prev_action=prev_action,
            reward=reward,
            continue_=continue_,
            terminal=terminal,
            reset=terminal or bool(truncated),
        )
        if terminal or truncated:
            self._finish_active_episode()

    def finish_episode(self) -> None:
        if self.active.obs:
            self._finish_active_episode()

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        if batch_size <= 0 or sequence_length <= 0:
            return False
        return len(self) >= sequence_length

    def state_dict(self) -> dict:
        return {
            "capacity_steps": self.capacity_steps,
            "episodes": self.episodes,
            "active": {
                "obs": self.active.obs,
                "actions": self.active.actions,
                "prev_actions": self.active.prev_actions,
                "rewards": self.active.rewards,
                "continues": self.active.continues,
                "terminals": self.active.terminals,
                "resets": self.active.resets,
            },
            "num_steps": self.num_steps,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity_steps = int(state["capacity_steps"])
        self.episodes = state["episodes"]
        active = state["active"]
        self.active = _EpisodeBuilder(
            obs=active["obs"],
            actions=active["actions"],
            prev_actions=active.get("prev_actions", [0] * len(active["actions"])),
            rewards=active["rewards"],
            continues=active.get("continues", [1.0] * len(active["actions"])),
            terminals=active["terminals"],
            resets=active.get("resets", list(active["terminals"])),
        )
        self.num_steps = int(state["num_steps"])
        self.rng.bit_generator.state = state["rng_state"]

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        informative_fraction: float = 0.0,
        informative_candidates: int = 16,
    ) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not 0.0 <= informative_fraction <= 1.0:
            raise ValueError("informative_fraction must be in [0, 1]")
        if informative_candidates <= 0:
            raise ValueError("informative_candidates must be positive")

        chunks = self._storage_chunks()
        total_steps = sum(len(chunk["actions"]) for chunk in chunks)
        if total_steps < sequence_length:
            raise ValueError(f"fewer than {sequence_length} replay steps are available")

        timeline = {
            key: np.concatenate(
                [
                    chunk.get(key, chunk["terminals"] if key == "resets" else None)
                    for chunk in chunks
                ],
                axis=0,
            )
            for key in ("actions", "prev_actions", "rewards", "continues", "resets")
        }
        is_first = np.zeros(total_steps, dtype=np.float32)
        is_first[0] = 1.0
        is_first[1:] = timeline["resets"][:-1].astype(np.float32)

        obs_chunks = []
        action_chunks = []
        prev_action_chunks = []
        reward_chunks = []
        continue_chunks = []
        is_first_chunks = []

        for _ in range(batch_size):
            if self.rng.random() < informative_fraction:
                start = self._informative_window(
                    chunks,
                    timeline,
                    total_steps,
                    sequence_length,
                    informative_candidates,
                )
            else:
                start = self._uniform_window(total_steps, sequence_length)
            end = start + sequence_length

            obs_chunks.append(self._slice_observations(chunks, start, end))
            action_chunks.append(timeline["actions"][start:end])
            prev_action_chunks.append(timeline["prev_actions"][start:end])
            reward_chunks.append(timeline["rewards"][start:end])
            continue_chunks.append(timeline["continues"][start:end])
            is_first_chunks.append(is_first[start:end])

        return ReplayBatch(
            obs=_decode_observations(np.stack(obs_chunks, axis=0)),
            actions=np.stack(action_chunks, axis=0).astype(np.int64),
            prev_actions=np.stack(prev_action_chunks, axis=0).astype(np.int64),
            rewards=np.stack(reward_chunks, axis=0).astype(np.float32),
            continues=np.stack(continue_chunks, axis=0).astype(np.float32),
            is_first=np.stack(is_first_chunks, axis=0).astype(np.float32),
        )

    def _uniform_window(
        self,
        total_steps: int,
        sequence_length: int,
    ) -> int:
        return int(self.rng.integers(0, total_steps - sequence_length + 1))

    def _informative_window(
        self,
        chunks: list[dict[str, np.ndarray]],
        timeline: dict[str, np.ndarray],
        total_steps: int,
        sequence_length: int,
        candidates: int,
    ) -> int:
        sampled = [
            self._uniform_window(total_steps, sequence_length) for _ in range(candidates)
        ]
        return max(
            sampled,
            key=lambda start: self._window_information_score(
                observations=self._slice_observations(
                    chunks,
                    start,
                    start + sequence_length,
                ),
                actions=timeline["actions"][start : start + sequence_length],
                rewards=timeline["rewards"][start : start + sequence_length],
                continues=timeline["continues"][start : start + sequence_length],
            ),
        )

    @staticmethod
    def _window_information_score(
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        continues: np.ndarray,
    ) -> float:
        reward_event = float(np.log1p(np.abs(rewards)).mean())
        terminal_event = float(np.any(continues < 0.5))
        action_change = (
            float(np.mean(actions[1:] != actions[:-1])) if len(actions) > 1 else 0.0
        )
        visual_change = 0.0
        if len(observations) > 1:
            height, width = observations.shape[-2:]
            pooled_height = height // 8
            pooled_width = width // 8
            if pooled_height and pooled_width:
                cropped = observations[..., : pooled_height * 8, : pooled_width * 8]
                spatial_sample = cropped.reshape(
                    *cropped.shape[:-2], pooled_height, 8, pooled_width, 8
                ).mean(axis=(-3, -1))
            else:
                spatial_sample = observations
            frame_delta = np.abs(spatial_sample[1:] - spatial_sample[:-1])
            visual_change = float(np.clip(frame_delta.mean(), 0.0, 1.0))

        return reward_event + terminal_event + action_change + 5.0 * visual_change

    def _finish_active_episode(self) -> None:
        episode = self.active.to_episode()
        self.episodes.append(episode)
        self.num_steps += len(episode["actions"])
        self.active.clear()
        self._trim_to_capacity()

    def _storage_chunks(self) -> list[dict[str, np.ndarray]]:
        chunks = list(self.episodes)
        if self.active.obs:
            chunks.append(self.active.to_episode())
        return chunks

    @staticmethod
    def _slice_observations(
        chunks: list[dict[str, np.ndarray]],
        start: int,
        end: int,
    ) -> np.ndarray:
        pieces = []
        offset = 0
        for chunk in chunks:
            chunk_end = offset + len(chunk["actions"])
            if start < chunk_end and end > offset:
                local_start = max(start - offset, 0)
                local_end = min(end - offset, len(chunk["actions"]))
                pieces.append(_decode_observations(chunk["obs"][local_start:local_end]))
            if chunk_end >= end:
                break
            offset = chunk_end
        if not pieces:
            raise RuntimeError("replay observation slice was empty")
        return np.concatenate(pieces, axis=0)

    def _trim_to_capacity(self) -> None:
        while self.num_steps > self.capacity_steps and self.episodes:
            removed = self.episodes.pop(0)
            self.num_steps -= len(removed["actions"])


def _encode_observation(obs: np.ndarray) -> np.ndarray:
    observation = np.asarray(obs)
    if np.issubdtype(observation.dtype, np.floating):
        minimum = float(observation.min())
        maximum = float(observation.max())
        if 0.0 <= minimum and maximum <= 1.0:
            return np.rint(observation * 255.0).astype(np.uint8)
    return observation.astype(np.float32)


def _decode_observations(observations: np.ndarray) -> np.ndarray:
    if observations.dtype == np.uint8:
        return observations.astype(np.float32) / 255.0
    return observations.astype(np.float32)
