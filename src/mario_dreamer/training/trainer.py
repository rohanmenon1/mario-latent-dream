from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mario_dreamer.envs import ACTION_SETS, MarioEnvConfig, make_mario_env
from mario_dreamer.models import Actor, ActorCriticConfig, Critic, WorldModel, WorldModelConfig
from mario_dreamer.replay import EpisodeReplayBuffer
from mario_dreamer.training import (
    WorldModelLossConfig,
    imagine_rollout,
    lambda_returns,
    replay_transition_discounts,
)
from mario_dreamer.utils.distributions import two_hot_loss
from mario_dreamer.utils.optim import DreamerOptimizer


@dataclass(frozen=True)
class TrainerConfig:
    seed: int = 0
    env: MarioEnvConfig = MarioEnvConfig()
    replay_capacity: int = 10000
    batch_size: int = 4
    sequence_length: int = 16
    warmup_steps: int = 200
    imagination_horizon: int = 15
    discount: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 3e-4
    exploration_repeat_min: int = 1
    exploration_repeat_max: int = 1
    informative_replay_fraction: float = 0.0
    informative_replay_candidates: int = 16
    return_scale_decay: float = 0.99
    slow_critic_decay: float = 0.98
    slow_critic_reg_scale: float = 1.0
    replay_value_scale: float = 0.3
    world_model_lr: float = 4e-5
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5
    optimizer_warmup_steps: int = 1000
    device: str = "cpu"


class DreamerTrainer:
    """DreamerV3 trainer with compute-scalable model and replay settings."""

    def __init__(
        self,
        config: TrainerConfig,
        world_model_config: WorldModelConfig,
        loss_config: WorldModelLossConfig = WorldModelLossConfig(),
    ) -> None:
        self.config = config
        self.world_model_config = world_model_config
        self.loss_config = loss_config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.env = make_mario_env(config.env)
        self.replay = EpisodeReplayBuffer(config.replay_capacity, seed=config.seed)
        self.world_model = WorldModel(world_model_config, loss_config=loss_config).to(self.device)
        actor_config = ActorCriticConfig(
            feature_dim=self.world_model.rssm.feature_dim,
            action_dim=world_model_config.action_dim,
            hidden_dim=world_model_config.hidden_dim,
        )
        self.actor = Actor(actor_config).to(self.device)
        self.critic = Critic(actor_config).to(self.device)
        self.slow_critic = copy.deepcopy(self.critic).to(self.device)
        self.slow_critic.requires_grad_(False)
        self.actor_config = actor_config

        self.world_model_opt = DreamerOptimizer(
            self.world_model.parameters(),
            lr=config.world_model_lr,
            warmup_steps=config.optimizer_warmup_steps,
        )
        self.actor_opt = DreamerOptimizer(
            self.actor.parameters(), lr=config.actor_lr, warmup_steps=config.optimizer_warmup_steps
        )
        self.critic_opt = DreamerOptimizer(
            self.critic.parameters(), lr=config.critic_lr, warmup_steps=config.optimizer_warmup_steps
        )

        self.obs, _ = self.env.reset(seed=config.seed)
        self.policy_state = self.world_model.rssm.initial(1, self.device)
        self.prev_action = torch.zeros(1, self.env_action_dim, device=self.device)
        self.prev_action_index = 0
        self.state_reward = 0.0
        self.state_continue = 1.0
        self.env_steps = 0
        self.episodes = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.return_scale = torch.tensor(1.0, device=self.device)

    def close(self) -> None:
        self.env.close()

    def collect(self, steps: int, random: bool = False) -> dict[str, float]:
        return self.collect_with_exploration(steps, epsilon=1.0 if random else 0.0)

    def collect_with_exploration(self, steps: int, epsilon: float = 0.0) -> dict[str, float]:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        repeat_min = self.config.exploration_repeat_min
        repeat_max = self.config.exploration_repeat_max
        if repeat_min <= 0 or repeat_max < repeat_min:
            raise ValueError("exploration repeat range must satisfy 1 <= min <= max")
        completed_returns = []
        completed_lengths = []
        action_counts = np.zeros(self.env_action_dim, dtype=np.int64)
        collected_rewards = []
        visual_changes = []
        action_changes = 0
        previous_collected_action: int | None = None
        terminal_steps = 0
        max_x_position = 0.0
        random_action_steps = 0
        random_action_chunks = 0
        forced_action: int | None = None
        forced_steps_remaining = 0
        for _ in range(steps):
            if forced_steps_remaining == 0 and self.rng.random() < epsilon:
                forced_action = int(self.rng.integers(0, self.env_action_dim))
                forced_steps_remaining = int(self.rng.integers(repeat_min, repeat_max + 1))
                random_action_chunks += 1

            use_random = forced_steps_remaining > 0
            random_action_steps += int(use_random)
            action = self._select_action(
                random=False,
                forced_action=forced_action if use_random else None,
            )
            if use_random:
                forced_steps_remaining -= 1
                if forced_steps_remaining == 0:
                    forced_action = None
            action_counts[action] += 1
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            collected_rewards.append(float(reward))
            visual_changes.append(float(np.abs(next_obs - self.obs).mean()))
            action_changes += int(
                previous_collected_action is not None and action != previous_collected_action
            )
            previous_collected_action = action
            terminal_steps += int(terminated)
            max_x_position = max(max_x_position, float(info.get("x_pos", 0.0)))
            self.replay.add_state(
                obs=self.obs,
                action=action,
                prev_action=self.prev_action_index,
                reward=self.state_reward,
                continue_=self.state_continue,
                terminal=False,
            )

            self.env_steps += 1
            self.episode_return += float(reward)
            self.episode_length += 1

            if terminated or truncated:
                self.replay.add_state(
                    obs=next_obs,
                    action=0,
                    prev_action=action,
                    reward=float(reward),
                    continue_=0.0 if terminated else 1.0,
                    terminal=bool(terminated),
                    truncated=True,
                )
                completed_returns.append(self.episode_return)
                completed_lengths.append(self.episode_length)
                self.episodes += 1
                self.episode_return = 0.0
                self.episode_length = 0
                self.obs, _ = self.env.reset()
                self.state_reward = 0.0
                self.state_continue = 1.0
                self.prev_action_index = 0
                self._reset_policy_state()
                forced_action = None
                forced_steps_remaining = 0
            else:
                self.obs = next_obs
                self.prev_action_index = action
                self.state_reward = float(reward)
                self.state_continue = 1.0

        metrics = {
            "env_steps": float(self.env_steps),
            "episodes": float(self.episodes),
            "completed_episodes": float(len(completed_returns)),
            "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
            "replay_steps": float(len(self.replay)),
            "collect_epsilon": float(epsilon),
            "collect_random_action_steps": float(random_action_steps),
            "collect_random_action_chunks": float(random_action_chunks),
            "collect_reward_mean": float(np.mean(collected_rewards)) if collected_rewards else 0.0,
            "collect_reward_std": float(np.std(collected_rewards)) if collected_rewards else 0.0,
            "collect_reward_nonzero_frac": (
                float(np.mean(np.abs(collected_rewards) > 1e-6)) if collected_rewards else 0.0
            ),
            "collect_terminal_frac": float(terminal_steps / max(steps, 1)),
            "collect_visual_change_mean": (
                float(np.mean(visual_changes)) if visual_changes else 0.0
            ),
            "collect_visual_change_p90": (
                float(np.quantile(visual_changes, 0.9)) if visual_changes else 0.0
            ),
            "collect_action_change_frac": float(action_changes / max(steps - 1, 1)),
            "collect_max_x": float(max_x_position),
            "collect_exploration_repeat_min": float(repeat_min),
            "collect_exploration_repeat_max": float(repeat_max),
            "replay_informative_fraction": float(self.config.informative_replay_fraction),
            "replay_informative_candidates": float(self.config.informative_replay_candidates),
        }
        metrics.update(self._action_metrics("collect", action_counts))
        return metrics

    def train_world_model(self, updates: int = 1) -> dict[str, float]:
        metrics = []
        for _ in range(updates):
            obs, actions, prev_actions, rewards, continues, is_first = self._sample_batch()
            if obs.shape[1] > 1:
                replay_visual_change = (obs[:, 1:] - obs[:, :-1]).abs().mean()
                replay_action_change = (actions[:, 1:] != actions[:, :-1]).float().mean()
            else:
                replay_visual_change = torch.zeros((), device=self.device)
                replay_action_change = torch.zeros((), device=self.device)
            self.world_model_opt.zero_grad(set_to_none=True)
            output = self.world_model(
                obs,
                actions,
                rewards,
                self.config.discount * continues,
                prev_actions=prev_actions,
                is_first=is_first,
            )
            if output.losses is None:
                raise RuntimeError("world model did not return losses")
            output.losses.total.backward()
            self.world_model_opt.step()
            metrics.append(
                {
                    "wm_total": float(output.losses.total.detach()),
                    "wm_image": float(output.losses.image.detach()),
                    "wm_reward": float(output.losses.reward.detach()),
                    "wm_continuation": float(output.losses.continuation.detach()),
                    "wm_kl": float(output.losses.kl.detach()),
                    "wm_dyn_kl": float(output.losses.dyn_kl.detach()),
                    "wm_rep_kl": float(output.losses.rep_kl.detach()),
                    "wm_replay_reward_mean": float(rewards.mean().detach()),
                    "wm_replay_reward_std": float(rewards.std(unbiased=False).detach()),
                    "wm_replay_reward_nonzero_frac": float(
                        (rewards.abs() > 1e-6).float().mean().detach()
                    ),
                    "wm_replay_terminal_frac": float((continues < 0.5).float().mean().detach()),
                    "wm_replay_reset_frac": float(is_first[:, 1:].mean().detach()),
                    "wm_replay_visual_change": float(replay_visual_change.detach()),
                    "wm_replay_action_change_frac": float(replay_action_change.detach()),
                }
            )
        return self._mean_metrics(metrics)

    def train_actor_critic(
        self,
        updates: int = 1,
        *,
        include_world_model_loss: bool = False,
    ) -> dict[str, float]:
        metrics = []
        for _ in range(updates):
            (
                obs,
                actions,
                prev_actions,
                replay_rewards,
                replay_continues,
                replay_is_first,
            ) = self._sample_batch()
            wm_output = self.world_model(
                obs,
                actions,
                replay_rewards if include_world_model_loss else None,
                (
                    self.config.discount * replay_continues
                    if include_world_model_loss
                    else None
                ),
                prev_actions=prev_actions,
                is_first=replay_is_first,
            )
            start = self._all_states(wm_output.posterior)

            imagined = imagine_rollout(
                world_model=self.world_model,
                actor=self.actor,
                start=start,
                horizon=self.config.imagination_horizon,
            )
            all_values = self.critic(imagined.features).squeeze(-1)
            current_values = all_values[:, :-1]
            transition_rewards = imagined.rewards[:, 1:]
            transition_continues = imagined.continues[:, 1:]
            with torch.no_grad():
                continues = transition_continues.detach()
                bootstrap = all_values[:, -1].detach()
                returns = lambda_returns(
                    transition_rewards.detach(),
                    current_values.detach(),
                    continues,
                    bootstrap,
                    lambda_=self.config.lambda_,
                )
                replay_bootstrap_returns = returns[:, 0].reshape(obs.shape[:2])
                scale = self._update_return_scale(returns)
                advantages = (returns - current_values.detach()) / scale
                trajectory_weights = torch.cumprod(
                    imagined.continues[:, :-1].detach(),
                    dim=1,
                )

            dist = torch.distributions.Categorical(logits=imagined.action_logits)
            log_probs = dist.log_prob(imagined.actions)
            entropy_per_step = dist.entropy()
            weighted_entropy = self._weighted_mean(entropy_per_step, trajectory_weights)
            entropy = entropy_per_step.mean()
            actor_loss = -self._weighted_mean(
                log_probs * advantages,
                trajectory_weights,
            ) - self.config.entropy_scale * weighted_entropy

            critic_features = imagined.features[:, :-1].detach()
            critic_logits = self.critic.logits(critic_features)
            critic_prediction_loss = two_hot_loss(
                critic_logits,
                returns.detach(),
                reduction="none",
            )
            with torch.no_grad():
                slow_target = self.slow_critic(critic_features).squeeze(-1)
            slow_reg = two_hot_loss(
                critic_logits,
                slow_target,
                reduction="none",
            )
            critic_prediction_mean = self._weighted_mean(
                critic_prediction_loss, trajectory_weights
            )
            critic_slow_reg_mean = self._weighted_mean(slow_reg, trajectory_weights)
            replay_features = wm_output.features
            replay_critic_logits = self.critic.logits(replay_features[:, :-1])
            with torch.no_grad():
                replay_slow_values = self.slow_critic(replay_features.detach()).squeeze(-1)
                replay_returns = lambda_returns(
                    replay_rewards[:, 1:] * (1.0 - replay_is_first[:, 1:]),
                    replay_bootstrap_returns[:, :-1],
                    replay_transition_discounts(
                        replay_continues,
                        replay_is_first,
                        self.config.discount,
                    ),
                    replay_bootstrap_returns[:, -1],
                    lambda_=self.config.lambda_,
                )
            replay_prediction = two_hot_loss(
                replay_critic_logits,
                replay_returns,
                reduction="none",
            )
            replay_slow_reg = two_hot_loss(
                replay_critic_logits,
                replay_slow_values[:, :-1].detach(),
                reduction="none",
            )
            replay_value_loss = (replay_prediction + self.config.slow_critic_reg_scale * replay_slow_reg).mean()
            critic_loss = (
                critic_prediction_mean
                + self.config.slow_critic_reg_scale * critic_slow_reg_mean
                + self.config.replay_value_scale * replay_value_loss
            )
            total_loss = actor_loss + critic_loss
            if include_world_model_loss:
                if wm_output.losses is None:
                    raise RuntimeError("joint update did not produce world-model losses")
                total_loss = total_loss + wm_output.losses.total

            self.world_model_opt.zero_grad(set_to_none=True)
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()
            self.world_model_opt.step()
            self._update_slow_critic()

            row = {
                    "actor_loss": float(actor_loss.detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "critic_prediction_loss": float(critic_prediction_mean.detach()),
                    "critic_slow_reg": float(critic_slow_reg_mean.detach()),
                    "critic_replay_value_loss": float(replay_value_loss.detach()),
                    "critic_value_mean": float(
                        self._weighted_mean(current_values.detach(), trajectory_weights)
                    ),
                    "critic_target_mean": float(
                        self._weighted_mean(returns.detach(), trajectory_weights)
                    ),
                    "critic_abs_error": float(
                        self._weighted_mean(
                            (returns - current_values.detach()).abs(), trajectory_weights
                        )
                    ),
                    "imagined_reward": float(transition_rewards.mean().detach()),
                    "entropy": float(entropy.detach()),
                    "weighted_entropy": float(weighted_entropy.detach()),
                    "return_scale": float(self.return_scale.detach()),
                    "imagination_starts": float(start.deter.shape[0]),
                    "imagination_transitions": float(
                        start.deter.shape[0] * self.config.imagination_horizon
                    ),
                }
            if include_world_model_loss:
                losses = wm_output.losses
                assert losses is not None
                row.update(
                    {
                        "wm_total": float(losses.total.detach()),
                        "wm_image": float(losses.image.detach()),
                        "wm_reward": float(losses.reward.detach()),
                        "wm_continuation": float(losses.continuation.detach()),
                        "wm_kl": float(losses.kl.detach()),
                        "wm_dyn_kl": float(losses.dyn_kl.detach()),
                        "wm_rep_kl": float(losses.rep_kl.detach()),
                        "wm_replay_terminal_frac": float(
                            (replay_continues < 0.5).float().mean().detach()
                        ),
                        "wm_replay_reset_frac": float(
                            replay_is_first[:, 1:].mean().detach()
                        ),
                    }
                )
            metrics.append(row)
        return self._mean_metrics(metrics)

    def train_joint(self, updates: int = 1) -> dict[str, float]:
        """Optimize all DreamerV3 losses from the same uniformly sampled sequences."""

        return self.train_actor_critic(updates, include_world_model_loss=True)

    def evaluate(
        self,
        episodes: int = 1,
        max_steps: int = 1000,
        *,
        deterministic: bool = False,
    ) -> dict[str, float]:
        eval_env = make_mario_env(self.config.env)
        returns = []
        lengths = []
        max_x_positions = []
        completions = []
        action_counts = np.zeros(self.env_action_dim, dtype=np.int64)
        total_eval_steps = 0
        eval_generator = torch.Generator(device=self.device)
        eval_generator.manual_seed(self.config.seed + 1000)

        try:
            for episode in range(episodes):
                obs, _ = eval_env.reset(seed=self.config.seed + 1000 + episode)
                state = self.world_model.rssm.initial(1, self.device)
                prev_action = torch.zeros(1, self.env_action_dim, device=self.device)
                total_reward = 0.0
                max_x = 0
                completed = 0.0

                for step in range(max_steps):
                    obs_tensor = torch.from_numpy(obs[None]).to(self.device)
                    with torch.no_grad():
                        embed = self.world_model.encoder(obs_tensor)
                        posterior, _ = self.world_model.rssm.obs_step(state, prev_action, embed)
                        features = self.world_model.rssm.get_features(posterior)
                        if deterministic:
                            action = int(
                                torch.argmax(self.actor.logits(features), dim=-1).item()
                            )
                        else:
                            probabilities = self.actor.distribution(features).probs
                            action = int(
                                torch.multinomial(
                                    probabilities,
                                    num_samples=1,
                                    generator=eval_generator,
                                ).item()
                            )
                        action_counts[action] += 1
                        total_eval_steps += 1
                        prev_action = F.one_hot(
                            torch.tensor([action], device=self.device),
                            num_classes=self.env_action_dim,
                        ).float()
                        state = posterior

                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    total_reward += float(reward)
                    max_x = max(max_x, int(info.get("x_pos", 0)))
                    completed = max(completed, float(bool(info.get("flag_get", False))))
                    if terminated or truncated:
                        lengths.append(float(step + 1))
                        break
                else:
                    lengths.append(float(max_steps))

                returns.append(total_reward)
                max_x_positions.append(float(max_x))
                completions.append(completed)
        finally:
            eval_env.close()

        metrics = {
            "eval_return": float(np.mean(returns)) if returns else 0.0,
            "eval_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval_max_x": float(np.mean(max_x_positions)) if max_x_positions else 0.0,
            "eval_completion_rate": float(np.mean(completions)) if completions else 0.0,
            "eval_deterministic": float(deterministic),
        }
        metrics.update(self._action_metrics("eval", action_counts, total=total_eval_steps))
        return metrics

    def reset_behavior(self) -> None:
        """Reset actor, critic, slow critic, and behavior optimizers.

        Keeps the world model, world-model optimizer, replay buffer, env counters,
        and current observation intact.
        """

        self.actor = Actor(self.actor_config).to(self.device)
        self.critic = Critic(self.actor_config).to(self.device)
        self.slow_critic = copy.deepcopy(self.critic).to(self.device)
        self.slow_critic.requires_grad_(False)
        self.actor_opt = DreamerOptimizer(
            self.actor.parameters(),
            lr=self.config.actor_lr,
            warmup_steps=self.config.optimizer_warmup_steps,
        )
        self.critic_opt = DreamerOptimizer(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            warmup_steps=self.config.optimizer_warmup_steps,
        )
        self.return_scale = torch.tensor(1.0, device=self.device)
        self._reset_policy_state()
        self.prev_action_index = 0

    def set_loss_config(self, loss_config: WorldModelLossConfig) -> None:
        self.loss_config = loss_config
        self.world_model.loss_config = loss_config

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "config": self.config,
            "world_model_config": self.world_model_config,
            "loss_config": self.loss_config,
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "world_model_opt": self.world_model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "replay": self.replay.state_dict(),
            "obs": self.obs,
            "policy_state": {
                "deter": self.policy_state.deter.detach().cpu(),
                "stoch": self.policy_state.stoch.detach().cpu(),
                "logits": self.policy_state.logits.detach().cpu(),
            },
            "prev_action": self.prev_action.detach().cpu(),
            "prev_action_index": self.prev_action_index,
            "state_reward": self.state_reward,
            "state_continue": self.state_continue,
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "return_scale": float(self.return_scale.detach().cpu()),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

        metadata = {
            "schema_version": 1,
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "replay_steps": len(self.replay),
        }
        metadata_path = path.with_suffix(path.suffix + ".json")
        temporary_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_metadata_path.write_text(
            json.dumps(metadata, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_metadata_path.replace(metadata_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "loss_config" in checkpoint:
            self.loss_config = checkpoint["loss_config"]
            self.world_model.loss_config = self.loss_config
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "slow_critic" in checkpoint:
            self.slow_critic.load_state_dict(checkpoint["slow_critic"])
        else:
            self.slow_critic.load_state_dict(checkpoint["critic"])
        self.world_model_opt.load_state_dict(checkpoint["world_model_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
        if "replay" in checkpoint:
            self.replay.load_state_dict(checkpoint["replay"])
        if "obs" in checkpoint:
            self.obs = checkpoint["obs"]
        if "policy_state" in checkpoint:
            state = checkpoint["policy_state"]
            self.policy_state = type(self.policy_state)(
                deter=state["deter"].to(self.device),
                stoch=state["stoch"].to(self.device),
                logits=state["logits"].to(self.device),
            )
        else:
            self.policy_state = self.world_model.rssm.initial(1, self.device)
        if "prev_action" in checkpoint:
            self.prev_action = checkpoint["prev_action"].to(self.device)
        else:
            self.prev_action = torch.zeros(1, self.env_action_dim, device=self.device)
        self.prev_action_index = int(checkpoint.get("prev_action_index", 0))
        self.state_reward = float(checkpoint.get("state_reward", 0.0))
        self.state_continue = float(checkpoint.get("state_continue", 1.0))
        self.env_steps = int(checkpoint["env_steps"])
        self.episodes = int(checkpoint["episodes"])
        self.episode_return = float(checkpoint.get("episode_return", 0.0))
        self.episode_length = int(checkpoint.get("episode_length", 0))
        self.return_scale = torch.tensor(
            float(checkpoint.get("return_scale", 1.0)),
            device=self.device,
        )

    @property
    def env_action_dim(self) -> int:
        return int(self.env.action_space.n)

    def _select_action(self, random: bool, forced_action: int | None = None) -> int:
        obs = torch.from_numpy(self.obs[None, None]).to(self.device)
        obs = obs[:, 0]
        with torch.no_grad():
            embed = self.world_model.encoder(obs)
            posterior, _ = self.world_model.rssm.obs_step(
                self.policy_state,
                self.prev_action,
                embed,
            )
            self.policy_state = self._detach_state(posterior)
            features = self.world_model.rssm.get_features(self.policy_state)
            if forced_action is not None:
                action = int(forced_action)
            elif random or not self.replay.can_sample(1, self.config.sequence_length):
                action = int(self.rng.integers(0, self.env_action_dim))
            else:
                action = int(self.actor.distribution(features).sample().item())
            self.prev_action = F.one_hot(
                torch.tensor([action], device=self.device),
                num_classes=self.env_action_dim,
            ).float()
            return action

    def _action_metrics(
        self,
        prefix: str,
        action_counts: np.ndarray,
        total: int | None = None,
    ) -> dict[str, float | str]:
        if total is None:
            total = int(action_counts.sum())
        names = ACTION_SETS.get(self.config.env.action_set, [])
        metrics: dict[str, float | str] = {}
        for action, count in enumerate(action_counts):
            action_name = "+".join(names[action]) if action < len(names) else str(action)
            metrics[f"{prefix}_action_{action}_name"] = action_name
            metrics[f"{prefix}_action_{action}_count"] = float(count)
            metrics[f"{prefix}_action_{action}_frac"] = float(count / total) if total else 0.0
        return metrics

    def _sample_batch(self):
        batch = self.replay.sample(
            self.config.batch_size,
            self.config.sequence_length,
            informative_fraction=self.config.informative_replay_fraction,
            informative_candidates=self.config.informative_replay_candidates,
        )
        return (
            torch.from_numpy(batch.obs).to(self.device),
            torch.from_numpy(batch.actions).to(self.device),
            torch.from_numpy(batch.prev_actions).to(self.device),
            torch.from_numpy(batch.rewards).to(self.device),
            torch.from_numpy(batch.continues).to(self.device),
            torch.from_numpy(batch.is_first).to(self.device),
        )

    def _last_state(self, state):
        return self._detach_state(
            type(state)(
                deter=state.deter[:, -1],
                stoch=state.stoch[:, -1],
                logits=state.logits[:, -1],
            )
        )

    def _all_states(self, state):
        batch_size, sequence_length = state.deter.shape[:2]
        return self._detach_state(
            type(state)(
                deter=state.deter.reshape(batch_size * sequence_length, *state.deter.shape[2:]),
                stoch=state.stoch.reshape(batch_size * sequence_length, *state.stoch.shape[2:]),
                logits=state.logits.reshape(batch_size * sequence_length, *state.logits.shape[2:]),
            )
        )

    def _detach_state(self, state):
        return type(state)(
            deter=state.deter.detach(),
            stoch=state.stoch.detach(),
            logits=state.logits.detach(),
        )

    def _reset_policy_state(self) -> None:
        self.policy_state = self.world_model.rssm.initial(1, self.device)
        self.prev_action = torch.zeros(1, self.env_action_dim, device=self.device)

    def _mean_metrics(self, metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        keys = metrics[0].keys()
        return {key: float(np.mean([item[key] for item in metrics])) for key in keys}

    def _update_return_scale(self, returns: torch.Tensor) -> torch.Tensor:
        flat = returns.detach().flatten()
        if flat.numel() < 2:
            batch_scale = torch.tensor(1.0, device=self.device)
        else:
            low = torch.quantile(flat, 0.05)
            high = torch.quantile(flat, 0.95)
            batch_scale = (high - low).clamp_min(1.0)
        self.return_scale = (
            self.config.return_scale_decay * self.return_scale
            + (1.0 - self.config.return_scale_decay) * batch_scale
        ).detach()
        return self.return_scale.clamp_min(1.0)

    def _update_slow_critic(self) -> None:
        decay = self.config.slow_critic_decay
        with torch.no_grad():
            for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
                slow_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @staticmethod
    def _trajectory_weights(continues: torch.Tensor) -> torch.Tensor:
        first = torch.ones_like(continues[:, :1])
        return torch.cumprod(torch.cat([first, continues[:, :-1]], dim=1), dim=1)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).mean()
