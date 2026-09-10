from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mario_dreamer.models.layers import BlockLinear, RMSNorm, init_dreamer_module


@dataclass(frozen=True)
class RSSMConfig:
    action_dim: int
    embed_dim: int = 1024
    deter_dim: int = 1024
    stoch_classes: int = 32
    stoch_categories: int = 32
    hidden_dim: int = 1024
    unimix_ratio: float = 0.01
    blocks: int = 8


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


class RSSM(nn.Module):
    """Discrete recurrent state-space model."""

    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        self.config = config
        if config.deter_dim % config.blocks:
            raise ValueError("deter_dim must be divisible by RSSM blocks")
        stoch_dim = config.stoch_classes * config.stoch_categories
        self.deter_input = self._normalized_linear(config.deter_dim, config.hidden_dim)
        self.stoch_input = self._normalized_linear(stoch_dim, config.hidden_dim)
        self.action_input = self._normalized_linear(config.action_dim, config.hidden_dim)
        core_input_dim = config.deter_dim + config.blocks * 3 * config.hidden_dim
        self.core_hidden = nn.Sequential(
            BlockLinear(core_input_dim, config.deter_dim, config.blocks),
            RMSNorm(config.deter_dim),
            nn.SiLU(),
        )
        self.core_gates = BlockLinear(
            config.deter_dim,
            3 * config.deter_dim,
            config.blocks,
        )
        self.prior = nn.Sequential(
            nn.Linear(config.deter_dim, config.hidden_dim),
            RMSNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            RMSNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, stoch_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(config.deter_dim + config.embed_dim, config.hidden_dim),
            RMSNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, stoch_dim),
        )
        self.apply(init_dreamer_module)

    @property
    def stoch_dim(self) -> int:
        return self.config.stoch_classes * self.config.stoch_categories

    @property
    def feature_dim(self) -> int:
        return self.config.deter_dim + self.stoch_dim

    def initial(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        if device is None:
            device = next(self.parameters()).device
        deter = torch.zeros(batch_size, self.config.deter_dim, device=device)
        logits = torch.zeros(
            batch_size,
            self.config.stoch_classes,
            self.config.stoch_categories,
            device=device,
        )
        stoch = torch.zeros_like(logits)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def get_features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch.flatten(start_dim=-2)], dim=-1)

    def img_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        if action.ndim != 2:
            raise ValueError(f"expected action shape [B,A], got {tuple(action.shape)}")
        action = action / action.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        deter = self.deter_input(prev_state.deter)
        stoch = self.stoch_input(prev_state.stoch.flatten(start_dim=-2))
        action_embed = self.action_input(action)
        joined = torch.cat([deter, stoch, action_embed], dim=-1)
        repeated = joined.unsqueeze(-2).expand(*joined.shape[:-1], self.config.blocks, -1)
        grouped_deter = prev_state.deter.reshape(
            *prev_state.deter.shape[:-1],
            self.config.blocks,
            self.config.deter_dim // self.config.blocks,
        )
        core_input = torch.cat([grouped_deter, repeated], dim=-1).flatten(start_dim=-2)
        gates = self.core_gates(self.core_hidden(core_input))
        reset, candidate, update = gates.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1.0)
        deter = update * candidate + (1.0 - update) * prev_state.deter
        logits = self.prior(deter).view(
            -1,
            self.config.stoch_classes,
            self.config.stoch_categories,
        )
        stoch = self._sample_stoch(logits)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def obs_step(self, prev_state: RSSMState, action: torch.Tensor, embed: torch.Tensor):
        prior = self.img_step(prev_state, action)
        x = torch.cat([prior.deter, embed], dim=-1)
        logits = self.posterior(x).view(
            -1,
            self.config.stoch_classes,
            self.config.stoch_categories,
        )
        stoch = self._sample_stoch(logits)
        posterior = RSSMState(deter=prior.deter, stoch=stoch, logits=logits)
        return posterior, prior

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        start: RSSMState | None = None,
        is_first: torch.Tensor | None = None,
    ):
        if embeds.ndim != 3:
            raise ValueError(f"expected embeds shape [B,T,E], got {tuple(embeds.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"expected actions shape [B,T,A], got {tuple(actions.shape)}")
        batch_size, time_steps, _ = embeds.shape
        if is_first is not None and is_first.shape != (batch_size, time_steps):
            raise ValueError(
                f"expected is_first shape {(batch_size, time_steps)}, got {tuple(is_first.shape)}"
            )
        state = self.initial(batch_size, embeds.device) if start is None else start

        posteriors = []
        priors = []
        for index in range(time_steps):
            action = actions[:, index]
            if is_first is not None:
                keep = (1.0 - is_first[:, index].to(embeds.dtype)).unsqueeze(-1)
                state = RSSMState(
                    deter=state.deter * keep,
                    stoch=state.stoch * keep.unsqueeze(-1),
                    logits=state.logits * keep.unsqueeze(-1),
                )
                action = action * keep
            state, prior = self.obs_step(state, action, embeds[:, index])
            posteriors.append(state)
            priors.append(prior)
        return self._stack_states(posteriors), self._stack_states(priors)

    def imagine(self, actions: torch.Tensor, start: RSSMState) -> RSSMState:
        if actions.ndim != 3:
            raise ValueError(f"expected actions shape [B,T,A], got {tuple(actions.shape)}")
        state = start
        priors = []
        for index in range(actions.shape[1]):
            state = self.img_step(state, actions[:, index])
            priors.append(state)
        return self._stack_states(priors)

    def kl_loss(self, posterior: RSSMState, prior: RSSMState) -> torch.Tensor:
        return self._kl_from_logits(posterior.logits, prior.logits)

    def kl_losses(self, posterior: RSSMState, prior: RSSMState) -> tuple[torch.Tensor, torch.Tensor]:
        dyn = self._kl_from_logits(posterior.logits.detach(), prior.logits)
        rep = self._kl_from_logits(posterior.logits, prior.logits.detach())
        return dyn, rep

    def _sample_stoch(self, logits: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.OneHotCategoricalStraightThrough(probs=self._probs(logits))
        return dist.rsample()

    @staticmethod
    def _normalized_linear(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            RMSNorm(output_dim),
            nn.SiLU(),
        )

    def _mode_stoch(self, logits: torch.Tensor) -> torch.Tensor:
        index = logits.argmax(dim=-1)
        return F.one_hot(index, self.config.stoch_categories).to(logits.dtype)

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        uniform = torch.full_like(probs, 1.0 / self.config.stoch_categories)
        return (1.0 - self.config.unimix_ratio) * probs + self.config.unimix_ratio * uniform

    def _kl_from_logits(self, posterior_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        post_prob = self._probs(posterior_logits)
        prior_prob = self._probs(prior_logits)
        post_logprob = torch.log(post_prob.clamp_min(1e-8))
        prior_logprob = torch.log(prior_prob.clamp_min(1e-8))
        per_categorical = (post_prob * (post_logprob - prior_logprob)).sum(dim=-1)
        return per_categorical.sum(dim=-1)

    def _stack_states(self, states: list[RSSMState]) -> RSSMState:
        return RSSMState(
            deter=torch.stack([state.deter for state in states], dim=1),
            stoch=torch.stack([state.stoch for state in states], dim=1),
            logits=torch.stack([state.logits for state in states], dim=1),
        )
