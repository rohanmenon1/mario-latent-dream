from __future__ import annotations

import json

import torch

from mario_dreamer.analysis import finite_horizon_replay_returns, history_metrics


def test_finite_horizon_returns_use_next_state_rewards_without_circular_bootstrap() -> None:
    rewards = torch.tensor([[0.0, 1.0, 2.0]])
    continues = torch.ones_like(rewards)
    is_first = torch.zeros_like(rewards)

    targets = finite_horizon_replay_returns(
        rewards,
        continues,
        is_first,
        discount=0.5,
    )

    expected_last = 2.0
    expected_first = 1.0 + 0.5 * expected_last
    assert torch.allclose(targets, torch.tensor([[expected_first, expected_last]]))


def test_finite_horizon_returns_do_not_cross_replay_resets() -> None:
    rewards = torch.tensor([[0.0, 1.0, 100.0]])
    continues = torch.ones_like(rewards)
    is_first = torch.tensor([[1.0, 0.0, 1.0]])

    targets = finite_horizon_replay_returns(
        rewards,
        continues,
        is_first,
        discount=0.5,
    )

    assert torch.allclose(targets, torch.tensor([[1.0, 0.0]]))


def test_history_metrics_deduplicates_resumed_environment_steps(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "phase": "train",
            "env_steps": 500,
            "return_scale": 1.0,
            "critic_abs_error": 2.0,
            "evaluated": "True",
            "eval_max_x": 100.0,
            "eval_return": 50.0,
        },
        {
            "phase": "train",
            "env_steps": 500,
            "return_scale": 1.5,
            "critic_abs_error": 1.5,
            "evaluated": "True",
            "eval_max_x": 120.0,
            "eval_return": 60.0,
        },
        {
            "phase": "train",
            "env_steps": 1000,
            "return_scale": 2.0,
            "critic_abs_error": 1.0,
            "evaluated": "False",
        },
    ]
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = history_metrics(metrics_path)

    assert result["history_unique_train_rows"] == 2.0
    assert result["history_eval_rows"] == 1.0
    assert result["history_best_eval_max_x"] == 120.0
    assert result["history_return_scale_first"] == 1.5
    assert result["history_return_scale_last"] == 2.0
