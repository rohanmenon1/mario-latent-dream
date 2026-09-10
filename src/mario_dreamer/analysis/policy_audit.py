from __future__ import annotations

import json
import math
from pathlib import Path


def finite_horizon_replay_returns(rewards, continues, is_first, *, discount: float):
    """Compute reset-masked returns over a replay sequence without bootstrapping."""

    import torch

    targets = torch.empty_like(rewards[:, :-1])
    target = torch.zeros_like(rewards[:, -1])
    for index in reversed(range(rewards.shape[1] - 1)):
        transition_valid = 1.0 - is_first[:, index + 1]
        transition_continue = continues[:, index + 1] * transition_valid
        target = (
            rewards[:, index + 1] * transition_valid
            + discount * transition_continue * target
        )
        targets[:, index] = target
    return targets


def history_metrics(metrics_path: Path) -> dict[str, float | str]:
    """Summarize de-duplicated training and evaluation rows from JSONL metrics."""

    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return {"history_status": "missing"}

    rows_by_env_step = {}
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("phase") == "train" and "env_steps" in row:
                rows_by_env_step[int(float(row["env_steps"]))] = row

    rows = [rows_by_env_step[key] for key in sorted(rows_by_env_step)]
    eval_rows = [row for row in rows if row.get("evaluated") and "eval_max_x" in row]
    if not rows:
        return {"history_status": "empty"}

    result: dict[str, float | str] = {
        "history_status": "ok",
        "history_unique_train_rows": float(len(rows)),
        "history_eval_rows": float(len(eval_rows)),
        "history_return_scale_first": float(rows[0].get("return_scale", math.nan)),
        "history_return_scale_last": float(rows[-1].get("return_scale", math.nan)),
        "history_critic_abs_error_first": float(rows[0].get("critic_abs_error", math.nan)),
        "history_critic_abs_error_last": float(rows[-1].get("critic_abs_error", math.nan)),
    }
    if eval_rows:
        best = max(eval_rows, key=lambda row: float(row["eval_max_x"]))
        latest = eval_rows[-1]
        result.update(
            {
                "history_best_eval_env_steps": float(best["env_steps"]),
                "history_best_eval_max_x": float(best["eval_max_x"]),
                "history_best_eval_return": float(best.get("eval_return", math.nan)),
                "history_latest_eval_env_steps": float(latest["env_steps"]),
                "history_latest_eval_max_x": float(latest["eval_max_x"]),
                "history_latest_eval_return": float(latest.get("eval_return", math.nan)),
                "history_eval_table_json": json.dumps(
                    [
                        {
                            "env_steps": row.get("env_steps"),
                            "max_x": row.get("eval_max_x"),
                            "return": row.get("eval_return"),
                            "completion": row.get("eval_completion_rate"),
                            "return_scale": row.get("return_scale"),
                            "critic_error": row.get("critic_abs_error"),
                        }
                        for row in eval_rows
                    ],
                    separators=(",", ":"),
                ),
            }
        )
    return result

