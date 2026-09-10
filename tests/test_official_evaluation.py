from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mario_dreamer.official.evaluation import (
    EpisodeEvaluator,
    create_checkpoint_snapshot,
    resolve_elements_checkpoint,
    summarize_episodes,
)


def _transition(
    *,
    reward: float,
    x_pos: float,
    action: int,
    is_first: bool = False,
    is_last: bool = False,
    is_terminal: bool = False,
    flag_get: bool = False,
) -> dict:
    return {
        "reward": np.float32(reward),
        "log/x_pos": np.float32(x_pos),
        "log/flag_get": np.float32(flag_get),
        "action": np.int32(action),
        "is_first": np.bool_(is_first),
        "is_last": np.bool_(is_last),
        "is_terminal": np.bool_(is_terminal),
    }


def test_episode_evaluator_records_exact_metrics(tmp_path: Path) -> None:
    evaluator = EpisodeEvaluator(
        tmp_path / "eval",
        episode_count=1,
        action_count=5,
        action_names=["NOOP", "right", "right+A", "right+B", "right+A+B"],
        video_episodes=0,
    )
    evaluator(
        _transition(reward=0, x_pos=40, action=2, is_first=True),
        worker=0,
    )
    evaluator(_transition(reward=1, x_pos=200, action=4), worker=0)
    evaluator(
        _transition(
            reward=15,
            x_pos=3161,
            action=0,
            is_last=True,
            is_terminal=True,
            flag_get=True,
        ),
        worker=0,
    )

    rows, summary = evaluator.finish()

    assert rows[0]["return"] == 16.0
    assert rows[0]["length"] == 2
    assert rows[0]["max_x"] == 3161.0
    assert rows[0]["completed"] is True
    assert rows[0]["terminal"] is True
    assert rows[0]["action_counts"] == [0, 0, 1, 0, 1]
    assert summary["completions"] == 1
    assert summary["completion_rate"] == 1.0
    assert summary["action_names"][2] == "right+A"
    assert Path(summary["episodes_path"]).exists()
    assert Path(summary["summary_path"]).exists()


def test_episode_summary_aggregates_rows() -> None:
    summary = summarize_episodes(
        [
            {
                "return": 10,
                "length": 4,
                "max_x": 100,
                "completed": False,
                "action_counts": [1, 3],
            },
            {
                "return": 30,
                "length": 8,
                "max_x": 300,
                "completed": True,
                "action_counts": [2, 2],
            },
        ]
    )

    assert summary["episodes"] == 2
    assert summary["completion_rate"] == 0.5
    assert summary["return_mean"] == 20.0
    assert summary["length_median"] == 6.0
    assert summary["max_x_best"] == 300.0
    assert summary["action_counts"] == [3, 5]


def test_checkpoint_snapshot_is_immutable_and_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "ckpt"
    save = checkpoint / "20260811T120000-000000100000"
    save.mkdir(parents=True)
    (save / "agent-0000.pkl").write_bytes(b"original")
    (save / "done").write_bytes(b"")
    (checkpoint / "latest").write_text(save.name, encoding="utf-8")

    snapshot, created = create_checkpoint_snapshot(run_dir, "agent-100k")
    assert created is True
    assert snapshot.name == save.name
    assert (snapshot / "agent-0000.pkl").read_bytes() == b"original"

    (save / "agent-0000.pkl").write_bytes(b"new-latest")
    same_snapshot, created = create_checkpoint_snapshot(run_dir, "agent-100k")
    assert created is False
    assert same_snapshot == snapshot
    assert (same_snapshot / "agent-0000.pkl").read_bytes() == b"original"
    metadata = json.loads((snapshot.parent.parent / "metadata.json").read_text())
    assert metadata["snapshot_checkpoint"] == str(snapshot)


def test_checkpoint_resolver_accepts_direct_completed_save(tmp_path: Path) -> None:
    completed = tmp_path / "save"
    completed.mkdir()
    (completed / "done").write_bytes(b"")

    assert resolve_elements_checkpoint(completed) == completed
