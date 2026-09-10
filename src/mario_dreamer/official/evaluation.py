from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeEvaluator:
    """Collect exact episode metrics and optional raw-frame gameplay videos."""

    def __init__(
        self,
        output_dir: Path,
        episode_count: int,
        action_count: int,
        action_names: list[str] | None = None,
        video_episodes: int = 3,
        fps: int = 15,
    ) -> None:
        if episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if not 0 <= video_episodes <= episode_count:
            raise ValueError("video_episodes must be between zero and episode_count")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.episode_count = int(episode_count)
        self.action_count = int(action_count)
        if action_names is not None and len(action_names) != action_count:
            raise ValueError("action_names must contain one name per action")
        self.action_names = list(action_names) if action_names else None
        self.video_episodes = int(video_episodes)
        self.fps = int(fps)
        self.rows: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._writer = None
        self._video_path: Path | None = None

    def __call__(self, transition: dict[str, Any], worker: int) -> None:
        if worker != 0:
            raise RuntimeError("official evaluation expects exactly one environment")
        if bool(transition["is_first"]):
            self._start_episode()
        if self._current is None:
            raise RuntimeError("received a transition before an episode reset")

        self._current["return"] += float(transition["reward"])
        self._current["max_x"] = max(
            self._current["max_x"], float(transition["log/x_pos"])
        )
        self._current["completed"] = bool(
            self._current["completed"] or transition["log/flag_get"]
        )
        if not bool(transition["is_first"]):
            self._current["length"] += 1
        if not bool(transition["is_last"]):
            action = int(transition["action"])
            if not 0 <= action < self.action_count:
                raise ValueError(f"action {action} is outside the action space")
            self._current["action_counts"][action] += 1
        if self._writer is not None:
            self._writer.append_data(np.asarray(transition["log/render"]))

        if bool(transition["is_last"]):
            self._current["terminal"] = bool(transition["is_terminal"])
            self._finish_episode()

    def finish(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.close()
        if len(self.rows) != self.episode_count:
            raise RuntimeError(
                f"expected {self.episode_count} episodes, recorded {len(self.rows)}"
            )
        if self._current is not None:
            raise RuntimeError("evaluation ended in the middle of an episode")
        episodes_path = self.output_dir / "episodes.jsonl"
        with episodes_path.open("w", encoding="utf-8") as file:
            for row in self.rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
        summary = summarize_episodes(self.rows)
        if self.action_names:
            summary["action_names"] = self.action_names
        summary["video_paths"] = [
            row["video_path"] for row in self.rows if "video_path" in row
        ]
        summary["episodes_path"] = str(episodes_path)
        summary_path = self.output_dir / "summary.json"
        summary["summary_path"] = str(summary_path)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        return self.rows, summary

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _start_episode(self) -> None:
        if self._current is not None:
            raise RuntimeError("new episode started before the previous episode ended")
        index = len(self.rows) + 1
        self._current = {
            "episode": index,
            "return": 0.0,
            "length": 0,
            "max_x": 0.0,
            "completed": False,
            "terminal": False,
            "action_counts": [0] * self.action_count,
        }
        if index <= self.video_episodes:
            import imageio.v2 as imageio

            self._video_path = self.output_dir / f"gameplay_episode_{index:02d}.mp4"
            self._writer = imageio.get_writer(
                self._video_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )

    def _finish_episode(self) -> None:
        assert self._current is not None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._current["video_path"] = str(self._video_path)
        self.rows.append(self._current)
        self._current = None
        self._video_path = None


def summarize_episodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    returns = [float(row["return"]) for row in rows]
    lengths = [float(row["length"]) for row in rows]
    max_x = [float(row["max_x"]) for row in rows]
    completions = sum(bool(row["completed"]) for row in rows)
    action_counts = np.asarray([row["action_counts"] for row in rows]).sum(0)
    action_total = int(action_counts.sum())
    return {
        "episodes": len(rows),
        "completions": completions,
        "completion_rate": completions / len(rows),
        "return_mean": statistics.fmean(returns),
        "return_median": statistics.median(returns),
        "return_best": max(returns),
        "length_mean": statistics.fmean(lengths),
        "length_median": statistics.median(lengths),
        "length_best": max(lengths),
        "max_x_mean": statistics.fmean(max_x),
        "max_x_median": statistics.median(max_x),
        "max_x_best": max(max_x),
        "action_counts": action_counts.astype(int).tolist(),
        "action_fractions": (
            (action_counts / action_total).tolist()
            if action_total
            else [0.0] * len(action_counts)
        ),
    }


def create_checkpoint_snapshot(
    run_dir: Path,
    snapshot_name: str,
    *,
    source_name: str = "ckpt",
) -> tuple[Path, bool]:
    """Atomically preserve one checkpoint without overwriting an existing snapshot."""

    run_dir = Path(run_dir)
    source = run_dir / source_name
    if not source.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    source_save = resolve_elements_checkpoint(source)
    if not snapshot_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in snapshot_name
    ):
        raise ValueError(
            "snapshot_name may contain only letters, numbers, hyphens, and underscores"
        )

    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshots_dir / snapshot_name
    checkpoint = snapshot_dir / "ckpt"
    metadata_path = snapshot_dir / "metadata.json"
    if snapshot_dir.exists():
        if not checkpoint.exists() or not metadata_path.exists():
            raise RuntimeError(f"snapshot is incomplete and will not be overwritten: {snapshot_dir}")
        return resolve_elements_checkpoint(checkpoint), False

    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_name}-", dir=snapshots_dir))
    try:
        temporary_checkpoint = temporary / "ckpt"
        if source.is_dir():
            shutil.copytree(source, temporary_checkpoint)
        else:
            shutil.copy2(source, temporary_checkpoint)
        metadata = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_checkpoint": str(source),
            "source_save": str(source_save),
            "snapshot_checkpoint": str(checkpoint / source_save.name),
        }
        with (temporary / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        os.replace(temporary, snapshot_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return resolve_elements_checkpoint(checkpoint), True


def resolve_elements_checkpoint(path: Path) -> Path:
    """Resolve an Elements checkpoint index to its latest completed save."""

    path = Path(path)
    if (path / "done").is_file():
        return path
    latest_path = path / "latest"
    if not latest_path.is_file():
        raise RuntimeError(f"checkpoint has neither a done marker nor latest pointer: {path}")
    latest = latest_path.read_text(encoding="utf-8").strip()
    if not latest or Path(latest).name != latest:
        raise RuntimeError(f"checkpoint latest pointer is invalid: {latest!r}")
    completed_save = path / latest
    if not (completed_save / "done").is_file():
        raise RuntimeError(f"checkpoint latest save is incomplete: {completed_save}")
    return completed_save
