from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastFrame:
    output_index: int
    episode_step: int
    window_start: int
    context_steps: int
    horizon_step: int
    conditioning_action_step: int
    conditioning_action: int
    x_pos: float
    flag_get: bool


def forecast_window_starts(
    transitions: Sequence[Mapping[str, Any]],
    *,
    context_steps: int,
    horizon_steps: int,
) -> list[int]:
    """Return non-overlapping, boundary-safe forecast windows."""

    if context_steps <= 0 or horizon_steps <= 0:
        raise ValueError("context_steps and horizon_steps must be positive")
    length = context_steps + horizon_steps
    starts = []
    for start in range(0, len(transitions) - length + 1, horizon_steps):
        window = transitions[start : start + length]
        if any(bool(step["is_first"]) for step in window[1:]):
            continue
        if any(bool(step["is_last"]) for step in window[:-1]):
            continue
        starts.append(start)
    return starts


def build_report_batch(
    transitions: Sequence[Mapping[str, Any]],
    *,
    start: int,
    length: int,
    spaces: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Build one batch accepted by the upstream Agent.report method."""

    window = transitions[start : start + length]
    if len(window) != length:
        raise ValueError("the requested report window is incomplete")

    batch = {}
    for key, space in spaces.items():
        if key == "consec":
            values = np.zeros(length, dtype=space.dtype)
        elif key == "stepid":
            values = np.zeros((length, *space.shape), dtype=space.dtype)
        else:
            values = np.stack(
                [np.asarray(step[key], dtype=space.dtype) for step in window]
            )
            if key == "is_first":
                values = values.copy()
                values[0] = True
        batch[key] = values[None]
    return batch


def extract_openloop_predictions(
    report_video: np.ndarray,
    *,
    context_steps: int,
    horizon_steps: int,
    image_size: int,
) -> np.ndarray:
    """Extract the decoder panel and discard the true/error panels and border."""

    video = np.asarray(report_video, dtype=np.uint8)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"expected THWC report video, received {video.shape}")
    expected_height = 3 * image_size + 4
    expected_width = image_size + 4
    if video.shape[1:3] != (expected_height, expected_width):
        raise ValueError(
            "unexpected open-loop panel geometry: "
            f"expected {(expected_height, expected_width)}, got {video.shape[1:3]}"
        )
    prediction_y = 2 + image_size
    if video.shape[0] < context_steps + horizon_steps:
        raise ValueError("report video is shorter than the requested forecast")
    return video[
        context_steps : context_steps + horizon_steps,
        prediction_y : prediction_y + image_size,
        2 : 2 + image_size,
    ]


def compose_paired_frame(
    prediction: np.ndarray,
    actual: np.ndarray,
    *,
    panel_size: int = 256,
) -> np.ndarray:
    """Place nearest-neighbor prediction and crisp gameplay on one shared clock."""

    prediction = np.asarray(prediction, dtype=np.uint8)
    actual = np.asarray(actual, dtype=np.uint8)
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction must be an HWC RGB image")
    if actual.ndim != 3 or actual.shape[-1] != 3:
        raise ValueError("actual must be an HWC RGB image")

    import cv2

    dream = cv2.resize(
        prediction,
        (panel_size, panel_size),
        interpolation=cv2.INTER_NEAREST,
    )
    scale = min(panel_size / actual.shape[1], panel_size / actual.shape[0])
    width = max(1, round(actual.shape[1] * scale))
    height = max(1, round(actual.shape[0] * scale))
    gameplay = cv2.resize(actual, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    top = (panel_size - height) // 2
    left = (panel_size - width) // 2
    canvas[top : top + height, left : left + width] = gameplay
    return np.concatenate([dream, canvas], axis=1)


def frame_metadata(
    transitions: Sequence[Mapping[str, Any]],
    *,
    window_start: int,
    context_steps: int,
    horizon_steps: int,
    output_offset: int,
) -> list[dict[str, int | float]]:
    rows = []
    first_prediction = window_start + context_steps
    for horizon_index in range(horizon_steps):
        episode_step = first_prediction + horizon_index
        action_step = episode_step - 1
        row = ForecastFrame(
            output_index=output_offset + horizon_index,
            episode_step=episode_step,
            window_start=window_start,
            context_steps=context_steps,
            horizon_step=horizon_index + 1,
            conditioning_action_step=action_step,
            conditioning_action=int(transitions[action_step]["action"]),
            x_pos=float(transitions[episode_step]["log/x_pos"]),
            flag_get=bool(transitions[episode_step]["log/flag_get"]),
        )
        rows.append(asdict(row))
    return rows
