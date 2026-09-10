from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mario_dreamer.official.paired_diagnostic import (
    build_report_batch,
    compose_paired_frame,
    extract_openloop_predictions,
    forecast_window_starts,
    frame_metadata,
)


def _transitions(length: int) -> list[dict]:
    return [
        {
            "image": np.full((4, 4, 3), index, np.uint8),
            "reward": np.float32(index),
            "is_first": np.bool_(index == 0),
            "is_last": np.bool_(False),
            "is_terminal": np.bool_(False),
            "action": np.int32(index % 5),
            "log/x_pos": np.float32(40 + index),
            "log/flag_get": np.float32(False),
        }
        for index in range(length)
    ]


def test_forecast_windows_are_non_overlapping_and_boundary_safe() -> None:
    transitions = _transitions(13)
    assert forecast_window_starts(
        transitions, context_steps=3, horizon_steps=2
    ) == [0, 2, 4, 6, 8]

    transitions[7]["is_first"] = np.bool_(True)
    assert forecast_window_starts(
        transitions, context_steps=3, horizon_steps=2
    ) == [0, 2, 8]


def test_report_batch_has_expected_batch_and_time_axes() -> None:
    spaces = {
        "image": SimpleNamespace(dtype=np.uint8, shape=(4, 4, 3)),
        "reward": SimpleNamespace(dtype=np.float32, shape=()),
        "is_first": SimpleNamespace(dtype=np.bool_, shape=()),
        "action": SimpleNamespace(dtype=np.int32, shape=()),
        "consec": SimpleNamespace(dtype=np.int32, shape=()),
        "stepid": SimpleNamespace(dtype=np.uint8, shape=(20,)),
    }
    batch = build_report_batch(_transitions(5), start=1, length=3, spaces=spaces)
    assert batch["image"].shape == (1, 3, 4, 4, 3)
    assert batch["reward"].shape == (1, 3)
    assert batch["stepid"].shape == (1, 3, 20)
    assert batch["image"][0, 0, 0, 0, 0] == 1
    assert batch["is_first"][0].tolist() == [True, False, False]


def test_prediction_panel_extraction_discards_true_error_and_border() -> None:
    size = 4
    video = np.zeros((6, 3 * size + 4, size + 4, 3), np.uint8)
    video[:, 2 + size : 2 + 2 * size, 2 : 2 + size] = 73
    predictions = extract_openloop_predictions(
        video, context_steps=3, horizon_steps=3, image_size=size
    )
    assert predictions.shape == (3, size, size, 3)
    assert np.all(predictions == 73)


def test_composite_and_metadata_share_episode_clock() -> None:
    prediction = np.full((64, 64, 3), 20, np.uint8)
    actual = np.full((240, 256, 3), 200, np.uint8)
    paired = compose_paired_frame(prediction, actual)
    assert paired.shape == (256, 512, 3)
    assert np.all(paired[:, :256] == 20)
    assert np.all(paired[8:248, 256:] == 200)

    transitions = _transitions(10)
    rows = frame_metadata(
        transitions,
        window_start=0,
        context_steps=3,
        horizon_steps=2,
        output_offset=0,
    )
    assert rows[0]["episode_step"] == 3
    assert rows[0]["conditioning_action_step"] == 2
    assert rows[0]["conditioning_action"] == 2
    assert rows[0]["flag_get"] is False
    assert rows[1]["episode_step"] == 4
